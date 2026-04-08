import torch
import torch.nn as nn

DINOV3_ARCHS = {
    'dinov3_vits16': 384,
    'dinov3_vits16plus': 384,
    'dinov3_vitb16': 768,
    'dinov3_vitl16': 1024,
    'dinov3_vitl16plus': 1024,
    'dinov3_vith16plus': 1280,
    'dinov3_vit7b16': 4096,
}


class DINOv3(nn.Module):
    """
    DINOv3 model

    Args:
        model_name (str): The model architecture name.
        num_trainable_blocks (int): Number of last blocks to train.
        norm_layer (bool): If True, applies model norm before outputs.
        return_token (bool): If True, returns both feature map and cls token.
        pretrained (bool): If True, loads pretrained backbone weights.
        weights (str): DINOv3 weights identifier or URL/path.
        check_hash (bool): Whether to validate checkpoint hash if supported.
        repo_or_dir (str): Torch Hub repository or local path.
        source (str): Torch Hub source ('github' or 'local').
    """

    def __init__(
            self,
            model_name='dinov3_vitb16',
            num_trainable_blocks=2,
            norm_layer=False,
            return_token=False,
            pretrained=True,
            weights='LVD1689M',
            check_hash=False,
            repo_or_dir='facebookresearch/dinov3',
            source='github',
        ):
        super().__init__()

        assert model_name in DINOV3_ARCHS.keys(), f'Unknown model name {model_name}'

        # DINOv3 hub code may import custom_fwd/custom_bwd from torch.amp,
        # but these symbols/signatures differ across PyTorch versions.
        if hasattr(torch, 'amp') and not hasattr(torch.amp, 'custom_fwd'):
            from torch.cuda.amp import custom_bwd as legacy_custom_bwd
            from torch.cuda.amp import custom_fwd as legacy_custom_fwd

            def compat_custom_fwd(*, cast_inputs=None, device_type=None):
                # Older torch.cuda.amp.custom_fwd has no device_type argument.
                return legacy_custom_fwd(cast_inputs=cast_inputs)

            def compat_custom_bwd(*, device_type=None):
                # Older torch.cuda.amp.custom_bwd has no device_type argument.
                def decorator(bwd):
                    return legacy_custom_bwd(bwd)

                return decorator

            torch.amp.custom_fwd = compat_custom_fwd
            torch.amp.custom_bwd = compat_custom_bwd

        # Some DINOv3 modules set torch._dynamo.config.accumulated_cache_size_limit,
        # which may not exist in older torch versions.
        if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'config'):
            cfg = torch._dynamo.config
            if not hasattr(cfg, 'accumulated_cache_size_limit'):
                fallback = getattr(cfg, 'cache_size_limit', 64)

                if hasattr(cfg, '_config') and isinstance(cfg._config, dict):
                    cfg._config['accumulated_cache_size_limit'] = fallback

                if hasattr(cfg, '_default') and isinstance(cfg._default, dict):
                    cfg._default['accumulated_cache_size_limit'] = fallback

                if hasattr(cfg, '_allowed_keys'):
                    try:
                        cfg._allowed_keys.add('accumulated_cache_size_limit')
                    except Exception:
                        pass

                try:
                    setattr(cfg, 'accumulated_cache_size_limit', fallback)
                except Exception:
                    # In some torch builds setting unknown config attrs is blocked.
                    pass

        self.model = torch.hub.load(
            repo_or_dir,
            model_name,
            source=source,
            pretrained=pretrained,
            weights=weights,
            check_hash=check_hash,
        )
        self.num_channels = DINOV3_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

        self._configure_trainable_layers()
        self._print_parameter_stats()

    def _configure_trainable_layers(self):
        blocks = self.model.blocks
        self._total_blocks = len(blocks)
        self._trainable_blocks = max(0, min(self.num_trainable_blocks, self._total_blocks))
        self._frozen_blocks = self._total_blocks - self._trainable_blocks

        # Start with everything frozen, then unfreeze the last N blocks.
        self.model.requires_grad_(False)

        if self._trainable_blocks > 0:
            for blk in blocks[-self._trainable_blocks:]:
                blk.requires_grad_(True)

        # If norm is used in forward, keep it trainable.
        if self.norm_layer and hasattr(self.model, 'norm'):
            self.model.norm.requires_grad_(True)

    def _print_parameter_stats(self):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        print(
            f'[DINOv3] trainable params: {trainable:,} | '
            f'non-trainable params: {non_trainable:,} | '
            f'trainable blocks: {self._trainable_blocks}/{self._total_blocks}'
        )

    def forward(self, x):
        """
        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by model patch size.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // p, W // p].
            t (torch.Tensor): The cls token [B, C]. This is only returned if return_token is True.
        """

        B, _, H, W = x.shape

        tokens_and_shape = self.model.prepare_tokens_with_masks(x)
        if isinstance(tokens_and_shape, tuple):
            x, (fh, fw) = tokens_and_shape
        else:
            x = tokens_and_shape
            patch_size = getattr(self.model, 'patch_size', 16)
            if isinstance(patch_size, tuple):
                patch_size = patch_size[0]
            fh, fw = H // patch_size, W // patch_size

        blocks = self.model.blocks
        frozen_blocks = self._frozen_blocks

        if frozen_blocks > 0:
            with torch.no_grad():
                for blk in blocks[:frozen_blocks]:
                    x = blk(x)
            x = x.detach()

        for blk in blocks[frozen_blocks:]:
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)

        t = x[:, 0]
        n_storage_tokens = getattr(self.model, 'n_storage_tokens', 0)
        f = x[:, 1 + n_storage_tokens:]
        f = f.reshape((B, fh, fw, self.num_channels)).permute(0, 3, 1, 2)

        if self.return_token:
            return f, t
        return f
