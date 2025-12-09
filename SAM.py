import torch
from torch import nn
import torch.nn.functional as F
from KDNet.modeling.image_encoder import ImageEncoderViT
from KDNet.modeling.mask_decoder import MaskDecoder
from KDNet.modeling.prompt_encoder import PromptEncoder
from KDNet.modeling.transformer import TwoWayTransformer
from typing import Any, Dict, List, Tuple
from functools import partial
import numpy as np
from torch.nn.functional import interpolate

# from KDNet.DINOv3 import get_model


class Up(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=True):
        super(Up, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class MDSAM(nn.Module):
    def __init__(self, img_size=256):
        super().__init__()

        self.pixel_mean: List[float] = [123.675, 116.28, 103.53]
        self.pixel_std: List[float] = [58.395, 57.12, 57.375]

        adapter_config = {
            'reduction_ratio': 4,
            'freq_scales': (1, 2, 4),
            'gate_threshold': 0.1
        }

        # 图像编码器
        self.image_encoder = ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=img_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256,
            # use_adapter=True,
            # adapter_config=adapter_config
        )

        image_embedding_size = img_size // 16
        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
        )

        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        # DINOv3 Teacher模型
        # self.dinov3 = get_model()

        self.out_sam = nn.Sequential(
            Up(scale_factor=4, mode='bilinear', align_corners=True)
        )

    def forward(self, img, dep):
        """
        Args:
            img: Input image tensor [B, C, H, W]
            box: Optional bounding box (如果提供,会覆盖DINOv3的预测)
            mask: Optional mask input (如果提供,会覆盖DINOv3的预测)

        Returns:
            masks: 预测的mask
            low_res_masks: 低分辨率mask (用于后续处理)
        """
        B = img.shape[0]
        
        # 图像编码
        rgb_list = self.image_encoder(img, dep)

        # mask = self.dinov3(img)

        # Prompt编码 - 现在box是正确的tensor格式
        sparse_prompt, dense_prompt = self.prompt_encoder(
            points=None,
            boxes=None,      # (B, 4) tensor
            masks=None      # (B, 1, H, W) tensor
        )

        # Mask解码
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=rgb_list[-1],
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_prompt,
            dense_prompt_embeddings=dense_prompt,
            multimask_output=False,
        )

        # 后处理到原始分辨率
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=img.shape[-2:],
            original_size=img.shape[-2:],
        )

        masks = F.sigmoid(masks)

        return masks

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )

        return masks


def reshapePos(pos_embed, img_size):
    token_size = int(img_size // 16)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
    return pos_embed


def reshapeRel(k, rel_pos_params, img_size):
    if not ('2' in k or '5' in k or '8' in k or '11' in k):
        return rel_pos_params

    token_size = int(img_size // 16)
    h, w = rel_pos_params.shape
    rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
    rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
    return rel_pos_params[0, 0, ...]


def load(net, ckpt, img_size):
    ckpt = torch.load(ckpt, map_location='cpu')
    from collections import OrderedDict
    dict = OrderedDict()

    for k, v in ckpt.items():
        if 'pe_layer' in k:
            dict[k[15:]] = v
            continue
        if 'pos_embed' in k:
            dict[k] = reshapePos(v, img_size)
            continue
        if 'rel_pos' in k:
            dict[k] = reshapeRel(k, v, img_size)
        elif "image_encoder" in k:
            if "neck" in k:
                # Add the original final neck layer to 3, 6, and 9, initialization is the same.
                for i in range(4):
                    new_key = "{}.{}{}".format(k[:18], i, k[18:])
                    dict[new_key] = v
            else:
                dict[k] = v

    # 加载权重
    state1, state2 = net.load_state_dict(dict, strict=False)
    print("Missing keys:", state1)
    print("Unexpected keys:", state2)

    return "TRUE"


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print("="*70)
    print("初始化MDSAM模型...")
    print("="*70)
    
    model = MDSAM(256).cuda()
    state = load(model,
                 "/media/tbb/shuju/Backbone_PTH/sam_vit_b_01ec64.pth", 256)
    
    print("\n" + "="*70)
    print("测试前向传播...")
    print("="*70)
    
    rgb = torch.randn(2, 3, 256, 256).cuda()
    dep = torch.randn(2, 3, 256, 256).cuda()


    with torch.no_grad():
        masks, low_res_masks = model(rgb, dep)
    print(f"✓ Masks shape: {masks.shape}")
    print(f"✓ Low-res masks shape: {low_res_masks.shape}")

    from thop import profile
    flops, params = profile(model, (rgb, dep))
    print(f"✓ FLOPs: {flops / 1e9:.2f} G")
    print(f"✓ Params: {params / 1e6:.2f} M")

    print("\n" + "="*70)
    print("✓ 所有测试通过!")
    print("="*70)
