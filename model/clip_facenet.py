"""
CLIP-FACENet v2: CLIP ViT + Cross-Attention Fusion + MFMP + MC Loss.

Key components:
1. CLIP ViT-B/16 backbone (400M image-text pretrained)
2. Learnable cross-attention fusion (TI → RGB/NI, replaces hard FCE)
3. MFMP: Cross-attention between RGB and NI patch tokens + KL alignment
4. MC Loss: Modality consistency via KL divergence on classification scores
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPVisionConfig


def weights_init_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def weights_init_classifier(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class CrossModalFusion(nn.Module):
    """
    Learnable cross-modal fusion using multi-head cross-attention.
    Q (RGB or NI) attends to K/V (TI — thermal is clean, no flare).
    Gated residual: output = input + tanh(scale) * attention_output
    """
    def __init__(self, dim=768, num_heads=8, dropout=0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        
        self.gate_attn = nn.Parameter(torch.zeros(1))
        self.gate_ffn = nn.Parameter(torch.zeros(1))
        
        self.apply(weights_init_kaiming)
    
    def forward(self, feat_query, feat_key):
        identity = feat_query
        q = self.norm_q(feat_query)
        k = self.norm_kv(feat_key)
        v = self.norm_kv(feat_key)
        attended, _ = self.cross_attn(q, k, v)
        feat_query = identity + torch.tanh(self.gate_attn) * attended
        identity2 = feat_query
        feat_query = identity2 + torch.tanh(self.gate_ffn) * self.ffn(self.norm_out(feat_query))
        return feat_query


class CrossPatchAttention(nn.Module):
    """
    Cross-attention between RGB and NI patch tokens (excluding CLS).
    Used for MFMP: aligns R and N feature representations.
    Matching original FACENet's get_cls_feat / CrossAttentionModuleViT.
    """
    def __init__(self, dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        # FFN for post-attention refinement
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm_out = nn.LayerNorm(dim)
        self.apply(weights_init_kaiming)
    
    def forward(self, feat_R_patches, feat_N_patches):
        """
        Args:
            feat_R_patches: (B, N-1, D) — RGB patch tokens
            feat_N_patches: (B, N-1, D) — NI patch tokens
        Returns:
            R', N' — cross-attended patch tokens
        """
        # R → attends to N
        q_R = self.norm1(feat_R_patches)
        k_N = self.norm2(feat_N_patches)
        v_N = self.norm2(feat_N_patches)
        R_attended, _ = self.cross_attn(q_R, k_N, v_N)
        R_out = feat_R_patches + R_attended
        R_out = R_out + self.ffn(self.norm_out(R_out))
        
        # N → attends to R
        q_N = self.norm1(feat_N_patches)
        k_R = self.norm2(feat_R_patches)
        v_R = self.norm2(feat_R_patches)
        N_attended, _ = self.cross_attn(q_N, k_R, v_R)
        N_out = feat_N_patches + N_attended
        N_out = N_out + self.ffn(self.norm_out(N_out))
        
        return R_out, N_out


class CLIPBackbone(nn.Module):
    """Wrapper around CLIPVisionModel for ReID feature extraction."""
    def __init__(self, model_name='openai/clip-vit-base-patch16', freeze_blocks=0):
        super().__init__()
        self.vision = CLIPVisionModel.from_pretrained(model_name)
        self.config = self.vision.config
        self.dim = self.config.hidden_size
    
    def forward(self, x):
        out = self.vision(x, output_hidden_states=False)
        return out.last_hidden_state  # (B, 197, 768)


class CLIPFACENet(nn.Module):
    """
    CLIP-FACENet v2: CLIP ViT + Cross-Attention Fusion + MFMP + MC Loss.
    
    Architecture:
        RGB ──→ CLIP ViT ──→ featR ──→ CrossAttn(Q=R, KV=T) ──→ BN → clsR
        NI  ──→ CLIP ViT ──→ featN ──→ CrossAttn(Q=N, KV=T) ──→ BN → clsN
        TI  ──→ CLIP ViT ──→ featT ──→ (unchanged) ────────────→ BN → clsT
        
        MFMP: featR/Rpatches ←──CrossAttn──→ featN/Npatches → label cls → KL loss
        MC Loss: KL(clsR, clsN) + KL(clsT, clsN)
    """
    def __init__(self, num_classes, camera_num=0, view_num=0, cfg=None):
        super().__init__()
        
        self.num_classes = num_classes
        self.dim = 768  # CLIP ViT-B/16
        
        print('Loading CLIP ViT-B/16 backbones...')
        self.backbone_rgb = CLIPBackbone()
        self.backbone_ni  = CLIPBackbone()
        self.backbone_ti  = CLIPBackbone()
        print(f'CLIP backbones loaded. Total params: {sum(p.numel() for p in self.parameters())/1e6:.1f}M')
        
        # Cross-modal fusion (TI → RGB/NI)
        self.fusion_R = CrossModalFusion(dim=self.dim)
        self.fusion_N = CrossModalFusion(dim=self.dim)
        
        # MFMP: cross-attention between R and N patches
        self.cross_patch_attn = CrossPatchAttention(dim=self.dim)
        
        # BNNeck for each modality
        self.bottleneck_R = nn.BatchNorm1d(self.dim)
        self.bottleneck_R.bias.requires_grad_(False)
        self.bottleneck_R.apply(weights_init_kaiming)
        self.bottleneck_N = nn.BatchNorm1d(self.dim)
        self.bottleneck_N.bias.requires_grad_(False)
        self.bottleneck_N.apply(weights_init_kaiming)
        self.bottleneck_T = nn.BatchNorm1d(self.dim)
        self.bottleneck_T.bias.requires_grad_(False)
        self.bottleneck_T.apply(weights_init_kaiming)
        
        # Classifiers
        self.classifier_R = nn.Linear(self.dim, num_classes, bias=False)
        self.classifier_R.apply(weights_init_classifier)
        self.classifier_N = nn.Linear(self.dim, num_classes, bias=False)
        self.classifier_N.apply(weights_init_classifier)
        self.classifier_T = nn.Linear(self.dim, num_classes, bias=False)
        self.classifier_T.apply(weights_init_classifier)
        
        # MFMP label classifiers (separate branches for R and N)
        # These mirror the original FACENet b_R_label / b_N_label structure
        self.bottleneck_R_label = nn.BatchNorm1d(self.dim)
        self.bottleneck_R_label.bias.requires_grad_(False)
        self.bottleneck_R_label.apply(weights_init_kaiming)
        self.classifier_R_label = nn.Linear(self.dim, num_classes, bias=False)
        self.classifier_R_label.apply(weights_init_classifier)
        
        self.bottleneck_N_label = nn.BatchNorm1d(self.dim)
        self.bottleneck_N_label.bias.requires_grad_(False)
        self.bottleneck_N_label.apply(weights_init_kaiming)
        self.classifier_N_label = nn.Linear(self.dim, num_classes, bias=False)
        self.classifier_N_label.apply(weights_init_classifier)
        
        # Flags for processor compatibility
        self.use_mamba_enhance = False
        self.use_mcloss = False
        self.use_fce = False
        self.use_mfmp = False
    
    def get_cls_feat_mfmp(self, featR, featN):
        """
        Cross-attention between RGB and NI patch tokens for MFMP.
        Matches original FACENet's get_cls_feat behavior.
        """
        tokenR = featR[:, 0:1]   # (B, 1, D) — CLS
        tokenN = featN[:, 0:1]
        
        patchesR = featR[:, 1:]  # (B, 196, D) — patches
        patchesN = featN[:, 1:]
        
        patchesR, patchesN = self.cross_patch_attn(patchesR, patchesN)
        
        # Reassemble: CLS + cross-attended patches
        outR = torch.cat([tokenR, patchesR], dim=1)
        outN = torch.cat([tokenN, patchesN], dim=1)
        
        return outR, outN
    
    def forward(self, x1, x2, x3, label=None, cam_label=None, view_label=None, flare_label=None):
        """
        Args:
            x1, x2, x3: (B, 3, 224, 224) — RGB, NI, TI (CLIP preprocessed)
        Returns:
            Training: (score_R, feat), (score_N, feat), (score_T, feat), RFeat, scoreR_label, NFeat, scoreN_label
            Inference: torch.cat([bn_R, bn_N, bn_T], dim=1)
        """
        # 1. Extract CLIP features
        featR = self.backbone_rgb(x1)  # (B, 197, 768)
        featN = self.backbone_ni(x2)
        featT = self.backbone_ti(x3)
        
        # 2. MFMP: cross-attention between R and N patches (before fusion)
        if self.use_mfmp and self.training:
            featR_mfmp, featN_mfmp = self.get_cls_feat_mfmp(featR, featN)
        else:
            featR_mfmp, featN_mfmp = featR, featN
        
        # 3. Cross-modal fusion: repair RGB and NI using TI
        featR = self.fusion_R(featR, featT)
        featN = self.fusion_N(featN, featT)
        
        # 4. Extract CLS tokens
        cls_R = featR[:, 0]  # (B, 768)
        cls_N = featN[:, 0]
        cls_T = featT[:, 0]
        
        if self.training:
            bn_R = self.bottleneck_R(cls_R)
            bn_N = self.bottleneck_N(cls_N)
            bn_T = self.bottleneck_T(cls_T)
            
            score_R = self.classifier_R(bn_R)
            score_N = self.classifier_N(bn_N)
            score_T = self.classifier_T(bn_T)
            
            # MFMP: label classification for R and N cross-attended features
            if self.use_mfmp:
                cls_R_mfmp = featR_mfmp[:, 0]
                cls_N_mfmp = featN_mfmp[:, 0]
                
                bn_R_label = self.bottleneck_R_label(cls_R_mfmp)
                bn_N_label = self.bottleneck_N_label(cls_N_mfmp)
                
                score_R_label = self.classifier_R_label(bn_R_label)
                score_N_label = self.classifier_N_label(bn_N_label)
                
                return ([score_R], [cls_R]), ([score_N], [cls_N]), ([score_T], [cls_T]), \
                       cls_R_mfmp, score_R_label, cls_N_mfmp, score_N_label
            
            return ([score_R], [cls_R]), ([score_N], [cls_N]), ([score_T], [cls_T]), \
                   None, None, None, None
        else:
            bn_R = self.bottleneck_R(cls_R)
            bn_N = self.bottleneck_N(cls_N)
            bn_T = self.bottleneck_T(cls_T)
            return torch.cat([bn_R, bn_N, bn_T], dim=1)
    
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        self.load_state_dict(param_dict, strict=False)
        print(f'Loaded pretrained model from {trained_path}')
