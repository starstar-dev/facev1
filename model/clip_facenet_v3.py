"""
CLIP-FACENet v3.1: v2 + Token Scorer + FCE.
FCE: Feature Consistency Enhancement — explicit MSE loss pulling R/N features toward T.
Token Scorer: gated residual, alpha starts at 0.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel


def weights_init_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None: nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)


def weights_init_classifier(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None: nn.init.constant_(m.bias, 0.0)


class TokenScorer(nn.Module):
    """Gated token importance scoring. Starts as CLS-only (alpha=0)."""
    def __init__(self, dim=768):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(dim, dim // 4), nn.GELU(), nn.Linear(dim // 4, 1))
        self.alpha = nn.Parameter(torch.zeros(1))
        self.apply(weights_init_kaiming)

    def forward(self, tokens):
        scores = self.scorer(tokens).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        weighted = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return tokens[:, 0] + torch.tanh(self.alpha) * weighted


class CrossModalFusion(nn.Module):
    def __init__(self, dim=768, num_heads=8, dropout=0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(dim); self.norm_kv = nn.LayerNorm(dim); self.norm_out = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim), nn.Dropout(dropout))
        self.gate_attn = nn.Parameter(torch.zeros(1)); self.gate_ffn = nn.Parameter(torch.zeros(1))
        self.apply(weights_init_kaiming)

    def forward(self, feat_query, feat_key):
        identity = feat_query
        q = self.norm_q(feat_query); k = self.norm_kv(feat_key); v = self.norm_kv(feat_key)
        feat_query = identity + torch.tanh(self.gate_attn) * self.cross_attn(q, k, v)[0]
        identity2 = feat_query
        return identity2 + torch.tanh(self.gate_ffn) * self.ffn(self.norm_out(feat_query))


class CrossPatchAttention(nn.Module):
    def __init__(self, dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim), nn.Dropout(dropout))
        self.norm_out = nn.LayerNorm(dim)
        self.apply(weights_init_kaiming)

    def forward(self, feat_R_patches, feat_N_patches):
        q_R = self.norm1(feat_R_patches); k_N = self.norm2(feat_N_patches); v_N = self.norm2(feat_N_patches)
        R_out = feat_R_patches + self.cross_attn(q_R, k_N, v_N)[0]
        R_out = R_out + self.ffn(self.norm_out(R_out))
        q_N = self.norm1(feat_N_patches); k_R = self.norm2(feat_R_patches); v_R = self.norm2(feat_R_patches)
        N_out = feat_N_patches + self.cross_attn(q_N, k_R, v_R)[0]
        N_out = N_out + self.ffn(self.norm_out(N_out))
        return R_out, N_out


class CLIPBackbone(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch16'):
        super().__init__()
        self.vision = CLIPVisionModel.from_pretrained(model_name)
        self.dim = self.vision.config.hidden_size

    def forward(self, x):
        out = self.vision(x, output_hidden_states=False)
        return out.last_hidden_state


class CLIPFACENetV3(nn.Module):
    """CLIP-FACENet v3.1: v2 + Token Scorer + FCE."""
    def __init__(self, num_classes, camera_num=0, view_num=0, cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.dim = 768

        print('Loading CLIP ViT-B/16 backbones...')
        self.backbone_rgb = CLIPBackbone()
        self.backbone_ni  = CLIPBackbone()
        self.backbone_ti  = CLIPBackbone()

        self.fusion_R = CrossModalFusion(dim=self.dim)
        self.fusion_N = CrossModalFusion(dim=self.dim)
        self.cross_patch_attn = CrossPatchAttention(dim=self.dim)

        self.token_scorer_R = TokenScorer(dim=self.dim)
        self.token_scorer_N = TokenScorer(dim=self.dim)
        self.token_scorer_T = TokenScorer(dim=self.dim)

        self.bottleneck_R = nn.BatchNorm1d(self.dim); self.bottleneck_R.bias.requires_grad_(False); self.bottleneck_R.apply(weights_init_kaiming)
        self.bottleneck_N = nn.BatchNorm1d(self.dim); self.bottleneck_N.bias.requires_grad_(False); self.bottleneck_N.apply(weights_init_kaiming)
        self.bottleneck_T = nn.BatchNorm1d(self.dim); self.bottleneck_T.bias.requires_grad_(False); self.bottleneck_T.apply(weights_init_kaiming)

        self.classifier_R = nn.Linear(self.dim, num_classes, bias=False); self.classifier_R.apply(weights_init_classifier)
        self.classifier_N = nn.Linear(self.dim, num_classes, bias=False); self.classifier_N.apply(weights_init_classifier)
        self.classifier_T = nn.Linear(self.dim, num_classes, bias=False); self.classifier_T.apply(weights_init_classifier)

        self.bottleneck_R_label = nn.BatchNorm1d(self.dim); self.bottleneck_R_label.bias.requires_grad_(False); self.bottleneck_R_label.apply(weights_init_kaiming)
        self.classifier_R_label = nn.Linear(self.dim, num_classes, bias=False); self.classifier_R_label.apply(weights_init_classifier)
        self.bottleneck_N_label = nn.BatchNorm1d(self.dim); self.bottleneck_N_label.bias.requires_grad_(False); self.bottleneck_N_label.apply(weights_init_kaiming)
        self.classifier_N_label = nn.Linear(self.dim, num_classes, bias=False); self.classifier_N_label.apply(weights_init_classifier)

        self.use_mamba_enhance = False
        self.use_mcloss = True
        self.use_fce = True   # v3.1: enable FCE
        self.use_mfmp = True
        self.num_parts = 0

        n_params = sum(p.numel() for p in self.parameters())
        print(f'CLIP-FACENet v3.1 built (TokenScorer + FCE). Params: {n_params/1e6:.1f}M')

    def get_cls_feat_mfmp(self, featR, featN):
        tokenR = featR[:, 0:1]; tokenN = featN[:, 0:1]
        patchesR = featR[:, 1:]; patchesN = featN[:, 1:]
        patchesR, patchesN = self.cross_patch_attn(patchesR, patchesN)
        return torch.cat([tokenR, patchesR], dim=1), torch.cat([tokenN, patchesN], dim=1)

    def forward(self, x1, x2, x3, label=None, cam_label=None, view_label=None, flare_label=None):
        featR = self.backbone_rgb(x1)
        featN = self.backbone_ni(x2)
        featT = self.backbone_ti(x3)

        if self.use_mfmp and self.training:
            featR_mfmp, featN_mfmp = self.get_cls_feat_mfmp(featR, featN)
        else:
            featR_mfmp, featN_mfmp = featR, featN

        featR = self.fusion_R(featR, featT)
        featN = self.fusion_N(featN, featT)

        global_R = self.token_scorer_R(featR)
        global_N = self.token_scorer_N(featN)
        global_T = self.token_scorer_T(featT)

        if self.training:
            bn_R = self.bottleneck_R(global_R)
            bn_N = self.bottleneck_N(global_N)
            bn_T = self.bottleneck_T(global_T)

            score_R = self.classifier_R(bn_R)
            score_N = self.classifier_N(bn_N)
            score_T = self.classifier_T(bn_T)

            # FCE features: BN outputs for MSE loss (v3.1)
            fce_feats = (bn_R, bn_N, bn_T) if self.use_fce else None

            if self.use_mfmp:
                cls_R_mfmp = featR_mfmp[:, 0]; cls_N_mfmp = featN_mfmp[:, 0]
                bn_R_label = self.bottleneck_R_label(cls_R_mfmp)
                bn_N_label = self.bottleneck_N_label(cls_N_mfmp)
                score_R_label = self.classifier_R_label(bn_R_label)
                score_N_label = self.classifier_N_label(bn_N_label)
                return ([score_R], [global_R]), ([score_N], [global_N]), ([score_T], [global_T]), \
                       cls_R_mfmp, score_R_label, cls_N_mfmp, score_N_label, fce_feats

            return ([score_R], [global_R]), ([score_N], [global_N]), ([score_T], [global_T]), \
                   None, None, None, None, fce_feats
        else:
            bn_R = self.bottleneck_R(global_R)
            bn_N = self.bottleneck_N(global_N)
            bn_T = self.bottleneck_T(global_T)
            return torch.cat([bn_R, bn_N, bn_T], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        self.load_state_dict(param_dict, strict=False)
        print(f'Loaded pretrained model from {trained_path}')
