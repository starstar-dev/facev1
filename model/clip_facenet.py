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
import transformers.modeling_utils as _tf_modeling_utils
_tf_modeling_utils.check_torch_load_is_safe = lambda: None  # only for trusted local CLIP official weight
from model.coen_lite import compute_combined_quality, patch_quality_to_scalar


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
    Tri-modal cross-attention fusion for flare repair.
    
    Two reference sources for each modality being repaired:
    1. TI (thermal): always clean, never affected by flare — the anchor
    2. Peer (N→R or R→N): often less damaged than self
    
    Quality-aware per-sample gating:
    - q_self low → borrow more from TI (always available)
    - q_peer > q_self → also borrow from peer (it's cleaner)
    
    Example: R has flare (q_R=0.3), N is clean (q_N=0.9)
    → R borrows TI at 1.7x AND N at 0.6x
    → N borrows TI at 1.1x only, ignores damaged R
    """
    def __init__(self, dim=768, num_heads=8, dropout=0.0):
        super().__init__()
        # TI reference — the anchor, always clean
        self.cross_attn_T = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        # Peer modality reference — borrow if peer > self
        self.cross_attn_peer = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv_T = nn.LayerNorm(dim)
        self.norm_kv_peer = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        
        self.gate_T = nn.Parameter(torch.zeros(1))
        self.gate_peer = nn.Parameter(torch.zeros(1))
        self.gate_ffn = nn.Parameter(torch.zeros(1))
        
        self.apply(weights_init_kaiming)
    
    def forward(self, feat_query, feat_T, feat_peer, q_self=None, q_peer=None):
        """
        Args:
            feat_query: (B, N, D) — modality to repair (R or N patches)
            feat_T:     (B, N, D) — TI patches, always clean reference
            feat_peer:  (B, N, D) — other modality patches (N for R, R for N)
            q_self:     (B,) scalar or (B, 196) per-patch quality of query [0,1]
            q_peer:     (B,) scalar or (B, 196) per-patch quality of peer [0,1]
            
        Token-level: per-patch quality → per-patch gate → local repair.
        Scalar q broadcasts to all patches for backward compat.
        None → no quality gating.
        """
        identity = feat_query
        q = self.norm_q(feat_query)
        
        # 1. Cross-attend to TI (the anchor, always available)
        k_T = self.norm_kv_T(feat_T)
        v_T = self.norm_kv_T(feat_T)
        attn_T, _ = self.cross_attn_T(q, k_T, v_T)

        # 2. Cross-attend to peer modality
        k_peer = self.norm_kv_peer(feat_peer)
        v_peer = self.norm_kv_peer(feat_peer)
        attn_peer, _ = self.cross_attn_peer(q, k_peer, v_peer)
        
        B, N, D = feat_query.shape

        q_self_token = self._quality_to_token_gate(q_self, N)
        q_peer_token = self._quality_to_token_gate(q_peer, N)

        if q_self_token is not None:
            gate_T = 1.0 + (1.0 - q_self_token)  # [B,197,1]
        else:
            gate_T = 1.0

        if q_self_token is not None and q_peer_token is not None:
            peer_adv = q_peer_token - q_self_token
            gate_peer = ((peer_adv - 0.03) / 0.25).clamp(0, 1) # [B,197,1]
        else:
            gate_peer = 0.0

        # 诊断用：记录 peer gate 是否大量误开
        if isinstance(gate_peer, torch.Tensor):
            patch_peer = gate_peer[:, 1:, :]  # 去掉 CLS，只看 patch
            self._last_peer_gate_mean = patch_peer.detach().mean()
            self._last_peer_open_ratio = (patch_peer.detach() > 0.05).float().mean()
        else:
            self._last_peer_gate_mean = torch.tensor(0.0, device=feat_query.device)
            self._last_peer_open_ratio = torch.tensor(0.0, device=feat_query.device)
        
        # Combine: self + TI repair + peer repair
        feat_query = identity \
                     + torch.tanh(self.gate_T) * attn_T * gate_T \
                     + torch.tanh(self.gate_peer) * attn_peer * gate_peer
        
        # Shared FFN with gated residual
        identity2 = feat_query
        feat_query = identity2 + torch.tanh(self.gate_ffn) * self.ffn(self.norm_out(feat_query))
        return feat_query

    def _quality_to_token_gate(self, q, num_tokens):
        """
        Convert quality to token-level gate.

        q:
            [B]          -> [B,197,1]
            [B,196]      -> [B,197,1]
            [B,196,1]    -> [B,197,1]
        """
        if q is None:
            return None

        if q.dim() == 1:
            return q.view(-1, 1, 1).expand(-1, num_tokens, 1)

        if q.dim() == 2:
            q_patch = q.unsqueeze(-1)
        else:
            q_patch = q

        assert q_patch.size(1) == num_tokens - 1, \
            f"q_patch length {q_patch.size(1)} != num_tokens-1 {num_tokens - 1}"

        q_cls = patch_quality_to_scalar(q_patch).view(-1, 1, 1)
        return torch.cat([q_cls, q_patch], dim=1)

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
    def __init__(self, model_name='/root/autodl-tmp/pretrained/clip-vit-base-patch16', freeze_blocks=0):
        super().__init__()
        self.vision = CLIPVisionModel.from_pretrained(model_name, local_files_only=True)
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
        
        self.qmap_head_R = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim // 4),
            nn.GELU(),
            nn.Linear(self.dim // 4, 1)
        )

        self.qmap_head_N = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim // 4),
            nn.GELU(),
            nn.Linear(self.dim // 4, 1)
        )
        # 初始化为“默认大部分 patch 是干净的”
        nn.init.constant_(self.qmap_head_R[-1].bias, -1.0)
        nn.init.constant_(self.qmap_head_N[-1].bias, -1.0)
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
        self.use_coen_lite = False
    
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
        self._qmap_aux_loss = None
        # 1. Extract CLIP features
        featR = self.backbone_rgb(x1)  # (B, 197, 768)
        featN = self.backbone_ni(x2)
        featT = self.backbone_ti(x3)
        
        # 2. MFMP: cross-attention between R and N patches (before fusion)
        if self.use_mfmp and self.training:
            featR_mfmp, featN_mfmp = self.get_cls_feat_mfmp(featR, featN)
        else:
            featR_mfmp, featN_mfmp = featR, featN
        
        # 3. CoEN quality detection AFTER MFMP (mirrors original FCE→MFMP dependency)
        # Original FACENet: MFMP aligns R↔N → FCE detects damage → hard repair
        # CoEN:           MFMP aligns R↔N → quality from aligned features → soft repair + gate
        if self.use_coen_lite:
            if self.use_coen_lite:
                bad_logit_R = self.qmap_head_R(featR_mfmp[:, 1:, :])  # [B,196,1]
                bad_logit_N = self.qmap_head_N(featN_mfmp[:, 1:, :])  # [B,196,1]

                bad_learn_R = torch.sigmoid(bad_logit_R)  # [B,196,1]
                bad_learn_N = torch.sigmoid(bad_logit_N)  # [B,196,1]
                # H2 debug: 只打印一次，检查 flare_label 和 learned qmap 是否正常
                if self.training and not hasattr(self, "_debug_h2_qmap_printed"):
                    print("\n[H2 DEBUG] ===== qmap head check =====")

                    if flare_label is None:
                        print("[H2 DEBUG] flare_label: None")
                    else:
                        print("[H2 DEBUG] flare_label shape:", flare_label.shape)
                        print("[H2 DEBUG] flare_label dtype:", flare_label.dtype)
                        print("[H2 DEBUG] flare_label unique:", torch.unique(flare_label.detach().cpu()))
                        print("[H2 DEBUG] flare_label mean:",
                            flare_label.float().detach().mean().item())

                    print("[H2 DEBUG] bad_logit_R mean/min/max:",
                        bad_logit_R.detach().mean().item(),
                        bad_logit_R.detach().min().item(),
                        bad_logit_R.detach().max().item())

                    print("[H2 DEBUG] bad_logit_N mean/min/max:",
                        bad_logit_N.detach().mean().item(),
                        bad_logit_N.detach().min().item(),
                        bad_logit_N.detach().max().item())

                    print("[H2 DEBUG] bad_learn_R mean/min/max:",
                        bad_learn_R.detach().mean().item(),
                        bad_learn_R.detach().min().item(),
                        bad_learn_R.detach().max().item())

                    print("[H2 DEBUG] bad_learn_N mean/min/max:",
                        bad_learn_N.detach().mean().item(),
                        bad_learn_N.detach().min().item(),
                        bad_learn_N.detach().max().item())

                    print("[H2 DEBUG] ===========================\n")
                    self._debug_h2_qmap_printed = True
                # 轻量弱监督：只有图像级 flare_label，没有 patch 标注
                # 思路：
                # 1. flare 图：允许少量 top-k patch 高响应
                # 2. 非 flare 图：整体 bad map 应该低
                # 3. 稀疏约束：避免整张图都变红
                if self.training and flare_label is not None:
                    target = flare_label.float().view(-1, 1)  # [B,1]

                    bad_prob_R = bad_learn_R.squeeze(-1)      # [B,196]
                    bad_prob_N = bad_learn_N.squeeze(-1)

                    bad_logit_R_2d = bad_logit_R.squeeze(-1)  # [B,196]
                    bad_logit_N_2d = bad_logit_N.squeeze(-1)

                    # top-k patch logits，用 logits 做 BCEWithLogits，AMP 安全
                    k = max(1, int(bad_logit_R_2d.size(1) * 0.10))

                    pred_logit_R = bad_logit_R_2d.topk(k, dim=1, largest=True).values.mean(dim=1, keepdim=True)
                    pred_logit_N = bad_logit_N_2d.topk(k, dim=1, largest=True).values.mean(dim=1, keepdim=True)

                    pos_mask = target.squeeze(1) > 0.5
                    neg_mask = target.squeeze(1) <= 0.5

                    loss_items = []

                    # 正样本：鼓励 flare 图里 top-k patch 变高
                    if pos_mask.any():
                        pos_target = torch.ones_like(pred_logit_R[pos_mask])

                        loss_pos_R = F.binary_cross_entropy_with_logits(
                            pred_logit_R[pos_mask], pos_target
                        )
                        loss_pos_N = F.binary_cross_entropy_with_logits(
                            pred_logit_N[pos_mask], pos_target
                        )

                        loss_items.append(0.5 * (loss_pos_R + loss_pos_N))

                    # 负样本：鼓励 non-flare 图 top-k patch 变低
                    if neg_mask.any():
                        neg_target = torch.zeros_like(pred_logit_R[neg_mask])

                        loss_neg_R = F.binary_cross_entropy_with_logits(
                            pred_logit_R[neg_mask], neg_target
                        )
                        loss_neg_N = F.binary_cross_entropy_with_logits(
                            pred_logit_N[neg_mask], neg_target
                        )

                        # 负样本多，所以权重小一点，防止压扁 qmap
                        loss_items.append(0.3 * 0.5 * (loss_neg_R + loss_neg_N))

                    if len(loss_items) > 0:
                        bce_balanced = sum(loss_items)
                    else:
                        bce_balanced = bad_prob_R.mean() * 0.0

                    # 稀疏项降低，避免把 learned qmap 过度压平
                    sparse_R = bad_prob_R.mean()
                    sparse_N = bad_prob_N.mean()

                    self._qmap_aux_loss = bce_balanced + 0.01 * (sparse_R + sparse_N)
            # Use MFMP-aligned features during training, backbone features at inference
            q_R, q_N = compute_combined_quality(
                x1, x2,
                featR_mfmp, featN_mfmp,  # MFMP output (training) or backbone (inference)
                self.training,
                bad_learn_R=bad_learn_R,
                bad_learn_N=bad_learn_N
            )
            if self.training and not hasattr(self, "_debug_qmap_printed"):
                print("[Patch-CoEN] q_R:", q_R.shape, "q_N:", q_N.shape)
                print("[Patch-CoEN] featR:", featR.shape, "featT:", featT.shape)
                self._debug_qmap_printed = True
            q_R_sample = patch_quality_to_scalar(q_R)
            q_N_sample = patch_quality_to_scalar(q_N)

            self._coen_qR = q_R_sample.detach().mean()
            self._coen_qN = q_N_sample.detach().mean()

            self._coen_qR_log = float(self._coen_qR.item())
            self._coen_qN_log = float(self._coen_qN.item())
            
            # 保存最近一个 batch，用于 q_map 可视化
            self._last_q_R_map = q_R.detach()
            self._last_q_N_map = q_N.detach()
            self._last_img_R = x1.detach()
            self._last_img_N = x2.detach()
        else:
            q_R = None
            q_N = None
        
        

        # Save pre-fusion features for peer reference (both use originals)
        featR_in = featR
        featN_in = featN
        
        featR = self.fusion_R(featR_in, featT, featN_in, q_R, q_N)
        featN = self.fusion_N(featN_in, featT, featR_in, q_N, q_R)

        self._peer_R_open_ratio = self.fusion_R._last_peer_open_ratio
        self._peer_N_open_ratio = self.fusion_N._last_peer_open_ratio
        self._peer_R_gate_mean = self.fusion_R._last_peer_gate_mean
        self._peer_N_gate_mean = self.fusion_N._last_peer_gate_mean
        
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
                       cls_R_mfmp, score_R_label, cls_N_mfmp, score_N_label, None
            
            return ([score_R], [cls_R]), ([score_N], [cls_N]), ([score_T], [cls_T]), \
                   None, None, None, None, None
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
