"""
QTA-ReID with CNN backbone.
Reuses TPQE (CoEN-Lite) + TAQGR (CrossModalFusion) from clip_facenet.py.
Only the encoder changes — CLIP ViT → CNN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.clip_facenet import CrossModalFusion, CrossPatchAttention, \
    weights_init_kaiming, weights_init_classifier
from model.coen_lite import compute_combined_quality, patch_quality_to_scalar
from model.backbones.cnn_adapter import ResNet50Adapter, OSNetAdapter


class QTAReIDCNN(nn.Module):
    """
    QTA-ReID with CNN backbone (ResNet-50 or OSNet).
    Architecture: CNN × 3 → TPQE → TAQGR (CrossModalFusion) → BNNeck → cls
    """
    def __init__(self, num_classes, backbone_name='resnet50', cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.dim = 768
        
        # Select backbone
        if backbone_name == 'resnet50':
            self.backbone_rgb = ResNet50Adapter(out_dim=self.dim)
            self.backbone_ni  = ResNet50Adapter(out_dim=self.dim)
            self.backbone_ti  = ResNet50Adapter(out_dim=self.dim)
            print(f'Using ResNet-50 backbone. Total params: {sum(p.numel() for p in self.parameters())/1e6:.1f}M')
        elif backbone_name == 'osnet':
            self.backbone_rgb = OSNetAdapter(out_dim=self.dim)
            self.backbone_ni  = OSNetAdapter(out_dim=self.dim)
            self.backbone_ti  = OSNetAdapter(out_dim=self.dim)
            print(f'Using OSNet backbone. Total params: {sum(p.numel() for p in self.parameters())/1e6:.1f}M')
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        # TPQE: quality map heads (same as CLIP version)
        self.qmap_head_R = nn.Sequential(
            nn.LayerNorm(self.dim), nn.Linear(self.dim, self.dim // 4),
            nn.GELU(), nn.Linear(self.dim // 4, 1)
        )
        self.qmap_head_N = nn.Sequential(
            nn.LayerNorm(self.dim), nn.Linear(self.dim, self.dim // 4),
            nn.GELU(), nn.Linear(self.dim // 4, 1)
        )
        nn.init.constant_(self.qmap_head_R[-1].bias, -1.0)
        nn.init.constant_(self.qmap_head_N[-1].bias, -1.0)
        
        # MFMP
        self.cross_patch_attn = CrossPatchAttention(dim=self.dim)
        
        # TAQGR: Cross-modal fusion (same as CLIP version)
        self.fusion_R = CrossModalFusion(dim=self.dim)
        self.fusion_N = CrossModalFusion(dim=self.dim)
        
        # BNNeck per modality
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
        
        # MFMP label classifiers
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
        
        # Flags
        self.use_mamba_enhance = False
        self.use_fce = False
        self.use_mfmp = True
        self.use_mcloss = True
        self.use_coen_lite = True
        self.use_fusion = True
        self.use_quality_loss_gate = True
        self.use_supcon = True
        self.coen_use_learned_qmap = True
        self.coen_use_image_prior = True
        self.coen_use_disagreement = True
        self.use_qmap_aux_loss = True
        self.use_global_static_fusion = False
    
    def forward(self, x1, x2, x3, label=None, cam_label=None, view_label=None, flare_label=None):
        """Same flow as CLIPFACENet.forward()"""
        self._qmap_aux_loss = None
        
        featR = self.backbone_rgb(x1)
        featN = self.backbone_ni(x2)
        featT = self.backbone_ti(x3)
        
        if self.use_mfmp:
            # Same MFMP logic as CLIPFACENet
            tokenR, tokenN = featR[:, 0:1], featN[:, 0:1]
            patchesR, patchesN = featR[:, 1:], featN[:, 1:]
            patchesR, patchesN = self.cross_patch_attn(patchesR, patchesN)
            featR_mfmp = torch.cat([tokenR, patchesR], dim=1)
            featN_mfmp = torch.cat([tokenN, patchesN], dim=1)
        else:
            featR_mfmp, featN_mfmp = featR, featN
        
        # TPQE (same as CLIPFACENet)
        if self.use_coen_lite:
            if self.coen_use_learned_qmap:
                bad_logit_R = self.qmap_head_R(featR_mfmp[:, 1:, :])
                bad_logit_N = self.qmap_head_N(featN_mfmp[:, 1:, :])
                bad_learn_R = torch.sigmoid(bad_logit_R)
                bad_learn_N = torch.sigmoid(bad_logit_N)
                # Aux loss (simplified — full version in CLIPFACENet)
                if self.training and flare_label is not None and self.use_qmap_aux_loss:
                    # Simplified supervised qmap loss using flair label
                    target = flare_label.float().view(-1, 1)
                    pred_logit_R = bad_logit_R.mean(dim=1)
                    pred_logit_N = bad_logit_N.mean(dim=1)
                    self._qmap_aux_loss = F.binary_cross_entropy_with_logits(
                        (pred_logit_R + pred_logit_N) / 2, target.clamp(0, 1)
                    ) * 0.1
            else:
                bad_learn_R, bad_learn_N = None, None
            
            q_R, q_N = compute_combined_quality(
                x1, x2, featR_mfmp, featN_mfmp, self.training,
                bad_learn_R=bad_learn_R, bad_learn_N=bad_learn_N,
                w_learned_train=0.15 if self.coen_use_learned_qmap else 0.0,
                w_img_train=0.55 if self.coen_use_image_prior else 0.0,
                w_disagree_train=0.30 if self.coen_use_disagreement else 0.0,
                w_learned_eval=0.05 if self.coen_use_learned_qmap else 0.0,
                w_img_eval=0.70 if self.coen_use_image_prior else 0.0,
                w_disagree_eval=0.25 if self.coen_use_disagreement else 0.0
            )
            q_R_sample = patch_quality_to_scalar(q_R)
            q_N_sample = patch_quality_to_scalar(q_N)
            self._coen_qR = q_R_sample.detach().mean()
            self._coen_qN = q_N_sample.detach().mean()
            self._coen_qR_log = float(self._coen_qR.item())
            self._coen_qN_log = float(self._coen_qN.item())
        else:
            q_R, q_N = None, None
        
        featR_in, featN_in = featR, featN
        
        # TAQGR (same fusion module)
        if self.use_fusion:
            featR = self.fusion_R(featR_in, featT, featN_in, q_R, q_N)
            featN = self.fusion_N(featN_in, featT, featR_in, q_N, q_R)
        else:
            featR, featN = featR_in, featN_in
        
        cls_R, cls_N, cls_T = featR[:, 0], featN[:, 0], featT[:, 0]
        
        if self.training:
            bn_R = self.bottleneck_R(cls_R)
            bn_N = self.bottleneck_N(cls_N)
            bn_T = self.bottleneck_T(cls_T)
            score_R = self.classifier_R(bn_R)
            score_N = self.classifier_N(bn_N)
            score_T = self.classifier_T(bn_T)
            
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
        param_dict = torch.load(trained_path, map_location='cpu')
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if k.startswith('module.'):
                k = k[7:]
        self.load_state_dict(param_dict, strict=False)
        print(f'Loaded pretrained model from {trained_path}')