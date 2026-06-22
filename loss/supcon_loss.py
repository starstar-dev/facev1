"""
Strict cross-modal supervised contrastive loss.
Positive: same vehicle ID AND different modality.
Negative: different vehicle ID.
Adapted from H2_supcon branch.
"""
import torch
import torch.nn.functional as F


def cross_modal_supcon_loss(feat_R, feat_N, feat_T, labels, temperature=0.07, eps=1e-8):
    """
    Strict cross-modal supervised contrastive loss.

    Positive:
        same vehicle ID AND different modality

    Negative:
        different vehicle ID

    feat_R/N/T: [B, D]
    labels:     [B]
    """
    B = feat_R.size(0)
    device = feat_R.device

    features = torch.cat([feat_R, feat_N, feat_T], dim=0).float()  # [3B, D]
    features = F.normalize(features, dim=1)

    labels_all = labels.repeat(3).view(-1, 1)  # [3B, 1]

    modal_ids = torch.cat([
        torch.zeros(B, device=device, dtype=torch.long),
        torch.ones(B, device=device, dtype=torch.long),
        torch.full((B,), 2, device=device, dtype=torch.long),
    ], dim=0).view(-1, 1)

    sim = torch.matmul(features, features.t()) / temperature
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()

    self_mask = torch.eye(3 * B, device=device)

    same_id = labels_all.eq(labels_all.t())
    diff_modal = modal_ids.ne(modal_ids.t())

    pos_mask = (same_id & diff_modal).float()
    pos_mask = pos_mask * (1.0 - self_mask)

    exp_sim = torch.exp(sim) * (1.0 - self_mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + eps)

    pos_count = pos_mask.sum(dim=1)
    loss = -(pos_mask * log_prob).sum(dim=1) / (pos_count + eps)

    valid = pos_count > 0
    if valid.any():
        return loss[valid].mean()
    return features.new_tensor(0.0)
