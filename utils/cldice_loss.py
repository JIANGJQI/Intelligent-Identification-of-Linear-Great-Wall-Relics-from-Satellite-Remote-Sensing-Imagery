"""
soft-clDice loss — 拓扑保持损失
CVPR 2021: Shit et al., "clDice - a Novel Topology-Preserving Loss Function"
Soft-skeletonization via iterative differentiable min/max pooling
"""

import torch
import torch.nn.functional as F


def soft_skel(x, k=5):
    """
    Soft-skeletonization by iterative min/max pooling (clDice CVPR 2021).
    Each iteration: erode→dilate(opening)→skeleton=ReLU(x-opening)→use opening for next
    x: [B, 1, H, W] sigmoid probabilities
    k: iterations (5 is the standard value)
    returns: soft skeleton [B, 1, H, W]
    """
    if k <= 0:
        return x
    skel = 0
    for _ in range(k):
        # Morphological opening: erode(min-pool) then dilate(max-pool)
        eroded = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        opened = F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)
        # Skeleton at current scale = original - opening
        skel = skel + F.relu(x - opened)
        # Use opening for next thinner iteration
        x = opened
    return skel


def soft_cl_dice(pred, target, k=5):
    """
    pred:   [B, 1, H, W] sigmoid probabilities
    target: [B, 1, H, W] binary mask
    k:      soft-skeleton iterations

    returns: 1 - clDice (for minimization). Returns 1.0 for background-only tiles.
    """
    # 纯背景 tile: clDice=1.0 (no skeleton possible)
    if target.sum() == 0:
        return torch.tensor(1.0, device=pred.device)

    # Soft skeleton of prediction
    pred_skel = soft_skel(pred, k=k)

    # Soft skeleton of GT (on-the-fly, binary mask → skeletonizes naturally)
    target_skel = soft_skel(target, k=k)

    # tprec: fraction of pred skeleton within target mask
    tprec = (pred_skel * target).sum(dim=(2, 3)) / (pred_skel.sum(dim=(2, 3)) + 1e-8)

    # tsens: fraction of target skeleton within pred mask
    tsens = (target_skel * pred).sum(dim=(2, 3)) / (target_skel.sum(dim=(2, 3)) + 1e-8)

    cl_dice = 2 * tprec * tsens / (tprec + tsens + 1e-8)
    return (1 - cl_dice).mean()
