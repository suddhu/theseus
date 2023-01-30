# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Loss terms for SDF, from https://arxiv.org/pdf/2204.02296.pdf and https://arxiv.org/pdf/2104.04532.pdf
"""

import torch


def bounds_pc(pc, z_vals, depth_sample):
    """
    Equation 4 iSDF https://arxiv.org/pdf/2204.02296.pdf
    """
    with torch.set_grad_enabled(True):
        surf_pc = pc[:, 0]  # first element always pc?
        diff = pc[:, :, None] - surf_pc
        dists = diff.norm(dim=-1)
        dists, closest_ixs = dists.min(axis=-1)
        behind_surf = torch.abs(z_vals) > torch.abs(depth_sample[:, None])
        dists[behind_surf] *= -1
        bounds = dists
    return bounds


def sdf_loss(sdf, bounds, t, loss_type="L1"):
    """
    params:
    sdf: predicted sdf values.
    bounds: upper bound on abs(sdf)
    t: truncation distance up to which the sdf value is directly supevised.
    loss_type: L1 or L2 loss.
    """
    # free_space_loss_mat, trunc_loss_mat = full_sdf_loss(sdf, bounds, t)
    free_space_loss_mat, trunc_loss_mat = tsdf_loss(sdf, bounds, t)

    # decide which supervision
    free_space_ixs = bounds > t
    free_space_loss_mat[~free_space_ixs] = 0.0
    trunc_loss_mat[free_space_ixs] = 0.0

    sdf_loss_mat = free_space_loss_mat + trunc_loss_mat

    if loss_type == "L1":
        sdf_loss_mat = torch.abs(sdf_loss_mat)
    elif loss_type == "L2":
        sdf_loss_mat = torch.square(sdf_loss_mat)
    else:
        raise ValueError("Must be L1 or L2")

    return sdf_loss_mat, free_space_ixs


def full_sdf_loss(sdf, target_sdf, trunc_dist, free_space_factor=5.0):
    """
    For samples that lie in free space before truncation region:
        loss(sdf_pred, sdf_gt) =  { max(0, sdf_pred - sdf_gt), if sdf_pred >= 0
                                  { exp(-sdf_pred) - 1, if sdf_pred < 0

    For samples that lie in truncation region:
        loss(sdf_pred, sdf_gt) = sdf_pred - sdf_gt
    """

    # free_space_loss_mat = torch.max(
    #     torch.nn.functional.relu(sdf - target_sdf),
    #     torch.exp(-free_space_factor * sdf) - 1.
    # )
    free_space_loss_mat = sdf - trunc_dist
    trunc_loss_mat = sdf - target_sdf

    return free_space_loss_mat, trunc_loss_mat


def tsdf_loss(sdf, target_sdf, trunc_dist):
    """
    tsdf loss from: https://arxiv.org/pdf/2104.04532.pdf
    SDF values in truncation region are scaled in range [0, 1].
    """
    free_space_loss_mat = sdf - torch.ones(sdf.shape, device=sdf.device)
    trunc_loss_mat = sdf - target_sdf

    return free_space_loss_mat, trunc_loss_mat


def tot_loss(
    sdf_loss_mat,
    free_space_ixs,
    trunc_weight,
):
    """
    Applies truncation weight for non-free space and outputs average loss
    """
    sdf_loss_mat[~free_space_ixs] *= trunc_weight

    losses = {"sdf_loss": sdf_loss_mat.mean().item()}
    tot_loss_mat = sdf_loss_mat

    tot_loss = tot_loss_mat.mean()
    losses["total_loss"] = tot_loss

    return tot_loss, tot_loss_mat, losses
