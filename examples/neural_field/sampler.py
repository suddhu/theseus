# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

import numpy as np


class Sampler:
    def __init__(self, sample_cfg, camera_cfg, device):
        super(Sampler, self).__init__()
        self.n_rays = sample_cfg.n_rays
        self.dist_behind_surf = sample_cfg.dist_behind_surf
        self.n_strat_samples = sample_cfg.n_strat_samples
        self.n_surf_samples = sample_cfg.n_surf_samples
        self.surface_samples_offset = sample_cfg.surface_samples_offset
        self.device = device

        self.yfov = camera_cfg.intrinsics.yfov
        self.W = camera_cfg.intrinsics.w
        self.H = camera_cfg.intrinsics.h
        self.cx, self.cy = self.W / 2.0, self.H / 2.0
        self.fx = self.W * 0.5 / np.tan(np.deg2rad(self.yfov) * 0.5)
        self.fy = self.H * 0.5 / np.tan(np.deg2rad(self.yfov) * 0.5)
        self.cam_dist = camera_cfg.cam_dist

        self.max_depth = np.sign(self.cam_dist + 1e-8) * sample_cfg.depth_range[1]
        self.min_depth = np.sign(self.cam_dist + 1e-8) * sample_cfg.depth_range[0]

    def ray_dirs_C(self, B, H, W, fx, fy, cx, cy, device, depth_type="z"):
        c, r = torch.meshgrid(
            torch.arange(W, device=device), torch.arange(H, device=device)
        )
        c, r = c.t().float(), r.t().float()
        size = [B, H, W]

        C = torch.empty(size, device=device)
        R = torch.empty(size, device=device)
        C[:, :, :] = c[None, :, :]
        R[:, :, :] = r[None, :, :]

        z = torch.ones(size, device=device)
        x = (C - cx) / fx
        y = (R - cy) / fy

        if H > W:
            x = -x  # tactile
        else:
            y, z = -y, -z  # realsense

        dirs = torch.stack((x, y, z), dim=3)
        if depth_type == "euclidean":
            norm = torch.norm(dirs, dim=3)
            dirs = dirs * (1.0 / norm)[:, :, :, None]

        return dirs

    def sample_points(
        self,
        depth_batch,
        T_WC_batch,
    ):
        """
        Sample points by first sampling pixels, then sample depths along
        the backprojected rays.
        """

        n_frames = depth_batch.shape[0]
        # randomly sample pixels on image batch (200 per image)

        dirs_C = self.ray_dirs_C(
            1,
            self.H,
            self.W,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.device,
            depth_type="z",
        )

        indices_b, indices_h, indices_w = self.sample_pixels(
            depth_batch,
            self.n_rays,
            n_frames,
            self.H,
            self.W,
            device=self.device,
        )

        # get the points to sample across image batch
        (
            dirs_C_sample,
            depth_sample,
            T_WC_sample,
            indices_b,
            indices_h,
            indices_w,
        ) = self.get_batch_data(
            depth_batch,
            T_WC_batch,
            dirs_C,
            indices_b,
            indices_h,
            indices_w,
        )

        max_depth = depth_sample + np.sign(self.cam_dist + 1e-8) * self.dist_behind_surf
        pc, z_vals = self.sample_along_rays(
            T_WC_sample,
            self.min_depth,
            max_depth,
            self.n_strat_samples,
            self.n_surf_samples,
            self.surface_samples_offset,
            dirs_C_sample,
            gt_depth=depth_sample,
        )  # pc: (num_samples, N + M + 1, 3)

        sample_pts = {
            "depth_batch": depth_batch,
            "pc": pc,
            "z_vals": z_vals,
            "indices_b": indices_b,
            "indices_h": indices_h,
            "indices_w": indices_w,
            "dirs_C_sample": dirs_C_sample,
            "depth_sample": depth_sample,
            "T_WC_sample": T_WC_sample,
        }
        return sample_pts

    def sample_pixels(
        self, depth_batch, n_rays, n_frames, h, w, device, method="masked"
    ):

        if "masked" in method:
            grid_h, grid_w = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij",
            )
            grid_h = grid_h.unsqueeze(0).repeat(n_frames, 1, 1)
            grid_w = grid_w.unsqueeze(0).repeat(n_frames, 1, 1)

            grid_b = (
                torch.ones((n_frames, h, w), device=device, dtype=int)
                * torch.arange(n_frames, device=device, dtype=int)[:, None, None]
            )

            t_mask_bool = ~torch.isnan(depth_batch)
            grid_h = grid_h.masked_select(t_mask_bool)
            grid_w = grid_w.masked_select(t_mask_bool)
            grid_b = grid_b.masked_select(t_mask_bool)

            total_rays = n_rays * n_frames
            total_valid = len(grid_b)
            indices = torch.randint(0, total_valid, (total_rays,), device=device)

            assert torch.any(
                ~torch.isnan(
                    depth_batch[grid_b[indices], grid_h[indices], grid_w[indices]]
                )
            ), "NaN found in sampling"

            return grid_b[indices], grid_h[indices], grid_w[indices]
        else:
            ## iSDF naive sampling
            total_rays = n_rays * n_frames
            indices_h = torch.randint(0, h, (total_rays,), device=device)
            indices_w = torch.randint(0, w, (total_rays,), device=device)

            indices_b = torch.arange(n_frames, device=device)
            indices_b = indices_b.repeat_interleave(n_rays)

            return indices_b, indices_h, indices_w

    def get_batch_data(
        self,
        depth_batch,
        T_WC_batch,
        dirs_C,
        indices_b,
        indices_h,
        indices_w,
    ):
        """
        Get depth, ray direction and pose for the sampled pixels.
        Only render where depth is valid.
        """
        depth_sample = depth_batch[indices_b, indices_h, indices_w].view(
            -1
        )  # n_rays * n_frames elements
        mask_valid_depth = depth_sample != torch.nan

        depth_sample = depth_sample[mask_valid_depth]

        indices_b = indices_b[mask_valid_depth]
        indices_h = indices_h[mask_valid_depth]
        indices_w = indices_w[mask_valid_depth]

        T_WC_sample = T_WC_batch[indices_b]
        dirs_C_sample = dirs_C[0, indices_h, indices_w, :].view(
            -1, 3
        )  # sampled ray directions

        return (
            dirs_C_sample,
            depth_sample,
            T_WC_sample,
            indices_b,
            indices_h,
            indices_w,
        )

    def stratified_sample(
        self,
        min_depth,
        max_depth,
        n_rays,
        device,
        n_stratified_samples,
        bin_length=None,
    ):
        """
        Random samples between min and max depth
        One sample from within each bin.

        If n_stratified_samples is passed then use fixed number of bins,
        else if bin_length is passed use fixed bin size.
        """

        # swap if mismatch
        # max_depth < min_depth
        # if torch.any(torch.tensor(max_depth < min_depth)):
        #     max_depth, min_depth = min_depth, max_depth

        if n_stratified_samples is not None:  # fixed number of bins
            n_bins = n_stratified_samples
            if isinstance(max_depth, torch.Tensor):
                sample_range = (max_depth - min_depth)[:, None]
                bin_limits = torch.linspace(0, 1, n_bins + 1, device=device)[None, :]
                bin_limits = bin_limits.repeat(n_rays, 1) * sample_range
                if isinstance(min_depth, torch.Tensor):
                    bin_limits = bin_limits + min_depth[:, None]
                else:
                    bin_limits = bin_limits + min_depth
                bin_length = sample_range / (n_bins)
            else:
                bin_limits = torch.linspace(
                    min_depth,
                    max_depth,
                    n_bins + 1,
                    device=device,
                )[None, :]
                bin_length = (max_depth - min_depth) / (n_bins)

        elif bin_length is not None:  # fixed size of bins
            bin_limits = torch.arange(
                min_depth,
                max_depth,
                bin_length,
                device=device,
            )[None, :]
            n_bins = bin_limits.size(1) - 1

        increments = torch.rand(n_rays, n_bins, device=device) * bin_length
        # increments = 0.5 * torch.ones(n_rays, n_bins, device=device) * bin_length
        lower_limits = bin_limits[..., :-1]
        z_vals = lower_limits + increments

        z_vals, _ = torch.sort(z_vals, dim=1)

        return z_vals

    def origin_dirs_W(self, T_WC, dirs_C):
        R_WC = T_WC[:, :3, :3]
        dirs_W = (R_WC * dirs_C[..., None, :]).sum(dim=-1)  # rotation
        origins = T_WC[:, :3, -1]

        return origins, dirs_W

    def sample_along_rays(
        self,
        T_WC,
        min_depth,
        max_depth,
        n_stratified_samples,
        n_surf_samples,
        surf_samples_offset,
        dirs_C,
        gt_depth=None,
    ):
        with torch.set_grad_enabled(True):
            # rays in world coordinate
            origins, dirs_W = self.origin_dirs_W(T_WC, dirs_C)

            origins = origins.view(-1, 3)
            dirs_W = dirs_W.view(-1, 3)
            n_rays = dirs_W.shape[0]

            # stratified sampling along rays # [total_n_rays, n_stratified_samples] between min_depth and max_depth
            z_vals = self.stratified_sample(
                min_depth,
                max_depth,
                n_rays,
                T_WC.device,
                n_stratified_samples,
                bin_length=None,
            )

            # if gt_depth is given, first sample at surface then around surface
            if gt_depth is not None and n_surf_samples > 0:
                surface_z_vals = gt_depth
                zeros = torch.zeros_like(surface_z_vals).to(z_vals.device)
                offsets = torch.normal(
                    torch.zeros(gt_depth.shape[0], n_surf_samples - 1),
                    surf_samples_offset,
                ).to(z_vals.device)
                near_surf_z_vals = gt_depth[:, None] + offsets
                if not isinstance(min_depth, torch.Tensor):
                    min_depth = torch.full(
                        near_surf_z_vals.shape, min_depth, dtype=max_depth.dtype
                    ).to(z_vals.device)[..., 0]

                if not isinstance(max_depth, torch.Tensor):
                    max_depth = torch.full(near_surf_z_vals.shape, max_depth).to(
                        z_vals.device
                    )[..., 0]

                # swap min-max if needed
                indices = max_depth < min_depth
                min_depth[indices], max_depth[indices] = (
                    max_depth[indices],
                    min_depth[indices],
                )

                near_surf_z_vals = torch.clamp(
                    near_surf_z_vals, min_depth[:, None], max_depth[:, None]
                )

                # 1 sample of surface, n_surf_samples around surface, n_stratified_samples along [min, max]
                z_vals = torch.cat(
                    (
                        surface_z_vals[:, None],
                        zeros[:, None],
                        near_surf_z_vals,
                        z_vals,
                    ),
                    dim=1,
                )

            # point cloud of 3d sample locations
            pc = origins[:, None, :] + (dirs_W[:, None, :] * z_vals[:, :, None])

        return pc, z_vals
