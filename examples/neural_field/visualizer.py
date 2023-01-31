import numpy as np
import open3d as o3d
import time

"""
Open3D visualizer class
"""


class Visualizer:
    def __init__(self, camera_cfg, scene_file, T_gt, T):

        gt_mesh = o3d.io.read_triangle_mesh(scene_file)
        gt_mesh.compute_vertex_normals()

        yfov = camera_cfg.intrinsics.yfov
        W = camera_cfg.intrinsics.w
        H = camera_cfg.intrinsics.h
        cx, cy = W / 2.0, H / 2.0
        fx = W * 0.5 / np.tan(np.deg2rad(yfov) * 0.5)
        fy = H * 0.5 / np.tan(np.deg2rad(yfov) * 0.5)

        self.flip_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        self.cam = o3d.camera.PinholeCameraIntrinsic(
            W,
            H,
            fx,
            fy,
            cx,
            cy,
        )

        init_T = T_gt.copy().to_matrix().detach().cpu().numpy().squeeze()
        opt_T = T.copy().to_matrix().detach().cpu().numpy().squeeze()

        self.gt_frustum = o3d.geometry.LineSet.create_camera_visualization(
            self.cam.width,
            self.cam.height,
            self.cam.intrinsic_matrix,
            np.linalg.inv(init_T @ self.flip_matrix),
            scale=0.1,
        )
        self.gt_frustum.paint_uniform_color([0.0, 0.0, 0.0])

        self.opt_frustum = o3d.geometry.LineSet.create_camera_visualization(
            self.cam.width,
            self.cam.height,
            self.cam.intrinsic_matrix,
            np.linalg.inv(opt_T @ self.flip_matrix),
            scale=0.1,
        )
        self.opt_frustum.paint_uniform_color([0.0, 0.0, 1.0])

        self.look_at = init_T[:3, 3] / 2.0

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.add_geometry(gt_mesh)
        self.vis.add_geometry(self.gt_frustum)
        self.vis.add_geometry(self.opt_frustum)

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(np.zeros((100, 3)))
        self.vis.add_geometry(self.pcd)

        # track position of camera too
        self.cam_pos = o3d.geometry.PointCloud()
        self.cam_pos.points = o3d.utility.Vector3dVector(
            np.expand_dims(opt_T[:3, 3], 0)
        )
        self.vis.add_geometry(self.cam_pos)

        # rot[:3, :3] = R.from_euler("y", 15, degrees=True).as_matrix()
        # rot = rot.dot(camera_params.extrinsic)
        # camera_params.extrinsic = rot
        # ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
        self.vis.update_renderer()

        self.vis.poll_events()
        self.vis.update_renderer()

    def run_vis(self):
        self.vis.run()
        while True:
            time.sleep(1)

    def set_cam(self, cam_pos=np.array([0.5, 0.5, 0.7])):
        ctr = self.vis.get_view_control()
        ctr.set_up((0, 0, 1))
        ctr.set_front(cam_pos)
        ctr.set_lookat(self.look_at)
        ctr.set_zoom(0.5)

    def update_cam(self, T):
        opt_T = T.copy().to_matrix().detach().cpu().numpy().squeeze()
        self.vis.remove_geometry(self.opt_frustum)
        self.opt_frustum = o3d.geometry.LineSet.create_camera_visualization(
            self.cam.width,
            self.cam.height,
            self.cam.intrinsic_matrix,
            np.linalg.inv(opt_T @ self.flip_matrix),
            scale=0.1,
        )
        self.cam_pos.points.extend(
            o3d.utility.Vector3dVector(np.expand_dims(opt_T[:3, 3], 0))
        )
        self.opt_frustum.paint_uniform_color([0.0, 0.0, 1.0])
        self.vis.add_geometry(self.opt_frustum)
        self.set_cam()
        self.vis.poll_events()
        self.vis.update_renderer()
        self.vis.update_geometry(self.cam_pos)
        return

    def update_pc(self, pc):
        self.vis.remove_geometry(self.pcd)
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(pc)
        self.vis.add_geometry(self.pcd)
        self.set_cam()
        self.vis.poll_events()
        self.vis.update_renderer()
        return
