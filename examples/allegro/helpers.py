import os
import open3d as o3d
import copy
from theseus.embodied.robot.forward_kinematics import Robot
from theseus.embodied.robot.forward_kinematics import (
    get_forward_kinematics,
)
import torch
import theseus as th


def get_inverse_kinematics(fkin, ee_pose_target):
    """Sets up inverse kinematics as an optimization problem that uses forward kinematics"""
    # Define cost (distance between desired and current ee pose)
    def ee_pose_err_fn(optim_vars, aux_vars):
        (theta,) = optim_vars
        (ee_pose_target,) = aux_vars

        ee_pose = fkin(theta.tensor)
        ee_pose = th.SE3(tensor=torch.vstack([pose for pose in ee_pose]))
        pose_err = ee_pose_target.local(ee_pose)
        return pose_err

    # Set up optimization
    optim_vars = (th.Vector(16, name="theta"),)
    aux_vars = (ee_pose_target,)

    cost_function = th.AutoDiffCostFunction(
        optim_vars,
        ee_pose_err_fn,
        6,
        aux_vars=aux_vars,
        name="ee_pose_err_fn",
        autograd_mode="dense",
    )
    objective = th.Objective()
    objective.add(cost_function)
    optimizer = th.LevenbergMarquardt(
        objective,
        max_iterations=15,
        step_size=1e-1,
    )
    theseus_optim = th.TheseusLayer(optimizer)

    # Optimize
    theseus_inputs = {
        "theta": torch.zeros(1, 16),
        "ee_pose_target": ee_pose_target,
    }

    with torch.set_grad_enabled(True):
        updated_inputs, info = theseus_optim.forward(
            theseus_inputs,
            optimizer_kwargs={
                "track_best_solution": True,
                "verbose": True,
                "track_error_history": True,
                "damping": 0.1,
            },
        )


def load_robot(urdf_file: str, device):
    robot = Robot.from_urdf_file(urdf_file, device=device)
    all_links = [
        "base_link",
        "link_8.0",
        "link_9.0",
        "link_10.0",
        "link_11.0",
        "link_11.0_tip",
        "link_4.0",
        "link_5.0",
        "link_6.0",
        "link_7.0",
        "link_7.0_tip",
        "link_0.0",
        "link_1.0",
        "link_2.0",
        "link_3.0",
        "link_3.0_tip",
        "link_12.0",
        "link_13.0",
        "link_14.0",
        "link_15.0",
        "link_15.0_tip",
    ]
    tip_ids = [5, 10, 15, 20]
    # index, middle, ring, thumb
    fkin, _ = get_forward_kinematics(robot, all_links)
    return robot, fkin, all_links, tip_ids


def viz_robot(poses, gt_poses, all_links: list, tip_ids: list, count):
    # Initialize the visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    n_links = len(all_links)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"

    unlit_mat = o3d.visualization.rendering.MaterialRecord()
    unlit_mat.shader = "unlitLine"
    unlit_mat.line_width = 10.0

    origin = o3d.geometry.TriangleMesh().create_coordinate_frame(
        size=0.02, origin=(0, 0, 0)
    )

    viz_poses = torch.eye(4, 4).unsqueeze(0).repeat(n_links, 1, 1)
    for i, pose in enumerate(poses):
        viz_poses[i, :3, :] = pose

    for i, pose in enumerate(viz_poses):
        tf_pose = copy.deepcopy(origin).transform(pose)
        if i in tip_ids:
            vis.add_geometry(tf_pose)
            # l = vis.add_3d_label(pose[:3, 3], all_links[i])
            # l.scale = 0.5

    viz_gt_poses = torch.eye(4, 4).unsqueeze(0).repeat(n_links, 1, 1)
    for i, pose in enumerate(gt_poses):
        viz_gt_poses[i, :3, :] = pose

    gt_points = [p[:3, 3].tolist() for p in viz_gt_poses]
    points = [p[:3, 3].tolist() for p in viz_poses]
    lines = [
        [0, 1],
        [0, 6],
        [0, 11],
        [0, 16],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [8, 9],
        [9, 10],
        [11, 12],
        [12, 13],
        [13, 14],
        [14, 15],
        [16, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]
    colors = [[0, 0, 1] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)

    line_set = o3d.geometry.LineSet()
    colors = [[0, 0, 0] for i in range(len(lines))]
    line_set.points = o3d.utility.Vector3dVector(gt_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(line_set)

    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0])
    ctr.set_front([0.1, 0, 0])
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(0.7)

    # Updates
    vis.poll_events()
    vis.update_renderer()

    # Capture image

    vis.capture_screen_image(f"output/{count}.jpg")
    vis.destroy_window()


def get_int(file: str) -> int:
    """
    Extract numeric value from file name
    """
    return int(file.split(".")[0])


def images_to_video(path: str, name: str, length: int = 120) -> None:
    import ffmpeg

    """
    https://stackoverflow.com/a/67152804 : list of images to .mp4
    """
    images = os.listdir(path)
    images = [im for im in images if im.endswith(".jpg")]

    rate = len(images) / length
    images = sorted(images, key=get_int)

    # Execute FFmpeg sub-process, with stdin pipe as input, and jpeg_pipe input format
    process = (
        ffmpeg.input("pipe:", r=str(rate))
        .output(os.path.join(path, f"{name}.mp4"))
        .global_args("-loglevel", "error")
        .global_args("-crf", "18")
        .global_args("-qscale", "0")
        .global_args("-y")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for image in images:
        image_path = os.path.join(path, image)
        with open(image_path, "rb") as f:
            # Read the JPEG file content to jpeg_data (bytes array)
            jpeg_data = f.read()
            # Write JPEG data to stdin pipe of FFmpeg process
            process.stdin.write(jpeg_data)

    # Close stdin pipe - FFmpeg fininsh encoding the output file.
    process.stdin.close()
    process.wait()
    print("converted")
