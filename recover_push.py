import os
import json
import numpy as np
import mengine as m
from scipy.spatial.transform import Rotation
from multiprocessing import Pool

OBJECT_ORI_POS_Z = 0.905
RENDER = False


def reset(position, table_friction=0.5, cube_mass=100, cube_friction=0.5):
    # Create environment and ground plane
    env.reset()
    env.render = RENDER
    ground = m.Ground()

    # Create table and cube
    table = m.URDF(filename=os.path.join(m.directory, 'table', 'table.urdf'),
                   static=True,
                   position=[0, 0, 0],
                   orientation=[0, 0, 0, 1],
                   maximal_coordinates=True)
    table.set_whole_body_frictions(lateral_friction=table_friction,
                                   spinning_friction=0,
                                   rolling_friction=0)
    # cube = None
    cube = m.Shape(m.Box(half_extents=[0.25, 0.14, 0.16]),
                   static=False,
                   mass=cube_mass,
                   position=[0, 0, OBJECT_ORI_POS_Z],
                   orientation=[0, 0, 0, 0.8],
                   rgba=[0, 1, 0, 1])
    cube.set_whole_body_frictions(lateral_friction=cube_friction,
                                  spinning_friction=0,
                                  rolling_friction=0)

    true_cube = m.Shape(m.Box(half_extents=[0.25, 0.14, 0.16]),
                        static=False,
                        mass=0,
                        position=[0, 0, OBJECT_ORI_POS_Z],
                        orientation=[0, 0, 0, 1],
                        rgba=[1, 0, 0, 0.8],
                        collision=False)
    # true_cube.set_whole_body_frictions(lateral_friction=cube_friction,
    #                               spinning_friction=0,
    #                               rolling_friction=0)
    # Create Panda robot
    robot = m.Robot.Panda(position=[0.5, 0, 0.75])

    # Initialize joint angles
    target_joint_angles = robot.ik(robot.end_effector,
                                   target_pos=position,
                                   target_orient=orient)
    robot.control(target_joint_angles, set_instantly=True)
    return robot, cube, true_cube


def trans_coordinate(poses):
    first_pose = np.array(poses[0])
    all_trans_pose = []
    for pose in poses:
        pose = np.array(pose)
        trans_pose = np.linalg.inv(first_pose) @ pose
        trans_pose[2, 3] += OBJECT_ORI_POS_Z
        all_trans_pose.append(trans_pose)

    return all_trans_pose


def get_robot_contact_pt(trans_poses):
    all_robot_pt = []
    T = np.eye(4)
    T[0, 3] = 0.3
    T[2, 3] = -0.02
    for pose in trans_poses:
        robot_pt = T @ pose
        all_robot_pt.append(robot_pt)
    return all_robot_pt


def compute_traj_diff(pred_traj, true_traj, weight=1):
    pose_diff_sum = 0
    ori_diff_sum = 0
    n_pose = len(pred_traj)
    for pred_pose, true_pose in zip(pred_traj, true_traj):
        pose_diff = np.linalg.norm(pred_pose[:3,-1] - true_pose[:3,-1])
        pred_ori_quat = Rotation.from_matrix(pred_pose[:3, :3]).as_quat()
        true_ori_quat = Rotation.from_matrix(true_pose[:3, :3]).as_quat()
        ori_diff = 1 - np.dot(pred_ori_quat, true_ori_quat)**2
        pose_diff_sum += pose_diff
        ori_diff_sum += ori_diff
    mean_error = (pose_diff_sum * weight + ori_diff_sum) / n_pose
    pose_error = pose_diff_sum / n_pose
    ori_error = ori_diff_sum / n_pose
    return mean_error, pose_error, ori_error


def compute_obj(force):
    pos = scenario['position']
    # Reset simulator
    robot, cube, true_cube = reset(
        scenario['position'],
        scenario['table_friction'],
        scenario['cube_mass'],
        scenario['cube_friction'],
    )

    target_joint_angles = robot.ik(robot.end_effector,
                                    target_pos=robot_pts[0][:3, -1],
                                    target_orient=orient,
                                    use_current_joint_angles=True)
    robot.set_joint_angles(target_joint_angles)

    pred_cube_poses = []
    for pose, robot_pt in zip(poses, robot_pts):
        R = pose[:3,:3]
        q = Rotation.from_matrix(R).as_quat()
        # cube.set_base_pos_orient(pose[:3,-1], q)
        true_cube.set_base_pos_orient(pose[:3,-1], q)

        target_joint_angles = robot.ik(robot.end_effector,
                                    target_pos=robot_pt[:3, -1],
                                    target_orient=orient,
                                    use_current_joint_angles=True)
        robot.control(target_joint_angles, forces=force)
        m.step_simulation(realtime=False)
        pred_cube_p, pred_cube_q = cube.get_base_pos_orient()
        R = Rotation.from_quat(pred_cube_q).as_matrix()
        T = np.r_[np.c_[R, pred_cube_p], np.array([[0, 0, 0, 1]])]
        pred_cube_poses.append(T)

    error = compute_traj_diff(pred_cube_poses, poses)
    print(f"Force = {force}, Error = {error}")
    return error, force




if __name__ == "__main__":
    with open("./april_tag_poses.json", "r") as f:
        poses = trans_coordinate(json.load(f))[:200]

    robot_pts = get_robot_contact_pt(poses)

    # Create environment and ground plane
    env = m.Env(render=RENDER)
    orient = m.get_quaternion([np.pi, 0, 0])

    # Investigate how the mass of the cube, the lateral_friction of the table, and the motor_force of the robot affects pushing the block.

    # Robot not strong enough to break static friction and push block
    scenario = dict(position=np.array([0.3, 0, 0.95]),
             table_friction=0.5,
             cube_mass=100,
             cube_friction=0.5,
             robot_force=50)
    # # Robot stronger and can break static friction
    # scenarios.append(
    #     dict(position=np.array([0.3, 0, 0.95]),
    #         table_friction=0.5,
    #         cube_mass=100,
    #         cube_friction=0.5,
    #         robot_force=100))

    # # Robot pushes cube along center axis (in a straight line)
    # scenarios.append(
    #     dict(position=np.array([0.3, 0, 0.95]),
    #         table_friction=0.5,
    #         cube_mass=1,
    #         cube_friction=0.5,
    #         robot_force=50))
    # # Robot pushes cube off-diagonal, causing the cube to spin
    # scenarios.append(
    #     dict(position=np.array([0.3, 0.05, 0.95]),
    #         table_friction=0.5,
    #         cube_mass=1,
    #         cube_friction=0.5,
    #         robot_force=50))
    # # Decrease friction between cube and robot. Watch how the end effector slides along the cube
    # scenarios.append(
    #     dict(position=np.array([0.3, 0.05, 0.95]),
    #         table_friction=0.5,
    #         cube_mass=1,
    #         cube_friction=0.01,
    #         robot_force=50))

    pool = Pool(128)

    results = pool.starmap(
        compute_obj,
        zip(range(100, 1000)))

    min_error = np.inf
    opt_force = None
    for error, force in results:
        if min_error > error[0]:
            min_error = error[0]
            opt_force = force
    print(f"Optimal force = {opt_force} with error {min_error}")


    # for force in range(100, 200):

        # for i in range(300):
        #     # Move the end effector to the left along a linear trajectory
        #     if pos[0] > -0.2:
        #         pos += np.array([-0.0025, 0, 0])
        #     target_joint_angles = robot.ik(robot.end_effector,
        #                                 target_pos=pos,
        #                                 target_orient=orient,
        #                                 use_current_joint_angles=True)
        #     robot.control(target_joint_angles, forces=s['robot_force'])

        #     m.step_simulation()

        #     # Show contact normals
        #     cp = robot.get_contact_points(bodyB=cube)
        #     m.clear_all_visual_items()
        #     if cp is not None:
        #         for c in cp:
        #             line = m.Line(c['posB'],
        #                         np.array(c['posB']) +
        #                         np.array(c['contact_normal']) * 0.2,
        #                         rgb=[1, 0, 0])
