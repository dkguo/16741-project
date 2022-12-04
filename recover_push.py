import os
import json
import numpy as np
import mengine as m
from scipy.spatial.transform import Rotation
from multiprocessing import Pool

OBJECT_ORI_POS_Z = 0.905
RENDER = False
SIM_SEED = 100
EXE_SEED = 32

P = 10

rng = np.random.default_rng(seed=100)


def reset(env, position, table_friction=0.5, cube_mass=100, cube_friction=0.5):
    # Create environment and ground plane
    env.reset()
    ground = m.Ground()

    # Create table and cube
    table = m.URDF(filename=os.path.join(m.directory, 'table', 'table.urdf'),
                   static=True,
                   position=[0, 0, 0],
                   orientation=[0, 0, 0, 1],
                   maximal_coordinates=True,
                   env=env)
    table.set_whole_body_frictions(
        lateral_friction=table_friction,
        spinning_friction=0,
        rolling_friction=0,
    )
    # cube = None
    cube = m.Shape(m.Box(half_extents=[0.25, 0.14, 0.16]),
                   static=False,
                   mass=cube_mass,
                   position=[0, 0, OBJECT_ORI_POS_Z],
                   orientation=[0, 0, 0, 0.8],
                   rgba=[0, 1, 0, 1],
                   env=env)
    cube.set_whole_body_frictions(lateral_friction=cube_friction,
                                  spinning_friction=0,
                                  rolling_friction=0)

    true_cube = m.Shape(m.Box(half_extents=[0.25, 0.14, 0.16]),
                        static=False,
                        mass=0,
                        position=[0, 0, OBJECT_ORI_POS_Z],
                        orientation=[0, 0, 0, 1],
                        rgba=[1, 0, 0, 0.8],
                        collision=False,
                        env=env)

    # Create Panda robot
    robot = m.Robot.Panda(position=[0.5, 0, 0.75], env=env)

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


def get_robot_contact_pt(trans_poses, x_offset=0.3, z_offset=-0.02):
    all_robot_pt = []
    T = np.eye(4)
    T[0, 3] = x_offset
    T[2, 3] = z_offset
    for pose in trans_poses:
        robot_pt = T @ pose
        all_robot_pt.append(robot_pt)
    return all_robot_pt


def compute_traj_diff(pred_traj, true_traj, weight=1):
    pose_diff_sum = 0
    ori_diff_sum = 0
    n_pose = len(pred_traj)
    for pred_pose, true_pose in zip(pred_traj, true_traj):
        pose_diff = np.linalg.norm(pred_pose[:3, -1] - true_pose[:3, -1])
        pred_ori_quat = Rotation.from_matrix(pred_pose[:3, :3]).as_quat()
        true_ori_quat = Rotation.from_matrix(true_pose[:3, :3]).as_quat()
        ori_diff = 1 - np.dot(pred_ori_quat, true_ori_quat)**2
        pose_diff_sum += pose_diff
        ori_diff_sum += ori_diff
    mean_error = (pose_diff_sum * weight + ori_diff_sum) / n_pose
    pose_error = pose_diff_sum / n_pose
    ori_error = ori_diff_sum / n_pose
    return mean_error, pose_error, ori_error


# def compute_obj(force, seed):
#     # Reset simulator
#     env.seed(seed)
#     robot, cube, true_cube = reset(
#         scenario['position'],
#         scenario['table_friction'],
#         scenario['cube_mass'],
#         scenario['cube_friction'],
#     )

#     target_joint_angles = robot.ik(robot.end_effector,
#                                    target_pos=robot_pts[0][:3, -1],
#                                    target_orient=orient,
#                                    use_current_joint_angles=True)
#     robot.set_joint_angles(target_joint_angles)

#     pred_cube_poses = []
#     for pose, robot_pt in zip(poses, robot_pts):
#         R = pose[:3, :3]
#         q = Rotation.from_matrix(R).as_quat()
#         # cube.set_base_pos_orient(pose[:3,-1], q)
#         true_cube.set_base_pos_orient(pose[:3, -1], q)

#         target_joint_angles = robot.ik(robot.end_effector,
#                                        target_pos=robot_pt[:3, -1],
#                                        target_orient=orient,
#                                        use_current_joint_angles=True)
#         robot.control(target_joint_angles, forces=force)
#         m.step_simulation(realtime=False)
#         pred_cube_p, pred_cube_q = cube.get_base_pos_orient()
#         R = Rotation.from_quat(pred_cube_q).as_matrix()
#         T = np.r_[np.c_[R, pred_cube_p], np.array([[0, 0, 0, 1]])]
#         pred_cube_poses.append(T)

#     error = compute_traj_diff(pred_cube_poses, poses)
#     print(f"Force = {force}, Error = {error}")
#     return error, force


def simulate_one_step(mode,
                      robot_pt,
                      force,
                      cube_pose=None,
                      robot_joint_angles=None,
                      desire_cube_pose=None):
    """
    :param cube_pose: from last step
    :param robot_joint_angles: from last step
    :param robot_pt: the desired pt for robot to push

    :return: new cube pose (4, 4)
    """
    if mode == 'EXECUTE':
        env = exe_env
        env.seed(EXE_SEED)
        robot = exe_robot
        cube = exe_cube
        R = desire_cube_pose[:3, :3]
        q = Rotation.from_matrix(R).as_quat()
        exe_true_cube.set_base_pos_orient(desire_cube_pose[:3, -1], q)
    elif mode == 'SIMULATE':
        env = sim_env
        env.seed(SIM_SEED)
        # Reset simulator
        robot, cube, true_cube = reset(
            env,
            scenario['position'],
            scenario['table_friction'],
            scenario['cube_mass'],
            scenario['cube_friction'],
        )
        robot.set_joint_angles(robot_joint_angles)
        R = cube_pose[:3, :3]
        q = Rotation.from_matrix(R).as_quat()
        cube.set_base_pos_orient(cube_pose[:3, -1], q)

    target_joint_angles = robot.ik(robot.end_effector,
                                   target_pos=robot_pt,
                                   target_orient=orient,
                                   use_current_joint_angles=True)
    robot.control(target_joint_angles, forces=int(force))
    m.step_simulation(env=env)

    new_cube_p, new_cube_q = cube.get_base_pos_orient()
    R = Rotation.from_quat(new_cube_q).as_matrix()
    new_cube_pose = np.r_[np.c_[R, new_cube_p], np.array([[0, 0, 0, 1]])]
    new_robot_joint_angles = robot.get_joint_angles()
    return new_cube_pose, new_robot_joint_angles


def search_action_one_step(
    cube_pose,
    robot_joint_angles,
    desired_cube_pose,
    prev_error,
):
    """
    :param cube_pose: from last step
    :param robot_joint_angles: from last step
    :param desired_cube_pose: true cube pose for next step (4, 4)

    :return: Optimal robot push pt and force for next step u_i+1
    """
    # search next action
    # propose new robot pt and forces
    num_search = 10
    contact_pt = get_robot_contact_pt([cube_pose])[0][:3, -1]
    # robot_pts = np.zeros((num_search, 3))
    # robot_pts[:,1:] = rng.random(size=(num_search, 2)) * 0.05 + contact_pt[1:]
    # robot_pts[:,0] = contact_pt[0]
    robot_pts = rng.uniform(low=-0.1, high=0.1,
                            size=(num_search, 3)) + contact_pt
    robot_pts[0] -= P * prev_error
    forces = rng.integers(low=100, high=200, size=num_search)

    # prepare input for simulate_one_step
    # cube_pose, robot_joint_angles, robot_pt, force
    inputs = []
    for i in range(num_search):
        inputs.append(('SIMULATE', robot_pts[i], forces[i], cube_pose,
                       robot_joint_angles))

    # Simulate each step
    pool = Pool(128)
    results = pool.starmap(simulate_one_step, inputs)

    # results = []
    # for inp in inputs:
    #     results.append(simulate_one_step(*inp))

    # Calculate error of each step and find optimal robot pt
    min_error = np.inf
    opt_pt = None
    opt_force = None
    for i in range(num_search):
        new_cube_pose = results[i][0]
        errors = compute_traj_diff([new_cube_pose], [desired_cube_pose])
        if min_error > errors[0]:
            min_error = errors[0]
            opt_pt = robot_pts[i]
            opt_force = forces[i]
    print(f"Optimal push pt: {opt_pt}, force: {opt_force}, error: {min_error}")

    return opt_pt, opt_force, min_error


if __name__ == "__main__":
    # Create environment and ground plane
    scenario = dict(position=np.array([0.3, 0, 0.95]),
                    table_friction=0.5,
                    cube_mass=100,
                    cube_friction=0.5,
                    robot_force=50)
    sim_env = m.Env(render=False)
    orient = m.get_quaternion([np.pi, 0, 0])
    exe_env = m.Env(render=False)
    exe_env.seed(EXE_SEED)
    exe_robot, exe_cube, exe_true_cube = reset(
        exe_env,
        scenario['position'],
        scenario['table_friction'],
        scenario['cube_mass'],
        scenario['cube_friction'],
    )

    with open("./april_tag_poses.json", "r") as f:
        desire_cube_poses = trans_coordinate(json.load(f))[:10]

    # set exe_robot to somewhere near the first pose
    robot_pt = get_robot_contact_pt(
        [desire_cube_poses[0]],
        x_offset=0.5,
    )[0][:3, -1]
    cube_pose, robot_joint_angles = simulate_one_step(
        'EXECUTE',
        robot_pt,
        force=100,
        desire_cube_pose=desire_cube_poses[0],
    )

    # loop through all poses
    robot_control_result = []
    prev_error = 0
    for desire_cube_pose in desire_cube_poses:
        contact_pt = get_robot_contact_pt([cube_pose], x_offset=0.28)[0][:3,
                                                                         -1]
        print(f"Contact pt: {contact_pt}")
        robot_pt, force, min_error = search_action_one_step(
            cube_pose,
            robot_joint_angles,
            desire_cube_pose,
            prev_error,
        )
        robot_control_result.append((robot_pt, force))
        cube_pose, robot_joint_angles = simulate_one_step(
            'EXECUTE',
            robot_pt,
            force,
            desire_cube_pose=desire_cube_pose,
        )

        prev_error = min_error

    # Simulate final result
    exe_env = m.Env(render=True)
    exe_env.seed(EXE_SEED)
    exe_robot, exe_cube, exe_true_cube = reset(
        exe_env,
        scenario['position'],
        scenario['table_friction'],
        scenario['cube_mass'],
        scenario['cube_friction'],
    )

    # set exe_robot to somewhere near the first pose
    robot_pt = get_robot_contact_pt(
        [desire_cube_poses[0]],
        x_offset=0.5,
    )[0][:3, -1]
    cube_pose, robot_joint_angles = simulate_one_step(
        'EXECUTE',
        robot_pt,
        force=100,
        desire_cube_pose=desire_cube_poses[0],
    )

    for (robot_pt, force), desire_cube_pose in zip(
            robot_control_result,
            desire_cube_poses,
    ):
        simulate_one_step(
            'EXECUTE',
            robot_pt,
            force,
            desire_cube_pose=desire_cube_pose,
        )

    # Investigate how the mass of the cube, the lateral_friction of the table, and the motor_force of the robot affects pushing the block.

    # pool = Pool(128)

    # results = pool.starmap(compute_obj, zip([100] * 20, range(20)))

    # min_error = np.inf
    # opt_force = None
    # for error, force in results:
    #     if min_error > error[0]:
    #         min_error = error[0]
    #         opt_force = force
    # print(f"Optimal force = {opt_force} with error {min_error}")

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