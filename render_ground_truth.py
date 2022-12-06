import os
import json
import numpy as np
import mengine as m
from scipy.spatial.transform import Rotation
from multiprocessing import Pool

from recover_push import trans_coordinate

OBJECT_ORI_POS_Z = 0.905
RENDER = False
SIM_SEED = 100
EXE_SEED = 32

P = 10
N_POINTS = 200
SEARCH_RANGE = 0.1
N_SAMPLE = 100

rng = np.random.default_rng(seed=100)

scenario = dict(position=np.array([0.3, 0, 0.95]),
                table_friction=0.5,
                cube_mass=100,
                cube_friction=0.5,
                robot_force=50)


def reset(env, position, table_friction=0.5, cube_mass=100, cube_friction=0.5):
    # Create environment and ground plane
    env.reset()
    orient = m.get_quaternion([np.pi, 0, 0])
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

    true_cube = m.Shape(m.Box(half_extents=[0.25, 0.14, 0.16]),
                        static=False,
                        mass=0,
                        position=[0, 0, OBJECT_ORI_POS_Z],
                        orientation=[0, 0, 0, 1],
                        rgba=[1, 0, 0, 0.8],
                        collision=False,
                        env=env)

    return true_cube


def main():
    # Create environment
    env = m.Env(render=True)
    env.seed(EXE_SEED)
    true_cube = reset(
        env,
        scenario['position'],
        scenario['table_friction'],
        scenario['cube_mass'],
        scenario['cube_friction'],
    )

    with open("data/april_tag_poses.json", "r") as f:
        desire_cube_poses = trans_coordinate(json.load(f))[:N_POINTS]

    # wait 1000 step to adjust camera angle
    for i in range(500):
        m.step_simulation()

    # loop through all poses
    for desire_cube_pose in desire_cube_poses:
        R = desire_cube_pose[:3, :3]
        q = Rotation.from_matrix(R).as_quat()
        true_cube.set_base_pos_orient(desire_cube_pose[:3, -1], q)
        m.step_simulation(env=env)


if __name__ == "__main__":
    main()
