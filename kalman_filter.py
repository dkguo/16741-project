import json

import numpy as np


def load_time(time_file_path):
    with open(time_file_path) as f:
        return json.load(f).values()


def load_object_poses(file_path):
    """
    This function loads object poses assuming there is only one identical object in each frame.
    :param file_path: the file has dict[frame][object_id] = pose

    :return: object_poses: dict[object_id] = pose
    """
    with open(file_path) as f:
        frame_object_poses = json.load(f)

    # find obj keys
    obj_ids = set()
    for object_poses in frame_object_poses.values():
        for obj_id in object_poses.keys():
            obj_ids.add(obj_id)

    object_poses = {}
    for obj_id in obj_ids:
        poses = []
        last_frame = int(list(frame_object_poses.keys())[-1])
        for frame in range(last_frame + 1):
            if str(frame) in frame_object_poses:
                if obj_id in frame_object_poses[str(frame)]:
                    pose = np.array(frame_object_poses[str(frame)][obj_id])
                    assert pose.shape == (4, 4)
                    poses.append(frame_object_poses[str(frame)][obj_id])
                else:
                    poses.append(None)
            else:
                poses.append(None)
        object_poses[obj_id] = poses

    return object_poses


def kalman_filter(times, poses):
    """
    :param times: list of time from frame 0 to end
    :param poses: list of pose from frame 0 to end

    :return zs: estimated poses
    """
    pass


if __name__ == '__main__':
    times = load_time('./time.json')
    object_poses = load_object_poses('./object_poses.json')

    object_est_poses = {}
    for object_id, poses in object_poses.items():
        object_est_poses[object_id] = kalman_filter(times, poses)
