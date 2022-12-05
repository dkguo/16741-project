import glob
import json

import cv2
from tqdm import tqdm

from apriltag_detection import detect_april_tag, apriltag_detect_error_thres


if __name__ == '__main__':

    image_paths = sorted(glob.glob(f'/Users/gdk/Downloads/pushing/scene_2212031401/camera_03_827112072509/rgb/*.png'))
    camera_params = (615.587890625, 615.9234619140625, 326.09588623046875, 242.5946044921875)
    tag_size = 0.08

    poses = []
    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)
        poses_errs, overlay = detect_april_tag(image, camera_params, tag_size, visualize=True)

        found_pose = False
        for tag_id, pose, error in poses_errs:
            if tag_id == 0:
                poses.append(pose.tolist())
                found_pose = True
                break
        if not found_pose:
            poses.append(None)

    print(poses)
    with open('/Users/gdk/Downloads/pushing/scene_2212031401/camera_03_827112072509/april_tag_poses.json', 'w') as f:
        json.dump(poses, f, indent=4)
