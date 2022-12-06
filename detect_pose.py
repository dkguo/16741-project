import json
import cv2
from apriltag_detection import detect_april_tag


if __name__ == '__main__':
    cap = cv2.VideoCapture('./data/pushing_straight.mp4')
    camera_params = (615.587890625, 615.9234619140625, 326.09588623046875, 242.5946044921875)

    tag_size = 0.08

    poses = []
    while (cap.isOpened()):
        ret, image = cap.read()
        poses_errs, overlay = detect_april_tag(image, camera_params, tag_size, visualize=False)

        found_pose = False
        for tag_id, pose, error in poses_errs:
            if tag_id == 0:
                poses.append(pose.tolist())
                found_pose = True
                break
        if not found_pose:
            poses.append(None)

    print(poses)
    with open('./data/poses.mp4', 'w') as f:
        json.dump(poses, f, indent=4)
