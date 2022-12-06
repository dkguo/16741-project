import glob

import cv2

if __name__ == '__main__':
    image_paths = sorted(glob.glob(f'/Users/gdk/Downloads/pushing/scene_2212031401/camera_03_827112072509/rgb/*.png'))

    out = cv2.VideoWriter('pushing.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, (640, 480))

    for i in range(200):
        img = cv2.imread(image_paths[i])
        out.write(img)
    out.release()