import argparse
import sys
import time
import cv2
from queue import Queue

from mtcnn_cv2 import MTCNN

face_detector = MTCNN()

def run(camera_id: int, width: int, height: int,) -> None:
    """Continuously run inference on images acquired from the camera.

    Args:
        camera_id: The camera id to be passed to OpenCV.
        width: The width of the frame captured from the camera.
        height: The height of the frame captured from the camera.
    """

    # Variables to calculate FPS
    counter, fps = 0, 0

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization parameters
    row_size = 30  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # red
    font_size = 2
    font_thickness = 3
    fps_avg_frame_count = 10

    start_time = time.time()
    frame_lim = 20

    while cap.isOpened():
        success, image = cap.read()
        counter += 1
        if counter % frame_lim != 0:
            continue

        if not success:
            sys.exit('ERROR: Unable to read from webcam.')

        result = face_detector.detect_faces(image)

        if len(result) > 0:
            for face in result:
                # Place a rectangle around detected faces
                cv2.rectangle(image, face['box'], (0, 155, 255), 0)
                # Place dots on face features
                for _, face_feature in face['keypoints'].items():
                    cv2.circle(image, face_feature, 2, (0,155,255), 2)

        # Calculate the FPS
        if counter % fps_avg_frame_count == 0:
            fps = fps_avg_frame_count / (time.time() - start_time)
            fps = fps / frame_lim
            start_time = time.time()

        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(fps)
        text_location = (left_margin, row_size)
        cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

        cv2.imshow('object_detector', image)
    else:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--cameraId',
        help='Id of camera.',
        required=False,
        type=int,
        default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        type=int,
        default=640)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        type=int,
        default=480)
    args = parser.parse_args()

    run(int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
    main()
