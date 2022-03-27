"""Main script to run the object detection routine."""
import argparse
import sys
import time
import cv2
from object_detector import ObjectDetector
from object_detector import ObjectDetectorOptions
import utils as utils
import datetime
import base64
import json
from tracker import Tracker
import numpy as np
from PIL import Image

from mtcnn_cv2 import MTCNN

face_detector = MTCNN()

# Initialize Tracker
tracker = Tracker()


def image_to_base64(img: np.ndarray) -> str:
    """ Given a numpy 2D array, returns a JPEG image in base64 format """
    # using opencv 2, there are others ways
    img_buffer = cv2.imencode('.jpg', img)[1]
    return base64.b64encode(img_buffer).decode('utf-8')


def run(
	model: str,
	camera_id: int,
	width: int,
	height: int,
	num_threads: int,
	enable_edgetpu: bool,
	time_limit=30
) -> None:
	"""Continuously run inference on images acquired from the camera.
	Args:
	model: Name of the TFLite object detection model.
	camera_id: The camera id to be passed to OpenCV.
	width: The width of the frame captured from the camera.
	height: The height of the frame captured from the camera.
	num_threads: The number of CPU threads to run the model.
	enable_edgetpu: True/False whether the model is a EdgeTPU model.
	"""

	# Variables to calculate FPS
	counter, fps = 0, 0
	start_time = time.time()

	# Start capturing video input from the camera
	cap = cv2.VideoCapture(camera_id)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

	# Visualization parameters
	row_size = 20  # pixels
	left_margin = 24  # pixels
	text_color = (0, 0, 255)  # red
	font_size = 1
	font_thickness = 1
	fps_avg_frame_count = 10

	# Initialize the object detection model
	options = ObjectDetectorOptions(
		num_threads=num_threads,
		score_threshold=0.7,
		max_results=3,
		enable_edgetpu=enable_edgetpu)
	detector = ObjectDetector(model_path=model, options=options)

	# Continuously capture images from the camera and run inference
	output_list = []
	countdown = time.time() + time_limit

	while cap.isOpened() and time.time() < countdown:
		success, image = cap.read()

		if not success:
			sys.exit(
			'ERROR: Unable to read from webcam. Please verify your webcam settings.'
			)

		counter += 1
		#image = cv2.flip(image, 1)

		# Run object detection estimation using the model.
		detections = detector.detect(image)

		result = face_detector.detect_faces(image)

		if len(result) > 0:
			for face in result:
				keypoints = face['keypoints']
				# Place a rectangle around detected faces
				cv2.rectangle(image, face['box'], (0, 155, 255), 0)
				# Place dots on face features
				for key, face_feature in face['keypoints'].items():
					cv2.circle(image, face_feature, 2, (0,155,255), 2)

		bboxes = []
		confidences = []
		class_ids = []

		for detection in detections:
			bboxes.append(detection.bounding_box)
			confidences.append(detection.categories[0].score)
			class_ids.append(detection.categories[0].label)
			left, top, right, bottom = detection.bounding_box

		tracks = tracker.update(bboxes, confidences, class_ids)
		for track in tracks:
			track_age = track[8]

		image = utils.draw_tracks(image, tracks)

		# Draw keypoints and edges on input image
		image = utils.visualize(image, detections)

		# Calculate the FPS
		if counter % fps_avg_frame_count == 0:
			end_time = time.time()
			fps = fps_avg_frame_count / (end_time - start_time)
			start_time = time.time()

		# Show the FPS
		fps_text = 'FPS = {:.1f}'.format(fps)
		text_location = (left_margin, row_size)
		print(fps_text)
		cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
				font_size, text_color, font_thickness)

		# Stop the program if the ESC key is pressed.
		if cv2.waitKey(1) == 27:
			break
		cv2.imshow('object_detector', image)



	cap.release()
	print("Started writing JSON data into a file")
	with open("test.json", "w") as write_file:
		json.dump(output_list, write_file) # encode dict into JSON
	print("Done writing JSON data into .json file")

	cv2.destroyAllWindows()


def main():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument(
		'--model',
		help='Path of the object detection model.',
		required=False,
		default='cv/efficientdet_lite0.tflite')
	parser.add_argument(
		'--cameraId', help='Id of camera.', required=False, type=int, default=0)
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
	parser.add_argument(
		'--numThreads',
		help='Number of CPU threads to run the model.',
		required=False,
		type=int,
		default=4)
	parser.add_argument(
		'--enableEdgeTPU',
		help='Whether to run the model on EdgeTPU.',
		action='store_true',
		required=False,
		default=False)
	args = parser.parse_args()

	run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
		int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
	main()