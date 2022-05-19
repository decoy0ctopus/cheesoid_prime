import argparse

from detect import run

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