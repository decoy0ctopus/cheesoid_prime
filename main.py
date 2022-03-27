import argparse
import threading

import detect
import avoidance


if __name__ == '__main__':

	threading.Thread(target=detect.main, args=()).start()
	
	threading.Thread(target=avoidance.run, args=()).start()