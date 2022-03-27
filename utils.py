# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions to display the pose detection results."""

from typing import List

import cv2
import numpy as np
from object_detector import Detection

_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red

# Function for finding the center of a rectangle
def find_center(x, y, w, h):
	x1=int(w/2)
	y1=int(h/2)
	cx = x+x1
	cy=y+y1
	return cx, cy

def visualize(
image: np.ndarray,
detections: List[Detection],
) -> np.ndarray:
	"""Draws bounding boxes on the input image and return it.

	Args:
	image: The input RGB image.
	detections: The list of all "Detection" entities to be visualize.

	Returns:
	Image with bounding boxes.
	"""
	for detection in detections:
		# Draw bounding_box
		start_point = detection.bounding_box.left, detection.bounding_box.top
		end_point = detection.bounding_box.right, detection.bounding_box.bottom
		cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

		# Draw label and score
		category = detection.categories[0]
		class_name = category.label
		probability = round(category.score, 2)
		result_text = class_name + ' (' + str(probability) + ')'
		text_location = (_MARGIN + detection.bounding_box.left,
							_MARGIN + _ROW_SIZE + detection.bounding_box.top)
		cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
					_FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

	return image

def get_centroid(bboxes):
	"""
	Calculate centroids for multiple bounding boxes.
	Args:
		bboxes (numpy.ndarray): Array of shape `(n, 4)` or of shape `(4,)` where
		each row contains `(xmin, ymin, width, height)`.
	Returns:
		numpy.ndarray: Centroid (x, y) coordinates of shape `(n, 2)` or `(2,)`.
	"""

	one_bbox = False
	if len(bboxes.shape) == 1:
		one_bbox = True
		bboxes = bboxes[None, :]

	xmin = bboxes[:, 0]
	ymin = bboxes[:, 1]
	w, h = bboxes[:, 2], bboxes[:, 3]

	xc = xmin + 0.5*w
	yc = ymin + 0.5*h

	x = np.hstack([xc[:, None], yc[:, None]])

	if one_bbox:
		x = x.flatten()
	return x

def draw_tracks(image, tracks):
    """
    Draw on input image.
    Args:
        image (numpy.ndarray): image
        tracks (list): list of tracks to be drawn on the image.
    Returns:
        numpy.ndarray: image with the track-ids drawn on it.
    """

    for trk in tracks:

        trk_id = trk[1]
        xmin = trk[2]
        ymin = trk[3]
        width = trk[4]
        height = trk[5]

        xcentroid, ycentroid = int(xmin + 0.5*width), int(ymin + 0.5*height)

        text = "ID {}".format(trk_id)

        cv2.putText(image, text, (xcentroid - 10, ycentroid - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(image, (xcentroid, ycentroid), 4, (0, 255, 0), -1)

    return image