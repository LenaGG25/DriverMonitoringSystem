import cv2
import numpy as np
from numpy import linalg as LA

MOUTH_LMS_NUMS = [78, 308, 81, 178, 311, 402]


class MouthDetector:
    @staticmethod
    def _calc_MAR(mouth_points):
        mar = (
                      LA.norm(mouth_points[2] - mouth_points[3]) + LA.norm(mouth_points[4] - mouth_points[5])
              ) / (2 * LA.norm(mouth_points[0] - mouth_points[1]))

        return mar

    @staticmethod
    def show_mouth_keypoints(color_frame, landmarks, frame_size):
        for n in MOUTH_LMS_NUMS:
            x = int(landmarks[n, 0] * frame_size[0])
            y = int(landmarks[n, 1] * frame_size[1])
            cv2.circle(color_frame, (x, y), 1, (255, 0, 0), 5)

    def get_MAR(self, landmarks):
        mouth_pts = np.zeros(shape=(6, 2))

        for i in range(len(MOUTH_LMS_NUMS)):
            mouth_pts[i] = landmarks[MOUTH_LMS_NUMS[i], :2]

        return self._calc_MAR(mouth_pts)
