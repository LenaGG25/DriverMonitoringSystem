from enum import Enum

import cv2
import math
import numpy as np

RIGHT_HAND_LMS = [12, 14, 16]
LEFT_HAND_LMS = [11, 13, 15]


class Hand(Enum):
    RIGHT = 1
    LEFT = 2


class HandphoneDetector:
    def __init__(self, hand: Hand):
        match hand:
            case Hand.RIGHT:
                self.lms = RIGHT_HAND_LMS
            case Hand.LEFT:
                self.lms = LEFT_HAND_LMS

    def show_hand_keypoints(self, color_frame, landmarks, frame_size):
        for n in self.lms:
            x1 = int(landmarks[n, 0] * frame_size[0])
            y1 = int(landmarks[n, 1] * frame_size[1])
            cv2.circle(color_frame, (x1, y1), 1, (255, 255, 0), 10)

    @staticmethod
    def _compute_angle(p0, p1, p2):
        """
        compute angle (in degrees) for p0p1p2 corner
        Inputs:
            p0,p1,p2 - points in the form of [x,y]
        """
        v0 = np.array(p0) - np.array(p1)
        v1 = np.array(p2) - np.array(p1)

        angle = math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
        return np.degrees(angle)

    def get_angle(self, landmarks, frame_size):
        x1 = int(landmarks[self.lms[0], 0] * frame_size[0])
        y1 = int(landmarks[self.lms[0], 1] * frame_size[1])

        x2 = int(landmarks[self.lms[1], 0] * frame_size[0])
        y2 = int(landmarks[self.lms[1], 1] * frame_size[1])

        x3 = int(landmarks[self.lms[2], 0] * frame_size[0])
        y3 = int(landmarks[self.lms[2], 1] * frame_size[1])

        return self._compute_angle(
            [x3, y3],
            [x2, y2],
            [x1, y1],
        )

