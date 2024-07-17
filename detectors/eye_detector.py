import cv2
import numpy as np
from numpy import linalg as LA

EYES_LMS_NUMS = [33, 133, 160, 144, 158, 153, 362, 263, 385, 380, 387, 373]
LEFT_IRIS_NUM = 468
RIGHT_IRIS_NUM = 473


class EyeDetector:
    @staticmethod
    def _calc_EAR_eye(eye_pts):
        ear_eye = (
            LA.norm(eye_pts[2] - eye_pts[3]) + LA.norm(eye_pts[4] - eye_pts[5])
        ) / (2 * LA.norm(eye_pts[0] - eye_pts[1]))

        return ear_eye

    @staticmethod
    def show_eye_keypoints(color_frame, landmarks, frame_size):
        cv2.circle(
            color_frame,
            (landmarks[LEFT_IRIS_NUM, :2] * frame_size).astype(np.uint32),
            3,
            (255, 255, 255),
            cv2.FILLED,
        )
        cv2.circle(
            color_frame,
            (landmarks[RIGHT_IRIS_NUM, :2] * frame_size).astype(np.uint32),
            3,
            (255, 255, 255),
            cv2.FILLED,
        )

        for n in EYES_LMS_NUMS:
            x = int(landmarks[n, 0] * frame_size[0])
            y = int(landmarks[n, 1] * frame_size[1])
            cv2.circle(
                color_frame,
                (x, y),
                2,
                (0, 0, 255),
                -1,
            )

    def get_EAR(self, landmarks):
        eye_pts_l = np.zeros(shape=(6, 2))
        eye_pts_r = eye_pts_l.copy()

        for i in range(len(EYES_LMS_NUMS) // 2):
            eye_pts_l[i] = landmarks[EYES_LMS_NUMS[i], :2]
            eye_pts_r[i] = landmarks[EYES_LMS_NUMS[i + 6], :2]

        ear_left = self._calc_EAR_eye(eye_pts_l)
        ear_right = self._calc_EAR_eye(eye_pts_r)

        ear_avg = (ear_left + ear_right) / 2

        return ear_avg
