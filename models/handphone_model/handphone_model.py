import time
import cv2

from ultralytics import YOLO


class HandphoneYOLO:
    def __init__(self, weights_path: str, verbose: bool = False):
        self.model = YOLO(model=weights_path)

        self.verbose = verbose

    def detect(self, img) -> tuple[bool, float]:
        start = time.time()
        results = self.model.predict(img, verbose=False)
        end = time.time()

        is_detected = False
        conf = 0

        if len(results) != 0:
            boxes = results[0].boxes

            cls = boxes.cls.numpy()
            conf = boxes.conf.numpy()
            if len(cls) != 0:
                is_detected = True
                conf = conf[0]

        if self.verbose:
            print(f'Time: {end-start}\tIs Detected: {is_detected}\tConfidence: {conf}')

        return is_detected, conf
