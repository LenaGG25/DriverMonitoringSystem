import time
import numpy as np
from ultralytics import YOLO


class DmsYOLO:
    def __init__(self, weights_path: str, verbose: bool = False):
        self.model = YOLO(model=weights_path)
        self.class2idx = {
            'closed_eye': 1,
            'cigarette': 2,
        }

        self.verbose = verbose

    def detect(self, img) -> dict[str, dict[str, [bool | float]]]:
        start = time.time()
        results = self.model.predict(img, classes=list(self.class2idx.values()), verbose=False)
        end = time.time()

        detections = {
            'closed_eye': {
                'is_detected': False,
                'conf': 0,
            },
            'cigarette': {
                'is_detected': False,
                'conf': 0,
            },
        }

        if len(results) > 0:
            boxes = results[0].boxes

            cls = boxes.cls.numpy()
            conf = boxes.conf.numpy()

            detections['closed_eye']['is_detected'] = self.class2idx['closed_eye'] in cls
            detections['cigarette']['is_detected'] = self.class2idx['cigarette'] in cls

            if detections['closed_eye']['is_detected']:
                detections['closed_eye']['conf'] = np.max(conf[np.where(cls == self.class2idx['closed_eye'])[0]])

            if detections['cigarette']['is_detected']:
                detections['cigarette']['conf'] = np.max(conf[np.where(cls == self.class2idx['cigarette'])[0]])

        if self.verbose:
            print(f'Time: {end-start}\tDetections: {detections}')

        return detections
