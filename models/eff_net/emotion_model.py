import time

import numpy as np
import torch
from facenet_pytorch import MTCNN
from torch.nn.functional import softmax
from torchvision import transforms

from .dan import DAN


class EmotionModel:
    def __init__(self, weights_path: str, verbose: bool = False):
        self.mtcnn = MTCNN(keep_all=True, device='cpu')

        self.model = self._init_model(weights_path=weights_path)
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ]
        )

        self.class2emotion = {
            0: 'Surprise',
            1: 'Fear',
            2: 'Disgust',
            3: 'Happy',
            4: 'Sad',
            5: 'Angry',
            6: 'Neutral',
        }

        self.verbose = verbose

    @staticmethod
    def _init_model(weights_path):
        model = DAN(num_head=4, pretrained=False)
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model

    def detect(self, img) -> str:
        boxes, _ = self.mtcnn.detect(img)
        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = boxes[0].tolist()

            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            img = img[y1:y2, x1:x2]

        image = self.transform(img)
        image = image[None, :, :, :]

        start = time.time()
        out, feat, heads = self.model(image)
        end = time.time()

        _, predicts = torch.max(out, 1)
        probs = softmax(out).detach().numpy().tolist()
        emotion = self.class2emotion[np.argmax(probs[0])]

        if self.verbose:
            print(f'Time: {end - start}\tProbs: {probs}\tEmotion: {emotion}')

        return emotion
