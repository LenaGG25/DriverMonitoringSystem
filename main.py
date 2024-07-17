import time

import click
import cv2
import mediapipe as mp
import numpy as np

from models.dms_model.dms_model import DmsYOLO
from models.handphone_model.handphone_model import HandphoneYOLO
from models.eff_net.emotion_model import EmotionModel
from detectors.handphone_detector import Hand, HandphoneDetector
from detectors.eye_detector import EyeDetector
from detectors.mouth_detector import MouthDetector
from state_manager.state_manager import *


@click.command()
@click.option(
    '-c',
    '--camera',
    default=0,
    type=int,
    help='Camera number, default is 0 (webcam)',
)
@click.option(
    '--show_fps',
    type=bool,
    default=False,
    help='Show the processing time for a single frame, default is true',
)
@click.option(
    '--show_proc_time',
    type=bool,
    default=False,
    help='Show the processing time for a single frame, default is true',
)
@click.option(
    '--hand',
    type=bool,
    default=False,
    help='Right(true) or left(false) hand',
)
@click.option(
    '--debug',
    type=bool,
    default=False,
    help='Debug mode',
)

@click.option(
    '--face-min-detection-confidence',
    type=float,
    default=0.5,
)
@click.option(
    '--face-min-tracking-confidence',
    type=float,
    default=0.5,
)
@click.option(
    '--pose-min-detection-confidence',
    type=float,
    default=0.3,
)
@click.option(
    '--pose-model-complexity',
    type=int,
    default=2,
)
@click.option(
    '--handphone-weights-path',
    type=str,
    default='assets/yolo_handphone.pt',
)
@click.option(
    '--dms-weights-path',
    type=str,
    default='assets/yolo_handphone.pt',
)
@click.option(
    '--emotion-weights-path',
    type=str,
    default='assets/yolo_handphone.pt',
)
@click.option(
    '--fps',
    type=int,
    default=1,
)
def main(
    camera: int,
    show_fps: bool,
    show_proc_time: bool,
    face_min_detection_confidence: float,
    face_min_tracking_confidence: float,
    pose_min_detection_confidence: float,
    pose_model_complexity: int,
    handphone_weights_path: str,
    dms_weights_path: str,
    emotion_weights_path: str,
    hand: bool,
    debug: bool,
    fps: int,
):
    color = (51, 255, 51)

    fps = fps

    if not cv2.useOptimized():
        try:
            cv2.setUseOptimized(True)  # set OpenCV optimization to True
        except Exception:
            print(
                "OpenCV optimization could not be set to True, the script may be slower than expected"
            )

    if hand:
        hand = Hand.RIGHT
    else:
        hand = Hand.LEFT

    face_detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        min_detection_confidence=face_min_detection_confidence,
        min_tracking_confidence=face_min_tracking_confidence,
        refine_landmarks=True,
    )
    pose_detector = mp.solutions.pose.Pose(
        static_image_mode=True,
        min_detection_confidence=pose_min_detection_confidence,
        model_complexity=pose_model_complexity,
    )
    handphone_model = HandphoneYOLO(
        weights_path=handphone_weights_path,
        verbose=False,
    )
    dms_model = DmsYOLO(
        weights_path=dms_weights_path,
        verbose=False,
    )
    emotion_model = EmotionModel(
        weights_path=emotion_weights_path,
        verbose=True,
    )

    phone_state_manager = PhoneStateManager(
        n=fps * 6,
        alert_count=fps * 4,
        angle_threshold=35,
        conf_threshold=0.8,
    )
    eyes_state_manager = EyesStateManager(
        n=fps * 4,
        alert_count=fps * 2,
        ear_threshold=0.15,
        conf_threshold=0.8,
    )
    yawn_state_manager = YawnStateManager(
        n=fps * 60,
        alert_count=fps * 5,
        mar_threshold=0.9,
    )
    cigarette_state_manager = CigaretteStateManager(
        n=fps * 6,
        alert_count=fps * 1,
        conf_threshold=0.2,
    )
    emotion_state_manager = EmotionStateManager(
        n=fps * 10,
        alert_count=fps * 6,
    )

    eye_det = EyeDetector()
    mouth_det = MouthDetector()
    handphone_det = HandphoneDetector(hand=hand)

    t0 = time.perf_counter()

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    is_detected, conf = False, 0
    dms_detections = {
        'closed_eye': {
            'is_detected': False,
            'conf': 0,
        },
        'cigarette': {
            'is_detected': False,
            'conf': 0,
        },
    }

    i = 0
    time.sleep(0.0001)
    while True:
        t_now = time.perf_counter()
        fps = i / (t_now - t0)
        if fps == 0:
            fps = 10

        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame from camera/stream end")
            break

        if camera == 0:
            frame = cv2.flip(frame, 2)

        e1 = cv2.getTickCount()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_size = frame.shape[1], frame.shape[0]
        gray = np.expand_dims(cv2.bilateralFilter(gray, 5, 10, 10), axis=2)
        gray = np.concatenate([gray, gray, gray], axis=2)

        pose_lms = pose_detector.process(gray).pose_landmarks
        if pose_lms:
            landmarks = np.array([np.array([point.x, point.y, point.z]) for point in pose_lms.landmark])

            handphone_det.show_hand_keypoints(
                color_frame=frame, landmarks=landmarks, frame_size=frame_size,
            )

            angle = handphone_det.get_angle(
                landmarks=landmarks, frame_size=frame_size,
            )

            cv2.putText(
                img=frame,
                text="Angle: " + str(int(angle)),
                org=(10, 50),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=3,
                color=(51, 255, 51),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            is_detected, conf = handphone_model.detect(gray)

            phone_state_manager.update_state(angle, is_detected, conf)

        lms = face_detector.process(gray).multi_face_landmarks
        if lms:
            landmarks = get_face_landmarks(lms)

            eye_det.show_eye_keypoints(
                color_frame=frame, landmarks=landmarks, frame_size=frame_size,
            )
            mouth_det.show_mouth_keypoints(
                color_frame=frame, landmarks=landmarks, frame_size=frame_size,
            )

            ear = eye_det.get_EAR(landmarks=landmarks)
            mar = mouth_det.get_MAR(landmarks=landmarks)

            if ear is not None:
                cv2.putText(
                    frame,
                    "EAR:" + str(round(ear, 3)),
                    (10, 100),
                    cv2.FONT_HERSHEY_PLAIN,
                    3,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            if mar is not None:
                cv2.putText(
                    frame,
                    "MAR:" + str(round(mar, 3)),
                    (10, 150),
                    cv2.FONT_HERSHEY_PLAIN,
                    3,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            dms_detections = dms_model.detect(gray)

            eyes_state_manager.update_state(
                ear,
                dms_detections['closed_eye']['is_detected'],
                dms_detections['closed_eye']['conf'],
            )
            yawn_state_manager.update_state(mar)
            cigarette_state_manager.update_state(
                dms_detections['cigarette']['is_detected'],
                dms_detections['cigarette']['conf'],
            )

        emotion = emotion_model.detect(gray)
        emotion_state_manager.update_state(emotion)
        if debug:
            cv2.putText(
                frame,
                "PHONE DETECTED: " + str(is_detected),
                (10, 140),
                cv2.FONT_HERSHEY_TRIPLEX,
                2,
                (0, 100, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "PHONE CONFIDENCE: " + str(conf),
                (10, 170),
                cv2.FONT_HERSHEY_TRIPLEX,
                2,
                (0, 150, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "CLOSE EYE DETECTED: " + str(dms_detections['closed_eye']['is_detected']),
                (10, 200),
                cv2.FONT_HERSHEY_TRIPLEX,
                2,
                (0, 100, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "CLOSE EYE CONFIDENCE: " + str(dms_detections['closed_eye']['conf']),
                (10, 230),
                cv2.FONT_HERSHEY_TRIPLEX,
                2,
                (0, 150, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "CIGARETTE DETECTED: " + str(dms_detections['cigarette']['is_detected']),
                (10, 260),
                cv2.FONT_HERSHEY_TRIPLEX,
                2,
                (0, 100, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "CIGARETTE CONFIDENCE: " + str(dms_detections['cigarette']['conf']),
                (10, 290),
                cv2.FONT_HERSHEY_TRIPLEX,
                2,
                (0, 150, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "EMOTION: " + emotion,
                (1500, 120),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (110, 150, 255),
                2,
                cv2.LINE_AA,
            )

        show_verdicts(
            frame,
            phone_state_manager.verdict(),
            eyes_state_manager.verdict(),
            yawn_state_manager.verdict(),
            cigarette_state_manager.verdict(),
            emotion_state_manager.verdict(),
        )

        e2 = cv2.getTickCount()
        proc_time_frame_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000

        if show_fps:
            cv2.putText(
                frame,
                "FPS:" + str(round(fps)),
                (10, 400),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 0, 255),
                1,
            )
        if show_proc_time:
            cv2.putText(
                frame,
                "PROC. TIME FRAME:" + str(round(proc_time_frame_ms, 0)) + 'ms',
                (10, 430),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 0, 255),
                1,
            )

        cv2.imshow("Press 'q' to terminate", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        i += 1

    cap.release()
    cv2.destroyAllWindows()

    return


def show_verdicts(
        frame,
        handphone: bool,
        asleep: bool,
        yawn: bool,
        cigarette: bool,
        emotion_verdict: tuple[bool, str],
):
    font = cv2.FONT_HERSHEY_TRIPLEX
    x = 1500
    color = {
        True: (0, 0, 255),
        False: (255, 255, 255),
    }

    cv2.putText(
        frame,
        "Cellphone: " + ('true' if handphone else 'false'),
        (x, 40),
        font,
        1.2,
        color[handphone],
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Drowsy: " + ('true' if asleep else 'false'),
        (x, 80),
        font,
        1.2,
        color[asleep],
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Yawning: " + ('true' if yawn else 'false'),
        (x, 120),
        font,
        1.2,
        color[yawn],
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Smoking: " + ('true' if cigarette else 'false'),
        (x, 160),
        font,
        1.2,
        color[cigarette],
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Emotion: " + emotion_verdict[1],
        (x, 200),
        font,
        1.2,
        color[emotion_verdict[0]],
        2,
        cv2.LINE_AA,
    )


def get_face_landmarks(lms):
    surface = 0
    biggest_face = None
    for lms0 in lms:
        landmarks = [np.array([point.x, point.y, point.z]) for point in lms0.landmark]

        landmarks = np.array(landmarks)

        landmarks[landmarks[:, 0] < 0.0, 0] = 0.0
        landmarks[landmarks[:, 0] > 1.0, 0] = 1.0
        landmarks[landmarks[:, 1] < 0.0, 1] = 0.0
        landmarks[landmarks[:, 1] > 1.0, 1] = 1.0

        dx = landmarks[:, 0].max() - landmarks[:, 0].min()
        dy = landmarks[:, 1].max() - landmarks[:, 1].min()
        new_surface = dx * dy
        if new_surface > surface:
            biggest_face = landmarks

    return biggest_face


if __name__ == "__main__":
    main()
