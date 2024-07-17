import operator


class StateManager:
    def __init__(self, n, alert_count):
        self.state = [0] * n
        self.alert_count = alert_count

    def verdict(self) -> bool:
        return sum(self.state) >= self.alert_count


class PhoneStateManager(StateManager):
    def __init__(self, n, alert_count, angle_threshold, conf_threshold):
        super().__init__(n, alert_count)

        self.angle_threshold = angle_threshold
        self.conf_threshold = conf_threshold

    def update_state(self, angle, is_detected, conf):
        self.state = self.state[1:]

        if angle < self.angle_threshold and is_detected and conf > self.conf_threshold:
            self.state.append(1)
        else:
            self.state.append(0)


class EyesStateManager(StateManager):
    def __init__(self, n, alert_count, ear_threshold, conf_threshold):
        super().__init__(n, alert_count)

        self.ear_threshold = ear_threshold
        self.conf_threshold = conf_threshold

    def update_state(self, ear, is_detected, conf):
        self.state = self.state[1:]

        if ear < self.ear_threshold and is_detected and conf > self.conf_threshold:
            self.state.append(1)
        else:
            self.state.append(0)


class YawnStateManager(StateManager):
    def __init__(self, n, alert_count, mar_threshold):
        super().__init__(n, alert_count)

        self.mar_threshold = mar_threshold

    def update_state(self, mar):
        self.state = self.state[1:]

        if mar > self.mar_threshold:
            self.state.append(1)
        else:
            self.state.append(0)


class CigaretteStateManager(StateManager):
    def __init__(self, n, alert_count, conf_threshold):
        super().__init__(n, alert_count)

        self.conf_threshold = conf_threshold

    def update_state(self, is_detected, conf):
        self.state = self.state[1:]

        if is_detected and conf > self.conf_threshold:
            self.state.append(1)
        else:
            self.state.append(0)


class EmotionStateManager(StateManager):
    def __init__(self, n, alert_count):
        super().__init__(n, alert_count)

        self.dominance_emotion = "Neutral"
        self.state = ["Neutral"] * n

        self.emotion_counter = {
            'Neutral': n,
            'Happy': 0,
            'Sad': 0,
            'Surprise': 0,
            'Fear': 0,
            'Disgust': 0,
            'Angry': 0,
        }
        self.bad_emotion = ['Sad', 'Fear', 'Disgust', 'Angry']

    def verdict(self) -> tuple[bool, str]:
        if self.dominance_emotion not in self.bad_emotion:
            return False, self.dominance_emotion
        return self.emotion_counter[self.dominance_emotion] > self.alert_count, self.dominance_emotion

    def update_state(self, emotion):
        past_emotion = self.state.pop(0)
        self.emotion_counter[past_emotion] -= 1

        self.state.append(emotion)
        self.emotion_counter[emotion] += 1

        self.dominance_emotion = max(self.emotion_counter.items(), key=operator.itemgetter(1))[0]
