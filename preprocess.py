import cv2
import numpy as np
from collections import deque


def preprocess_frame(frame):
    if frame.shape[:2] != (84, 84):
        frame = cv2.resize(frame, (84, 84))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame / 255.0


class FrameStack:
    def __init__(self, stack_size=8):

        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

    def reset(self, frame):

        processed = preprocess_frame(frame)

        for _ in range(self.stack_size):
            self.frames.append(processed)

        return np.stack(self.frames, axis=0)

    def step(self, frame):
        processed = preprocess_frame(frame)

        self.frames.append(processed)

        return np.stack(self.frames, axis=0)