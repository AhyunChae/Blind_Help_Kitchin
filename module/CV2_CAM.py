import numpy as np

class CV2_CAM:
    def __init__(self, frame):
        self.annotated_image = np.copy(frame)
        self.H, self.W, self.C = frame.shape