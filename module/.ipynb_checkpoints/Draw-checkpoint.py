from .CV2_CAM import CV2_CAM

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

class Draw(CV2_CAM):
    def __init__(self, frame, tools_result, hand_result):
        super().__init__(frame)
        self.tools_result = tools_result
        self.hand_result = hand_result

    def draw_on_cv(self):
        # 도구 박스 그리기
        for (x1,y1,x2,y2), _, _ in self.tools_result:
            cv2.rectangle(self.annotated_image, (x1,y1), (x2,y2), (0,0,255), 2)

        # 손 랜드마크 그리기
        for landmark in self.hand_result:
            landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            for lm in landmark: 
                landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)])
                
            solutions.drawing_utils.draw_landmarks(
                self.annotated_image,
                landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style()
            )
    
        return self.annotated_image

    def draw_circle(self, center, color):
        cv2.circle(self.annotated_image, center, 5, color, -1, cv2.LINE_AA)