from .CV2_CAM import CV2_CAM
import cv2, torch, pandas, numpy as np, ncnn

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class Detector(CV2_CAM):
    def __init__(self, frame, tools):
        super().__init__(frame)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_bgr)
            
        self.hand_result = hand_detector.detect(mp_image).hand_landmarks
        self.tools_result = self.tools_inference()

    def get_result(self):
        return self.tools_result, self.hand_result

    def tools_inference(self):
        img_lbx, r, lpad, tpad = self.letterbox(IMG_SIZE)
        img_rgb = cv2.cvtColor(img_lbx, cv2.COLOR_BGR2RGB)
        
        input_mat = ncnn.Mat.from_pixels(img_rgb, ncnn.Mat.PixelType.PIXEL_RGB, IMG_SIZE, IMG_SIZE)
        input_mat.substract_mean_normalize([0,0,0], [1/255.0, 1/255.0, 1/255.0])
    
        # 추론 시작
        ex = net.create_extractor()
        ex.input("in0", input_mat)
        _, result = ex.extract("out0")
    
        arr = result.numpy()
        D, N = arr.shape
        A = arr.T  # (N, D) : 한 행이 한 후보
        cx, cy, w, h = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
    
        cls_scores = A[:, 4:7]              # (N,3)
        cls_ids = np.argmax(cls_scores, axis=1)
        scores  = cls_scores[np.arange(N), cls_ids]
        
        # 인식률로 가져오기
        keep = scores >= CONF_TH
        if not np.any(keep): return []
        cx = cx[keep]; cy = cy[keep]; w = w[keep]; h = h[keep]
        scores = scores[keep]; cls_ids = cls_ids[keep]
    
        # cxcywh -> xyxy (레터박스된 IMG_SIZE 기준)
        x1 = cx - w/2; y1 = cy - h/2
        x2 = cx + w/2; y2 = cy + h/2
    
        # 원본 좌표로 역변환
        x1 = (x1 - lpad) / r; y1 = (y1 - tpad) / r
        x2 = (x2 - lpad) / r; y2 = (y2 - tpad) / r
    
        # 클리핑
        x1 = np.clip(x1, 0, self.W); y1 = np.clip(y1, 0, self.H)
        x2 = np.clip(x2, 0, self.W); y2 = np.clip(y2, 0, self.H)
    
        dets = [(
            [int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])], 
            float(scores[i]), int(cls_ids[i])
            ) 
            for i in range(len(scores))
        ]
        dets = self.nms(dets, IOU_TH)
        
        return dets

    # 전처리
    def letterbox(self, new, color=(114,114,114)):
        r = min(new / self.H, new / self.W)
        nh, nw = int(round(self.H * r)), int(round(self.W * r))
        resized = cv2.resize(self.annotated_image, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.full((new, new, 3), color, dtype=np.uint8)
        top = (new - nh) // 2
        left = (new - nw) // 2
        canvas[top:top+nh, left:left+nw] = resized
        return canvas, r, left, top

    def nms(self, dets, iou_th):
        # dets: [([x1,y1,x2,y2], score, cls_id), ...] in letterbox 좌표
        if not dets: return []
        boxes = np.array([d[0] for d in dets], dtype=np.float32)
        scores = np.array([d[1] for d in dets], dtype=np.float32)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1: break
            rest = order[1:]
            xx1 = np.maximum(boxes[i,0], boxes[rest,0])
            yy1 = np.maximum(boxes[i,1], boxes[rest,1])
            xx2 = np.minimum(boxes[i,2], boxes[rest,2])
            yy2 = np.minimum(boxes[i,3], boxes[rest,3])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            area_i = (boxes[i,2] - boxes[i,0]) * (boxes[i,3] - boxes[i,1])
            area_r = (boxes[rest,2] - boxes[rest,0]) * (boxes[rest,3] - boxes[rest,1])
            iou = inter / (area_i + area_r - inter + 1e-6)
            order = rest[iou <= iou_th]
        return [dets[k] for k in keep]