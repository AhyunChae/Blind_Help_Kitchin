from .CV2_CAM import CV2_CAM
import sounddevice as sd, scipy.io.wavfile

class CheckBox(CV2_CAM):
    def __init__(self, frame, tools_result, hand_result):
        super().__init__(frame)
        self.flag = self.flag_init()
        self.hand_result = hand_result

        blade_xy = [ tools[0] for tools in tools_result if tools[2] == 0 ]
        handle_xy = [ tools[0] for tools in tools_result if tools[2] == 1 ]

        global HANDLE_PIXEL, BLADE_PIXEL
        if len(handle_xy) > 0:
            HANDLE_PIXEL = self.save_pixel(handle_xy)
        
        if len(blade_xy) > 0:
            BLADE_PIXEL = self.save_pixel(blade_xy)

    def flag_init(self):
        obj = {
            "handle": False,
            "blade": False
        }

        return obj

    def save_pixel(self, boxes):
        lr = []
        for x1,y1,x2,y2 in boxes:
            lr.append([x1,y1,x2,y2])
        return lr
    # ===========================================================
    def detect_box(self):
        for hand in self.hand_result:
            self.check_handle(hand)

            for lm in hand:
                self.check_blade(lm)

        self.check_flag()
        return self.annotated_image
    
    def check_handle(self, hand):
        middle_x = int(hand[9].x * self.W)
        middle_y = int(hand[9].y * self.H)
        cv2.circle(self.annotated_image, (middle_x, middle_y), 5, COLOR_BLUE, -1, cv2.LINE_AA)
        
        for handle in HANDLE_PIXEL:
            handle_middle_x = (handle[0] + handle[2]) // 2
            handle_middle_y = (handle[1] + handle[3]) // 2 
            cv2.circle(self.annotated_image, (handle_middle_x, handle_middle_y), 5, (255,255,0), -1, cv2.LINE_AA)

            if self.check_inside(handle, middle_x, middle_y):
                self.flag["handle"] = True
                break
                
            if middle_x < handle_middle_x:
                if middle_y < handle_middle_y:
                    self.draw_text("Right-Up", DERECTION_ORG, COLOR_BLUE)
                else:
                    self.draw_text("Right-Down", DERECTION_ORG, COLOR_BLUE)
                    
            elif middle_x > handle_middle_x:
                if middle_y < handle_middle_y:
                    self.draw_text("Left-Up", DERECTION_ORG, COLOR_BLUE)
                else:
                    self.draw_text("Left-Down", DERECTION_ORG, COLOR_BLUE)
                    
    def check_blade(self, lm):
        lm_x = int(lm.x * self.W)
        lm_y = int(lm.y * self.H)
        
        # 날 확인
        for blade in BLADE_PIXEL:
            if self.check_inside(blade, lm_x, lm_y):
                self.flag["blade"] = True
                break
                
    def check_flag(self):
        if self.flag["blade"]:
            if self.flag["handle"]:
                self.draw_text("DETECTED", WARNING_TXT_ORG, COLOR_GREEN)
            else:
                self.draw_text("DANGER", WARNING_TXT_ORG, COLOR_RED)
        elif self.flag["handle"]:
            self.draw_text("DETECTED", WARNING_TXT_ORG, COLOR_GREEN)
    # ===========================================================
    def check_inside(self, box, x, y):
        if box[0] <= x <= box[2] and box[1] <= y <= box[3]: 
            return True
        else: return False
            
    def draw_text(self, text, org, color):
        cv2.putText(self.annotated_image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)