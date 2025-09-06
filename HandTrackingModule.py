import cv2 as cv
import mediapipe as mp
import math

class HandDetector():
    def __init__(self, static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity 
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands=self.mp_hands.Hands(static_image_mode=self.static_image_mode,
                                       max_num_hands=self.max_num_hands,
                                       model_complexity=self.model_complexity,
                                       min_detection_confidence=self.min_detection_confidence,
                                       min_tracking_confidence=self.min_tracking_confidence)
        
        self.mp_draw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []
 

    def findHands(self, frame, draw=True):
            frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self.results = self.hands.process(frameRGB)
            allHands = []
            h, w, _ = frame.shape

            if self.results.multi_hand_landmarks:
                for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                    myHand = {}
                    mylmList = []
                    xList = []
                    yList = []
                    for id, lm in enumerate(handLms.landmark):
                        px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                        mylmList.append([px, py, pz])
                        xList.append(px)
                        yList.append(py)

                    xmin, xmax = min(xList), max(xList)
                    ymin, ymax = min(yList), max(yList)
                    boxW, boxH = xmax - xmin, ymax - ymin
                    bbox = xmin, ymin, xmax, ymax

                    cx, cy = xmin + (boxW // 2), ymin + (boxH // 2)

                    myHand["lmList"] = mylmList
                    myHand["bbox"] = bbox
                    myHand["center"] = (cx, cy)
                    myHand["type"] = handType.classification[0].label
                    allHands.append(myHand)

                    if draw:
                        self.mp_draw.draw_landmarks(frame, handLms, self.mp_hands.HAND_CONNECTIONS)
                        cv.rectangle(frame, (xmin - OFFSET, ymin - OFFSET), (xmax + OFFSET, ymax + OFFSET), (255, 0, 255), 2)
                        cv.putText(frame, myHand["type"], (xmin - 30, ymin - 30), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        
            return allHands, frame
    
    
    def fingersUp(self, myHand):
        fingers = []
        myHandType = myHand["type"]
        myLmList = myHand["lmList"]
        if self.results.multi_hand_landmarks:

            if myHandType == "Right":
                if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            for id in range(1, 5):
                if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        return fingers


    def getPixelDistance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)

        return length, info
    
    def getActualDistance(self, p1, p2, ref_pixel_dist, ref_actutal_dist):
        x1, y1 = p1
        x2, y2 = p2
        pixel_dist = math.hypot(x2 - x1, y2 - y1)
        estimated_dist = (pixel_dist / ref_pixel_dist) * ref_actutal_dist if ref_actutal_dist > 0 else 0
        
        return estimated_dist

# Constants
OFFSET = 20        

def main():

    cap = cv.VideoCapture(0)

    detector = HandDetector()
    if not cap.isOpened():
        print("Error: Could not open the camera.")
    else:
        while True:
            success, frame = cap.read()
            if not success:
                break
            hands, frame = detector.findHands(frame, draw=True)

            if hands:
                for hand in hands:
                    lmList = hand["lmList"]
                    if len(lmList) > 8:
                        detector.getPixelDistance(lmList[4][:2], lmList[8][:2])

            cv.imshow("Live Cam", frame)

            if cv.waitKey(1) == 27:
                break

    cap.release()
    cv.destroyAllWindows()
    

if __name__ == "__main__":
    main()