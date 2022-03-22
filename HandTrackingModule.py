import math

import cv2
import mediapipe as mp
import time

class handDetector():
    # the parameters of the constructer function needs to be Hands()'s parameters which are;
    # (self,
    #          static_image_mode=False,
    #          max_num_hands=2,
    #          model_complexity=1,
    #          min_detection_confidence=0.5,
    #          min_tracking_confidence=0.5):
    def __init__(self, mode = False, maxHands = 2, detectionConfidence = 0.5, trackingConfidence = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        # self.mode, self.maxHands, self.detectionConfidence, self.trackingConfidence
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        # drawing points and lines according to number of hands in the camera
        if self.results.multi_hand_landmarks:
            for handLandMark in self.results.multi_hand_landmarks:
                # drawing the lines for hands
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandMark, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo = 0, draw = True):
        pixelOfXList = []
        pixelOfYList = []
        bbox = []
        self.landMarkList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, landMark in enumerate(myHand.landmark):
                # print(id, landMark) # locations of landMarks

                # finding pixel equivelents of x and y locations
                height, width, channels = img.shape
                pixelOfX, pixelOfY = int(landMark.x * width), int(landMark.y * height)
                pixelOfXList.append(pixelOfX)
                pixelOfYList.append(pixelOfY)

                # printing coordinates of all cells as pixel
                # print(id, pixelOfX, pixelOfY)
                self.landMarkList.append([id, pixelOfX, pixelOfY])
                # drawing circle for the top of the index finger
                if draw:
                    cv2.circle(img, (pixelOfX, pixelOfY), 5, (0, 255, 255), cv2.FILLED)
                                                        #    (B, G  , R  )
            xmin, xmax = min(pixelOfXList), max(pixelOfXList)
            ymin, ymax = min(pixelOfYList), max(pixelOfYList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.landMarkList, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.landMarkList[self.tipIds[0]][1] < self.landMarkList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if self.landMarkList[self.tipIds[id]][2] < self.landMarkList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw = True, r = 15, t = 3):
        x1, y1 = self.landMarkList[p1][1:]
        x2, y2 = self.landMarkList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (255, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    previousTime = 0
    currentTime = 0

    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        landMarkList = detector.findPosition(img)

        if len(landMarkList) != 0:
            # print(landMarkList[4]) # thumb tip
            # print(landMarkList[8]) # index finger tip
            fingers = detector.fingersUp()
            print(fingers)

        # writing fps to console
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        # print(currentTime-previousTime)
        previousTime = currentTime
        cv2.putText(img, ('FPS: ' + str(int(fps))), (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255),
                    2)  # the console where text is going to be put, text, location, font, scale, color, thickness

        # displaying the camera view
        cv2.imshow("Image", img)
        cv2.waitKey(1)




if __name__ == "__main__":
    main()