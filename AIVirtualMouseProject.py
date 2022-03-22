import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

##################################

widthOfCam, heightOfCam = 640, 480
widthOfScreen, heightOfScreen = autopy.screen.size()
# print(widthOfScreen, heightOfScreen) # my computer's size are 1536, 864 and it might change according to computer
frameReduction = 100
smoothening = 7

##################################

previousTime = 0
previousLocationOfX, previousLocationOfY = 0, 0
currentLocationOfX, currentLocationOfY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, widthOfCam) # width
cap.set(4, heightOfCam) # heigth
detector = htm.handDetector(maxHands=1)

while True:
    # 1. Find Hand Landmarks
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    landMarkLists, bBox = detector.findPosition(img)
    cv2.rectangle(img, (frameReduction, frameReduction), (widthOfCam - frameReduction, heightOfCam - frameReduction),
                  (255, 0, 255), 2)

    # 2. Get the tip of the index finger and middle finger
    if len(landMarkLists)!=0:
        x1, y1 = landMarkLists[8][1:]
        x2, y2 = landMarkLists[12][1:]
        # print(x1, y1)
        # print(x2, y2)

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4. Only Index Finger : Moving Mode
        if fingers[1]==1 and fingers[2]==0: # index finger is up and middle finger is down
            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameReduction, widthOfCam - frameReduction), (0, widthOfScreen))
            y3 = np.interp(y1, (frameReduction, heightOfCam - frameReduction), (0, heightOfScreen))
            # 6. Smoothen Values
            currentLocationOfX = previousLocationOfX + (x3 - previousLocationOfX) / smoothening
            currentLocationOfY = previousLocationOfY + (y3 - previousLocationOfY) / smoothening

            # 7. Move Mouse
            autopy.mouse.move(currentLocationOfX, currentLocationOfY)
            cv2.circle(img, (x1, y1), 15, (0, 255, 255), cv2.FILLED) # when we are at the moving mode, there will be a bigger dot.
            previousLocationOfX, previousLocationOfY = currentLocationOfX, currentLocationOfY

        # 8. Both Index and Middle Fingers Up : Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:  # index finger is up and middle finger is up as well
            # 9. Find Distance Between Fingers
            lengthBetweenIndexFingerAndMiddleFinger, img, lineInfo = detector.findDistance(8, 12, img)
            # print(lengthBetweenIndexFingerAndMiddleFinger)
            # 10. Click Mouse If Distance Is Short
            if lengthBetweenIndexFingerAndMiddleFinger < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED) # when we are at the moving mode, there will be a bigger dot.
                autopy.mouse.click()

    # 11. Frame Rate
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    # 12. Display
    cv2.putText(img, ('FPS: ' + str(int(fps))), (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

