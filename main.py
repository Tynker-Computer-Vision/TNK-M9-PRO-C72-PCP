import cvzone
import cv2
import os
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceMeshModule import FaceMeshDetector

try:
    # Capture the camera feed and set the resolution
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    # Create list to hold face images
    faceImages = []
    path = "Images/"
    pathList = os.listdir(path)
    pathList.sort()

    # Load all the images in the list
    for x, pathImg in enumerate(pathList):
        img = (cv2.imread(path+"/"+pathImg, cv2.IMREAD_UNCHANGED))
        img = cv2.resize(img, (100, 100))
        faceImages.append(img)

    # Creating object to detect hand and face
    detector = HandDetector(detectionCon=0.8)
    faceDetector = FaceMeshDetector(maxFaces=2)

except Exception as e:
    print(e)


# Function to place objects on the face
def showObjectOnface(backImg, frontImg, xLoc, yLoc, dist, rf, rx, ry):
    resizefactor = dist/rf
    frontImg = cv2.resize(frontImg, (0, 0), fx=resizefactor, fy=resizefactor)
    backImg = cvzone.overlayPNG(backImg, frontImg, [int(
        xLoc - (resizefactor*rx)), int(yLoc - (resizefactor * ry))])
    return backImg


# Loop to display video
while True:
    try:
        # Get a single capture from the camera
        success, cameraFeedImg = cap.read()

        cameraFeedImg = cv2.resize(cameraFeedImg, (640, 480))
        cameraFeedImg = cv2.flip(cameraFeedImg, 1)

        # Get width and height of final output screen
        wHeight, wWidth, wChannel = cameraFeedImg.shape

        # Detecting face in the cameraFeedImg
        faces = False

        cameraFeedImg, faces = faceDetector.findFaceMesh(
            cameraFeedImg, draw=True)

        for face in faces:
            xLoc = face[21][0]
            yLoc = face[21][1]

            dist = math.dist(face[21], face[251])
            scale = 55
            dx = 25
            dy = 50

            # Distance between 13, 14 is mouth open distance
            # Distance between 76, 106 lip ends
            lipEndDistance = math.dist(face[76], face[306])
            lipOpenDistance = math.dist(face[13], face[14])

            faceImage = faceImages[0]
            print(dist/lipEndDistance)
            for i in range(0, 450):

                if i == 13 or i == 14 or i == 43 or i == 306 or i == 76:
                    cameraFeedImg = cv2.putText(cameraFeedImg, str(i), (face[i][0], face[i][1]), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.3, (255, 0, 0), 1, cv2.LINE_AA)

            if lipOpenDistance < 10:
                faceImage = faceImages[0]
            else:
                faceImage = faceImages[1]

            if (dist/lipEndDistance < 2.5):
                faceImage = faceImages[2]

            cameraFeedImg = showObjectOnface(
                cameraFeedImg, faceImage, xLoc, yLoc, dist, scale, dx, dy)

    except Exception as e:
        print("Exception", e)

    # Show final image
    cv2.imshow("Image", cameraFeedImg)
    cv2.waitKey(1)
