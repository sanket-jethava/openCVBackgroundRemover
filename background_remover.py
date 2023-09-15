import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

# Constants
IMAGE_FOLDER = 'images'
VIDEO_OUTPUT_FILE = 'backgroundRemover.avi'

# Initialize the camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

# Load background images
imglist = [cv2.resize(cv2.imread(os.path.join(IMAGE_FOLDER, imgpath)), (640, 480)) for imgpath in os.listdir(IMAGE_FOLDER)]

# Initialize SelfiSegmentation and FPS reader
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(VIDEO_OUTPUT_FILE, fourcc, 20, (640, 480))

ImageIndex = 0

while True:
    success, img = cap.read()
    
    # Remove the background
    imgOut = segmentor.removeBG(img, imglist[ImageIndex], threshold=0.80)

    # Display FPS and images
    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
    fps, imgStacked = fpsReader.update(imgStacked, color=(0, 0, 255))

    cv2.imshow("Background Remover", imgStacked)
    out.write(imgOut)

    key = cv2.waitKey(1)

    # Handle key presses
    if key == ord('n') and ImageIndex > 0:
        ImageIndex -= 1
    elif key == ord('p') and ImageIndex < len(imglist) - 1:
        ImageIndex += 1
    elif key == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()