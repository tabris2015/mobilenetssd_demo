from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np 
import argparse
import imutils
import cv2 
import datetime

ap = argparse.ArgumentParser()
ap.add_argument('-v','--video', required=True, help='path to video')
args = vars(ap.parse_args())

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

person_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')


cap = cv2.VideoCapture(args['video'])


frame = 0

while cap.isOpened():
    start = datetime.datetime.now()
    ret, image = cap.read()
    image = imutils.resize(image, width=min(1200, image.shape[1]))
    orig = image.copy()
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #detect 
    boxes = person_cascade.detectMultiScale(gray_frame)
    #draw
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)

    # non maxima supression
    rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in boxes])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    #draw final
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    frame += 1

    cv2.putText(image, 'frame: {}'.format(frame), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(image, 'fps: {}'.format(int(1/(datetime.datetime.now() - start).total_seconds())), (10,65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    # filename = imagePath[imagePath.rfind("/") + 1:]
    # print("[INFO] {}: {} original boxes, {} after suppression".format(filename, len(rects), len(pick)))
    # show the output images     
    # cv2.imshow("Before NMS", orig) 	
    cv2.imshow("After NMS", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()    
