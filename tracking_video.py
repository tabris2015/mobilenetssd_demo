from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np 
import argparse
import imutils
import cv2 
import datetime

## argparse para los argumentos
ap = argparse.ArgumentParser()
ap.add_argument('-v','--video', required=True, help='path to video')
ap.add_argument("-t", "--tracker", type=str, default="kcf", help="OpenCV object tracker type")
args = vars(ap.parse_args())

## definicion del descriptor para la deteccion de personas
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

person_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')



## objetos para el tracker
OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create
	}
 
	# grab the appropriate object tracker using our dictionary of
	# OpenCV object tracker objects
tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
 


## objeto para la captura de frames en el video
cap = cv2.VideoCapture(args['video'])

frame = 0

boxes = []
track_success = False

REDUCE_X = 4
REDUCE_Y = 7
while cap.isOpened():
    start = datetime.datetime.now()
    # leer un frame del video    
    ret, image = cap.read()
    image = imutils.resize(image, width=min(800, image.shape[1]))
    # orig = image.copy()
    # gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #detect if we have lost the object
    if not track_success:
        # detect
        boxes = person_cascade.detectMultiScale(image)
        # non maxima supression
        rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in boxes])
        selected = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        if len(selected) > 0:
            # continue
            pick = selected[0]
            initBB = (pick[0] + REDUCE_X , pick[1] + REDUCE_Y ,pick[2] - pick[0] - REDUCE_X, pick[3] - pick[1] - REDUCE_Y)
            # si hay detecciones se procesa
            if len(pick) != 0:
                tracker.init(image, initBB)
            #draw final
            # for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (initBB[0], initBB[1]), (initBB[0]+initBB[2], initBB[1]+initBB[3]), (0, 255, 0), 2)
            track_success = True
    # si seguimos trackeando el objeto
    else:
        # print('success')
        # grab the new bounding box coordinates of the object
        (track_success, box) = tracker.update(image)
        # check to see if the tracking was a track_success
        if track_success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(image, (x, y), (x + w, y + h),(0, 255, 255), 3)
            

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
