# import the necessary packages
from utilidades.centroidtracker import CentroidTracker
from utilidades.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
	help="# of skip frames between detections")
args = vars(ap.parse_args())

# clases de la red neuronal
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# cargar modelo
print("[INFO] Cargando modelo...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


if not args.get("input", False):
	print("[INFO] iniciando camara...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

else:
	print("[INFO] abriendo archivo...")
	vs = cv2.VideoCapture(args["input"])

# dimensiones del frame
W = None
H = None

# iniciar tracker
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# variables para contar personas
totalFrames = 0
totalDown = 0
totalUp = 0

# estimador de fps
fps = FPS().start()

# ancho de la imagen
IM_WIDTH = 500
# para cada frame
while True:
	# lectura del frame
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	# fin del video
	if args["input"] is not None and frame is None:
		break

	
	frame = imutils.resize(frame, width=IM_WIDTH)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# guardar las dimensiones
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# estado del programa
	status = "Waiting"
	rects = []

	# verificar si debemos correr el detector
	if totalFrames % args["skip_frames"] == 0:
		# inicializacion del detector
		status = "Detecting"
		trackers = []

		# convertir la imagen en un blob para pasar a la red neuronal
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		net.setInput(blob)
		detections = net.forward()

		# para cada deteccion
		for i in np.arange(0, detections.shape[2]):
			# extraer la confianza
			confidence = detections[0, 0, i, 2]

			# filtrar detecciones debiles
			if confidence > args["confidence"]:
				# extraer indice de la clase
				idx = int(detections[0, 0, i, 1])

				# corresponder con la deteccion de personas
				if CLASSES[idx] != "person":
					continue

				# calcular la region de la deteccion
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")

				# contruir una region para el trackeo con dlib
				tracker = dlib.correlation_tracker()
				rect = dlib.rectangle(startX, startY, endX, endY)
				tracker.start_track(rgb, rect)

				# agregar los trackers
				trackers.append(tracker)

	# seguir con el tracking
	else:
		# para cada tracker
		for tracker in trackers:
			# estado del programa
			status = "Tracking"

			# actualizar los trackers
			tracker.update(rgb)
			pos = tracker.get_position()

			# recuperar la posicion
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# agregar las regiones
			rects.append((startX, startY, endX, endY))

	# usar el tracker de centroide
	objects = ct.update(rects)

	# para cada objeto
	for (objectID, centroid) in objects.items():
		# si existe el objeto en la lista
		to = trackableObjects.get(objectID, None)

		# si no, crear una nueva entrada
		if to is None:
			to = TrackableObject(objectID, centroid)

		else:
			# la diferencia en el eje y nos dice la direccion
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)

			# verificar si ya ha sido contado
			if not to.counted:
				# si la direccion es negativa
				if direction < 0 and centroid[1] < H // 2:
					totalUp += 1
					to.counted = True

				# si la direccion es positiva
				elif direction > 0 and centroid[1] > H // 2:
					totalDown += 1
					to.counted = True

		trackableObjects[objectID] = to

		# plotear el id y el centroide
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		
		N_TRACE = 60	
		plot_centroids = to.centroids if len(to.centroids) < N_TRACE else to.centroids[-N_TRACE:]
		# dibujar las ultimas posiciones
		for i in range(0,len(plot_centroids),4):
			cv2.circle(frame, (plot_centroids[i][0], plot_centroids[i][1]), 4, (0, 255, 0), -1)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# increment the total number of frames processed thus far and
	# then update the FPS counter
	totalFrames += 1
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


# if we are not using a video file, stop the camera video stream
if not args.get("input", False):
	vs.stop()

# otherwise, release the video file pointer
else:
	vs.release()

# close any open windows
cv2.destroyAllWindows()