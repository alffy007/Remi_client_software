import numpy as np
import cv2
from keras.models import model_from_json, Sequential
import time
from keras.preprocessing.image import img_to_array

class EmotionDetector:
    def __init__(self):
        # Load the model
        json_file = open("Remi_eyes/facialemotionmodel.json", "r")
        model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(model_json, custom_objects={'Sequential': Sequential})
        self.model.load_weights("Remi_eyes/facialemotionmodel.h5")

        # Load Haar cascade
        haar_file = 'Remi_eyes/haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_file)

        # Video capture
        self.picam2 = cv2.VideoCapture(0)
        time.sleep(2)

        # Emotion labels
        self.labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

    def extract_features(self, image):
        feature = np.array(image)
        feature = feature.reshape(1, 48, 48, 1)
        return feature / 255.0

    def detect_emotions(self):
       try:
        while True:
            ret, array = self.picam2.read()
            if not ret:
                yield "Failed to capture image"
                break
            
            gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)

            for (x, y, w, h) in faces:
                face_image = gray[y:y+h, x:x+w]
                face_image = cv2.resize(face_image, (48, 48))
                img = self.extract_features(face_image)
                pred = self.model.predict(img)
                prediction_label = self.labels[pred.argmax()]
                yield prediction_label  # Streaming the detected emotion

            if cv2.waitKey(27) & 0xFF == ord('q'):
                break
        self.picam2.release()
        cv2.destroyAllWindows() 
       except KeyboardInterrupt:
       		exit()
            
              
