import cv2
import numpy as np
import glob
import mahotas  # Contiene la implementación de texturas de Haralick que necesitamos.
from sklearn.svm import LinearSVC
import joblib # Modelo que entrenaremos para clasificar las texturas.
import numpy as numpy



print('Extrayendo features...')
data = []
labels = []
path='texture/train/*.jpg'
path_test='texture/test/*.jpg'
listaPsroductos=[]
        # Ciclamos por las imágenes del conjunto de entrenamiento...
for image_path in glob.glob(path):
            # Cargamos la imagen y la convertimos a escala de grises.
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
            # Extraemos el nombre de la textura representada de la ruta de la imagen.
            texture = image_path[image_path.rfind('/') + 1:].split('_')[0]
        
            # Calculamos el feature vector usando la implementación de Haralick presente en Mahotas. Fíjate cómo promediamos
            # los cuatro vectores con el método mean().
            features = mahotas.features.haralick(image).mean(axis=0)
        
            # Añadimos los features y la etiqueta (textura) al conjunto de entrenamiento.
            data.append(features)
            labels.append(texture)
        
print('Entrenando el modelo...')
model = LinearSVC(C=10, random_state=42)
model.fit(data, labels)
print('Clasificando...')
joblib.dump(model,'clasTexture.pkl')
print("creado pkl")
modelo=joblib.load('clasTexture.pkl')

for img in glob.glob(path_test):
            # Cargamos la imagen y la convertimos a escala de grises.
            image = cv2.imread(img)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Calculamos el feature vector usando la implementación de Haralick presente en Mahotas. Fíjate cómo promediamos
            # los cuatro vectores con el método mean().
            features = mahotas.features.haralick(gray).mean(axis=0)
        
            # Imprimimos la predicción (textura) en la imagen.
            prediction = modelo.predict(features.reshape(1, -1))[0]
            cv2.putText(image , prediction, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.imshow('imagen',image )
            listaPsroductos.append(prediction)
            cv2.waitKey(0)
cv2.destroyAllWindows()

            