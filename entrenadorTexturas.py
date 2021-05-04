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
data1 = []
labels1 = []
path='texture/train/*.jpg'
path_dos='texture/train2/*.jpg'
path_test='texture/test/*[.jpg|.png|.JPG]'
path_c='Images/*.jpeg'
listaPsroductos=[]
for image_path in glob.glob(path_c):
        image = cv2.imread(image_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        colors, count = np.unique(hsv.reshape(-1, hsv.shape[-1]), axis=0, return_counts=True)
        print(f'imagen \n {colors[np.argsort(-count)][:5]}')
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
model1 = LinearSVC(C=10, random_state=42)
model1.fit(data, labels)
print('Clasificando...')
joblib.dump(model1,'clasTexture.pkl')
print("creado pkl")



for image_path in glob.glob(path_dos):
            # Cargamos la imagen y la convertimos a escala de grises.
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Extraemos el nombre de la textura representada de la ruta de la imagen.
            texture = image_path[image_path.rfind('/') + 1:].split('_')[0]
            # Calculamos el feature vector usando la implementación de Haralick presente en Mahotas. Fíjate cómo promediamos
            # los cuatro vectores con el método mean().
            features = mahotas.features.haralick(image).mean(axis=0)
            # Añadimos los features y la etiqueta (textura) al conjunto de entrenamiento.
            data1.append(features)
            labels1.append(texture)
        
print('Entrenando el modelo...')
model2 = LinearSVC(C=10, random_state=42)
model2.fit(data1, labels1)
print('Clasificando...')
joblib.dump(model2,'clasTexture2.pkl')
print("creado pkl")
modelo1=joblib.load('clasTexture.pkl')
modelo2=joblib.load('clasTexture2.pkl')

for img in glob.glob(path_test):
            # Cargamos la imagen y la convertimos a escala de grises.
            image = cv2.imread(img)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Calculamos el feature vector usando la implementación de Haralick presente en Mahotas. Fíjate cómo promediamos
            # los cuatro vectores con el método mean().
            features = mahotas.features.haralick(gray).mean(axis=0)
           
            # Imprimimos la predicción (textura) en la imagen.
            prediction = modelo1.predict(features.reshape(1, -1))[0]
            prediction2 = modelo2.predict(features.reshape(1, -1))[0]
            cv2.putText(image , prediction2, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            print(prediction2)
            cv2.imshow('imagen',image )
            listaPsroductos.append({prediction:prediction2})
            cv2.waitKey(0)
print(listaPsroductos)
cv2.destroyAllWindows()

            