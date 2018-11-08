import sys
import os
import numpy as np
from utilities import save_file, generate_datasets


# Usage
# generate_yolo_data.py dataset_path yolo_path train_percentaje
# dataset_path: Path de a carpeta contenedora de las imagenes
# yolo_path: Folder for copy images 
# test_percentaje: porcentaje por carpeta de imagenes para pruebas


image_ext = ".jpg"

dataset_path = sys.argv[1]
yolo_path = sys.argv[2]
train_percentaje = float(sys.argv[3])

train, validate, test, folders = generate_datasets(dataset_path, train_percentaje, yolo_path=yolo_path)
           
save_file(os.path.join(yolo_path, 'train.txt'), train)
save_file(os.path.join(yolo_path, 'test.txt'), test)
save_file(os.path.join(yolo_path, 'validation.txt'), test)

print("Se encontro {} directorios".format(len(folders)))
print("Se copia {} imagenes de entrenamiento".format(len(train)))
print("Se copia {} imagenes de validacion".format(len(validate)))
print("Se copia {} imagenes de pruebas".format(len(test)))

