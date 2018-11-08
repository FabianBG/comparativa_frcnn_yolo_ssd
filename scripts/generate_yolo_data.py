import sys
import os
from shutil import copy2
import numpy as np
from utilities import save_file


# Usage
# generate_yolo_data.py dataset_path yolo_path train_percentaje
# dataset_path: Path de a carpeta contenedora de las imagenes
# yolo_path: Folder for copy images 
# test_percentaje: porcentaje por carpeta de imagenes para pruebas


image_ext = ".jpg"

folders = []
train = []
validate = []
test = []
dataset_path = sys.argv[1]
yolo_path = sys.argv[2]
train_percentaje = float(sys.argv[3])

for path in os.listdir(dataset_path):
    directory = os.path.join(dataset_path, path)
    if len(directory.split(".")) == 1:
        folders.append(directory)
        images = []
        for filename in os.listdir(directory):
            if filename.endswith(image_ext):
                dest = os.path.join(yolo_path, 'data')
                images.append(os.path.join(dest, filename))
                copy2(os.path.join(directory, filename), dest)
                copy2(os.path.join(directory, filename.split('.')[0] + '.txt'), dest)
        perm = np.random.permutation(len(images))
        images = np.array(images)
        images = images[perm]
        partition = int(len(images) * train_percentaje)
        validation = int(partition * train_percentaje)
        train = train + images[0:validation].tolist()
        validate = validate + images[validation:partition].tolist()
        test = test + images[partition:].tolist()
        print("En {} se usa: \n{} entrenamiento \n{} validacion \n{} pruebas"
        .format(directory, validation, partition - validation, len(images) - validation))
            
	
save_file(os.path.join(yolo_path, 'train.txt'), train)
save_file(os.path.join(yolo_path, 'test.txt'), test)
save_file(os.path.join(yolo_path, 'validation.txt'), test)

print("Se encontro {} directorios".format(len(folders)))
print("Se copia {} imagenes de entrenamiento".format(len(train)))
print("Se copia {} imagenes de validacion".format(len(validate)))
print("Se copia {} imagenes de pruebas".format(len(test)))

