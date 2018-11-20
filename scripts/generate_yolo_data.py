import sys
import os
import numpy as np
from utilities import save_file, generate_datasets, generate_augmentations


# Usage
# generate_yolo_data.py dataset_path yolo_path train_percentaje val_percentaje aug
# dataset_path: Path de a carpeta contenedora de las imagenes
# yolo_path: Folder for copy images 
# train_percentaje: porcentaje por carpeta de imagenes para pruebas
# val_percentaje: porcentaje por carpeta de imagenes para pruebas
# aug: numero de uamentación por iamgen


image_ext = ".jpg"
augmentation = 0

if len(sys.argv) >= 5:
    augmentation = int(sys.argv[5])

dataset_path = sys.argv[1]
yolo_path = sys.argv[2]
train_percentaje = float(sys.argv[3])
val_percentaje = float(sys.argv[4])

maximun_aug = -1 #-1 para aumentar completo


train, validate, test, folders = generate_datasets(dataset_path,
train_percentaje, val_percentaje, yolo_path=yolo_path)


if augmentation != 0:
    print("Iniciando aumentacion de datos train")
    train = train + generate_augmentations(train, os.path.join(yolo_path, 'data'), 
        augmentation, maximun_aug)
    print("Iniciando aumentacion de datos validate")
    validate = validate + generate_augmentations(validate, os.path.join(yolo_path, 'data'), 
        augmentation, maximun_aug)
    print("Fin aumentacion de datos")
           
save_file(os.path.join(yolo_path, 'train.txt'), train)
save_file(os.path.join(yolo_path, 'test.txt'), test)
save_file(os.path.join(yolo_path, 'validation.txt'), validate)

print("Se encontro {} directorios".format(len(folders)))
print("Se copia {} imagenes de entrenamiento".format(len(train)))
print("Se copia {} imagenes de validacion".format(len(validate)))
print("Se copia {} imagenes de pruebas".format(len(test)))

