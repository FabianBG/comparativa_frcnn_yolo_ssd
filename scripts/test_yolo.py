import sys
import os
import cv2 as cv
sys.path.append(os.path.join('/home/mbastidas/git/darknet/python/'))
import darknet as dn
from utilities import read_yolo_images, classes_as_dicts, draw_box


'''
Usage
test_yolo.py archivo_pesos archivo_config archivo_meta path_imagenes

archivo_pesos: pesos producidos del entrenamiento de darknet
archivo_config: archivo e configuracion de darknet
archivo_meta: archivo de metadata de darknet
archivo_names: archivo de clases de darknet
path_imagenes: directorio con imagenes y cajas a validar
'''

weigths = sys.argv[1]
config = sys.argv[2]
meta = sys.argv[3]
names = sys.argv[4]
file_dir = sys.argv[5]

indexes, classes = classes_as_dicts(names)

data = read_yolo_images(file_dir)
net = dn.load_net(config.encode('utf-8'), weigths.encode('utf-8'), 0)
meta = dn.load_meta(meta.encode('utf-8'))



results = {}

positives = 0
total = 0
index = 0
for d in data:
    #[class, prob, (x,y, w,h)]
    result = {
        "image": d['image'],
        "boxes": d['boxes'],
        "predictions": None
    }
    predictions = dn.detect(net, meta, d['image'].encode('utf-8'), thresh=.3)
    print("{} predicions found {}".format(d['image'], len(predictions)))
    if len(predictions) > 0:
        image = cv.imread(d['image']) 
        for box in result['boxes']:
            name = classes[int(box[0])]
            for pred in predictions:
                total = total + 1
                print(pred[0].decode("utf-8"), name)
                if pred[0].decode("utf-8") == name:
                    positives = positives + 1
                draw_box(image, pred[0].decode("utf-8"), pred[2], pred[1], int(box[0]))
        index = index + 1
        cv.imwrite("./predicts/{}_{}.jpg".format(name, index), image)
    else:
        total = total + 1

print("Acertados {} de {}".format(positives, total))
print("Porcentage {}".format(positives / total))

