import sys
import os
import cv2 as cv
import time
import numpy as np
sys.path.append(os.path.join('/home/mbastidas/git/darknet/python/'))
import darknet as dn
from utilities import read_yolo_images, classes_as_dicts, draw_box, calcualte_iou, convert_axis
from utilities import desnormalize, coco_eval, convert_axis_no_shape

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
times = []
ious = []
ann_json = {
    "annotations":[],
    "categories":[],
    "images": []
    }
pred_json = []
true_class = [0] * len(classes)
total_class = [0] * len(classes)
image_id = 0
for d in data:
    #[class, prob, (x,y, w,h)]
    result = {
        "image": d['image'],
        "boxes": d['boxes'],
        "predictions": None
    }
    
    image_id = image_id + 1
    image = cv.imread(d['image']) 
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    ann_json["images"].append({
                    "id" : image_id, 
                    "width" : image.shape[1], 
                    "height" : image.shape[0], 
                    "file_name" : d['image']
                })
    start_time = time.time()
    predictions = dn.detect(net, meta, d['image'].encode('utf-8'), thresh=.5)
    times.append(time.time() - start_time)
    print("{} predicions found {}".format(d['image'], len(predictions)))
    total = total + len(result['boxes'])
    if len(predictions) > 0:
        for box in result['boxes']:
            name = classes[int(box[0])]
            box_id = int(box[0])
            total_class[box_id] = total_class[box_id] + 1
            real_box = np.array(box[1:], dtype=float)
            real_box = desnormalize(real_box, image.shape)
            coco_box = convert_axis(image.shape, np.array(box[1:], dtype=float))
            ann_json["annotations"].append({
                        "id" : image_id,
                        "image_id" : image_id,
                        "category_id" : box_id, 
                        #"segmentation" : RLE or [polygon], 
                        #"area" : float, 
                        "bbox" : [coco_box[0],coco_box[2],
                                            coco_box[1] - coco_box[0],coco_box[3] - coco_box[2]], 
                        "iscrowd" : 0,
                        "area": (coco_box[1] - coco_box[0]) * (coco_box[3] - coco_box[2])
                    })
            for pred in predictions:
                if pred[0].decode("utf-8") == name:
                    positives = positives + 1
                    class_idx = indexes[pred[0].decode("utf-8")]
                    true_class[box_id] = true_class[box_id] + 1
                    iou = calcualte_iou(real_box ,pred[2], is_yolo=True)
                    print("IOU", iou)
                    ious.append(iou)
                    pred_box = convert_axis_no_shape(pred[2])
                    pred_json.append({
                                "image_id" : image_id, 
                                "category_id" : int(class_idx), 
                                "bbox" : [pred_box[0],pred_box[1],
                                            pred_box[2] - pred_box[0],pred_box[3] - pred_box[1]], 
                                "score" : float(pred[1])
                            })
                    draw_box(image, "real", real_box, 100, 
                        class_index=20)
                    draw_box(image, pred[0].decode("utf-8"), pred[2], pred[1], 
                        class_index=class_idx)
                    
                    cv.imwrite("./predicts/{}.jpg".format(image_id), cv.cvtColor(image, cv.COLOR_RGB2BGR)) 
                    
                    break
        

print("Acertados {} de {} predicciones".format(positives, total))
print("Porcentage {}".format(positives / total))
print("Tiempo de inferencia promedio {} segundos".format(np.sum(times)/len(times)))
print("IOU promedio ~ {}".format(np.sum(ious)/len(ious)))
print("IOU promedio ~ {}".format(np.sum(ious)/len(ious)))
ap = []
for key in classes:
    i = key
    v = classes[key]
    ann_json['categories'].append({
        "id": i,
        "name": v
    })
    if total_class[i] > 0:
        ap.append(true_class[i] / total_class[i])
        print(v, true_class[i], "de", total_class[i], true_class[i] / total_class[i])
    else:
        print(v, "SIN DATOS")
print("Promedio de precisiones", np.sum(ap)/len(ap))
coco_eval(ann_json, pred_json)

