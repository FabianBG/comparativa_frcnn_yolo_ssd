import os
import sys
import cv2 as cv
import numpy as np
import tensorflow as tf
import sys
from shutil import copy2
sys.path.append(os.path.join('/home/mbastidas/git/tensorflow_models/research/'))
from object_detection.utils import dataset_util

def generate_datasets(dataset_path, train_percentaje, yolo_path=None, image_ext=".jpg"):
    train = []
    valdiate = []
    test = []
    folders = []
    for path in os.listdir(dataset_path):
        directory = os.path.join(dataset_path, path)
        if len(directory.split(".")) == 1:
            folders.append(directory)
            images = []
            for filename in os.listdir(directory):
                if filename.endswith(image_ext):
                    if yolo_path:
                        dest = os.path.join(yolo_path, 'data')
                        copy2(os.path.join(directory, filename), dest)
                        copy2(os.path.join(directory, filename.split('.')[0] + '.txt'), dest)
                    else:
                        dest = directory
                    images.append(os.path.join(dest, filename))
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
    return train, valdiate, test, folders

def save_file(path, data):
    with open(path, "w") as f:
        f.seek(0)
        for d in data:
            f.writelines("%s\n" % d)
        f.truncate()

def read_yolo_images(paths, image_ext='jpg'):
    data = []
    with open(paths) as path:
        for filename in path:
            with open(filename.split('.')[0] + '.txt') as boxes_file:
                boxes = []
                for line in boxes_file:
                    boxes.append(line.replace('\n', '').split(' '))
                data.append({
                    "image": filename.replace('\n', ''),
                    "boxes": boxes
                })
        return data


def classes_as_dicts(classes_file):
    indexes = {}
    classes = {}
    with open(classes_file) as names:
        index = 0
        for name in names:
            name = name.replace('\n','')
            indexes[name] = index
            classes[index] = name
            index = index + 1
    return indexes, classes

def draw_box(filename, new_name, class_name, points_yolo, acc, color=(0, 140, 0)):
    image = cv.imread(filename)
    x = int(points_yolo[0])
    y = int(points_yolo[1])
    x1 = int(points_yolo[2])
    y1 = int(points_yolo[3])
    
    cv.rectangle(image, (x,y), (x + 50,y - 15), color, -1)
    cv.rectangle(image, (x,y), (x1,y1), color, 2)
    cv.putText(image, "%.2f %s" % (round(acc, 2), class_name), (x, y-5),
     cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)      
    cv.imwrite('./predicts/' +  new_name, image)

# box x1 x2 y1 y2
def convert_yolo(size, box):
    dw = 1./size[1]
    dh = 1./size[0]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)# box x1 y1 x2 y2
# box x1 y1 x2 y2
def convert_axis(size, box):
    dw = 1./size[1]
    dh = 1./size[0]
    x = float(box[0])/dw
    w = float(box[2])/dw
    y = float(box[1])/dh
    h = float(box[3])/dh

    b1 = x + (w/2)
    b3 = y + (h/2)
    b0 = x + (w/2) - w
    b2 = y + (h/2) - h

    return (b0, b1, b2, b3)# box x1 x2 y1 y2

def generate_classes_pbtxt(path, output_path):
    with open(path, 'r') as classes:
        label_map = ""
        index = 1
        lines = classes.readlines()
        clean_lines = []
        for line in lines:
            clean_lines.append(line.replace('\n', ''))
            label = '''item {
            id: index      
            name: 'line'
            }'''.replace("line", line.replace('\n', '')).replace("index", str(index))
            label_map = label_map + label
        file = open(os.path.join(output_path, "classes.pbtxt"), "w") 
        file.write(label_map) 
        file.close()
        return clean_lines
    return ["default"]

def create_tfrecord_from_yolo(image, classes):
    record = {
        "classes_text": [],
        "classes_index": [],
        "xmax": [],
        "xmin": [],
        "ymax": [],
        "ymin": [],
        "image_encoded": None
    }
    with open(image.split["."][0] + ".txt") as boxes:
        image = cv.imread(image)
        image_format = b'jpg'
        fid = tf.gfile.GFile(image, 'rb')
        record["image_encoded"] = fid.read()
        for box in boxes.readlines():
            box = box.split(" ")
            record["classes_text"].append(classes[int(box[0])].encode("utf8"))
            record["classes_index"].append(int(box[0]) + 1)
            limits = convert_axis(image.shape, box[1:])
            record["xmin"].append(limits[0] / image.shape[1])
            record["xmax"].append(limits[1] / image.shape[1])
            record["ymin"].append(limits[2] / image.shape[0])
            record["ymax"].append(limits[3] / image.shape[0])
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(image.shape[0]),
            'image/width': dataset_util.int64_feature(image.shape[1]),
            'image/filename': dataset_util.bytes_feature(image.encode("utf8")),
            'image/source_id': dataset_util.bytes_feature(image.encode("utf8")),
            'image/encoded': dataset_util.bytes_feature(record["image_encoded"]),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(record["xmin"]),
            'image/object/bbox/xmax': dataset_util.float_list_feature(record["xmax"]),
            'image/object/bbox/ymin': dataset_util.float_list_feature(record["ymin"]),
            'image/object/bbox/ymax': dataset_util.float_list_feature(record["ymax"]),
            'image/object/class/text': dataset_util.bytes_list_feature(record["classes_text"]),
            'image/object/class/label': dataset_util.int64_list_feature(record["classes_index"]),
        }))
    return tf_example

def create_csv(src_folder, filename, classes, image_ext=".jpg"):
    lines = []
    #filename,width,height,class,xmin,ymin,xmax,ymax
    with open(os.path.join(src_folder, filename + ".txt")) as boxes:
        image = cv.imread(os.path.join(src_folder, filename + image_ext))
        for box in boxes.readlines():
            box = box.split(" ")
            limits = convert_axis(image.shape, box[1:])
            lines.append("{},{},{},{},{},{},{},{}".format(filename + image_ext, image.shape[1], image.shape[0],
                classes[int(box[0])], limits[0], limits[2], 
                limits[1], limits[3]))
    return lines

def generate_tf_records(output, files, classes):
    writer = tf.python_io.TFRecordWriter(output)
    index = 0
    for filename in files:
        index = index + 1
        tf_example = create_tfrecord_from_yolo(filename, classes)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print("Se escribio ", str(index), "datos de", output)