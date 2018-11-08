import os
import sys
import cv2 as cv



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


def convert_axis(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = box[0]/dw
    w = box[2]/dw
    y = box[1]/dh
    h = box[3]/dh

    b1 = x + (w/2)
    b3 = y + (h/2)
    b0 = x + (w/2) - w
    b2 = y + (h/2) - h

    return (b0, b1, b2, b3)