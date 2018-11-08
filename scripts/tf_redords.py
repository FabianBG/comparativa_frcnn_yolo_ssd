import tensorflow as tf
import cv2 as cv
import os
import io
import numpy as np
from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('train_path', '', 'Path to output TFRecord Train')
flags.DEFINE_string('test_path', '', 'Path to output TFRecord Test')
flags.DEFINE_string('src_path', '', 'Path of iamges folder')
flags.DEFINE_float('divide_ratio', 0.7, 'Ration of division of train/eval sets')
FLAGS = flags.FLAGS

image_ext = ".jpg"

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

def get_classes(path):
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
    file = open("classes.pbtxt","w") 
    file.write(label_map) 
    file.close()
    return clean_lines
  return ["default"]

def create_csv(src_folder, filename, classes):
  lines = [];
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


def create_tf_record(src_folder, filename, classes):
  #print("Trabajando archivo", filename, image_ext)
  record = {
        "classes_text": [],
        "classes_index": [],
        "xmax": [],
        "xmin": [],
        "ymax": [],
        "ymin": [],
        "image_encoded": None
      }
  with open(os.path.join(src_folder, filename + ".txt")) as boxes:
    image = cv.imread(os.path.join(src_folder, filename + image_ext))
    image_format = b'jpg'
    fid = tf.gfile.GFile(os.path.join(src_folder, filename + image_ext), 'rb')
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
        'image/filename': dataset_util.bytes_feature((filename + image_ext).encode("utf8")),
        'image/source_id': dataset_util.bytes_feature((filename + image_ext).encode("utf8")),
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

def divide_set(data, train_percentaje):
  clean_data = []
  for filename in data:
   if filename.endswith(image_ext):
     clean_data.append(filename)
  np.random.shuffle(clean_data)
  size = len(clean_data)
  train = int(size * train_percentaje)
  print("Total de archivos {} para entrenamiento {}".format(size, train))
  return clean_data[:train], clean_data[train:]

def generate_records(output, files, classes):
  writer = tf.python_io.TFRecordWriter(output)
  index = 0
  for filename in files:
    index = index + 1
    tf_example = create_tf_record(FLAGS.src_path, filename.split(".")[0], classes)
    writer.write(tf_example.SerializeToString())
  writer.close()
  print("Se escribio ", str(index), "datos de", output)

def generate_records_csv(output, files, classes):
  writer = tf.python_io.TFRecordWriter(output)
  index = 0
  lines = ["filename,width,height,class,xmin,ymin,xmax,ymax"]
  for filename in files:
    index = index + 1
    lines = lines + create_csv(FLAGS.src_path, filename.split(".")[0], classes)
  file=open(output  + ".csv", 'w')
  file.write('\n'.join(lines))
  file.close()
  

def main(_):
  classes = get_classes(os.path.join(FLAGS.src_path, "classes.txt"))
  train, test = divide_set(os.listdir(FLAGS.src_path), FLAGS.divide_ratio)
  generate_records(FLAGS.train_path, train, classes)
  generate_records(FLAGS.test_path, test, classes)
  generate_records_csv(FLAGS.train_path, train, classes)
  generate_records_csv(FLAGS.test_path, test, classes)
  
  


if __name__ == '__main__':
  tf.app.run()
