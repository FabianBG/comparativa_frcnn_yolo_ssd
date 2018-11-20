from object_detection.utils import dataset_util
import os
import sys
import cv2 as cv
import numpy as np
import tensorflow as tf
import random
import sys
from shutil import copy2
sys.path.append(os.path.join(
    '/home/mbastidas/git/tensorflow_models/research/'))
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
import contextlib2
from object_detection.dataset_tools import tf_record_creation_util

import imgaug as ia
from imgaug import augmenters as iaa

STANDARD_COLORS = [(255,248,240), (215,235,250), (255,255,0), (212,255,127),
(255,255,240), (220,245,245), (196,228,255), (0,0,0),
(205,235,255), (255,0,0), (226,43,138), (42,42,165),
(135,184,222), (160,158,95), (0,255,127),
(30,105,210), (80,127,255),(237,149,100), (220,248,255),
(60,20,220), (255,255,0),(139,0,0), (139,139,0),
(11,134,184), (169,169,169),(0,100,0), (169,169,169),
(107,183,189), (139,0,139),(47,107,85), (0,140,255),
(204,50,153), (0,0,139),(122,150,233), (143,188,143),
(139,61,72), (79,79,47),(79,79,47), (209,206,0),
(211,0,148), (147,20,255),(255,191,0), (105,105,105),
(105,105,105), (255,144,30),(34,34,178), (240,250,255),
(34,139,34), (255,0,255),(220,220,220), (255,248,248),
(0,215,255), (32,165,218),(128,128,128), (0,128,0),
(47,255,173),(128,128,128),(240,255,240),(180,105,255),
(92,92,205),(130,0,75),(240,255,255),(140,230,240),
(250,230,230),(245,240,255),(0,252,124),(205,250,255),
(230,216,173),(128,128,240),(255,255,224),(210,250,250),
(211,211,211),(144,238,144),(211,211,211),(193,182,255),
(122,160,255),(170,178,32),(250,206,135),
(153,136,119),(153,136,119),(222,196,176),(224,255,255),
(0,255,0),(50,205,50),(230,240,250),(255,0,255),
(0,0,128),(170,205,102),(205,0,0),(211,85,186),
(219,112,147),(113,179,60),(238,104,123),(154,250,0),
(204,209,72),(133,21,199),(112,25,25),(250,255,245),
(225,228,255),(181,228,255),(173,222,255),(128,0,0),
(230,245,253),(0,128,128),(35,142,107),
(0,165,255),(0,69,255),(214,112,218),(170,232,238),
(152,251,152),(238,238,175),(147,112,219),(213,239,255),
(185,218,255),(63,133,205),(203,192,255),(221,160,221),
(230,224,176),(128,0,128),(0,0,255),(143,143,188),
(225,105,65),(19,69,139),(114,128,250),(96,164,244),
(87,139,46),(238,245,255),(45,82,160),(192,192,192),
(235,206,135),(205,90,106),(144,128,112),(144,128,112),
(250,250,255),(127,255,0),(180,130,70),(140,180,210),
(128,128,0),(216,191,216),(71,99,255),(208,224,64),
(238,130,238),(179,222,245),(255,255,255),(245,245,245),
(0,255,255),(50,205,154)]


def generate_datasets(dataset_path, train_percentaje, val_percentaje, yolo_path=None, image_ext=".jpg"):
    train = []
    validate = []
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
                        copy2(os.path.join(
                            directory, filename.split('.')[0] + '.txt'), dest)
                    else:
                        dest = directory
                    images.append(os.path.join(dest, filename))
            perm = np.random.permutation(len(images))
            images = np.array(images)
            images = images[perm]
            partition = int(len(images) * train_percentaje)
            validation = int(partition * val_percentaje)
            train = train + images[0:validation].tolist()
            validate = validate + images[validation:partition].tolist()
            test = test + images[partition:].tolist()
            print("En {} se usa: \n{} entrenamiento \n{} validacion \n{} pruebas"
                  .format(directory, validation, partition - validation, len(images) - validation))
    shuffle_array(train)
    shuffle_array(validate)
    return train, validate, test, folders


def shuffle_array(array):
    random.shuffle(array)

def save_file(path, data):
    with open(path, "w") as f:
        f.seek(0)
        for d in data:
            f.writelines("%s\n" % d)
        f.truncate()

def read_file(path):
    salida = []
    with open(path, "r") as f:
        for line in f:
            salida.append(line.replace("\n", ""))
    return salida


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

def read_yolo_paths(paths):
    data = []
    for filename in paths:
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
            name = name.replace('\n', '')
            indexes[name] = index
            classes[index] = name
            index = index + 1
    return indexes, classes


def draw_box(image, class_name, points_yolo, acc, class_index=0, thickness=5):
    x = int(points_yolo[0] - (points_yolo[2]/2))
    y = int(points_yolo[1] - (points_yolo[3]/2))
    x1 = int(points_yolo[2] + x)
    y1 = int(points_yolo[3] + y)
    

    cv.rectangle(image, (x, y), (x + (25 * len(class_name)) , y - 20), STANDARD_COLORS[class_index], -1)
    cv.rectangle(image, (x, y), (x1, y1), STANDARD_COLORS[class_index], thickness)
    cv.putText(image, "%.2f %s" % (round(acc, 2), class_name), (x, y-5),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv.LINE_AA)

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
    return (x, y, w, h)  # box x1 y1 x2 y2
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

    return (b0, b1, b2, b3)  # box x1 x2 y1 y2


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
}
'''.replace("line", line.replace('\n', '')).replace("index", str(index))
            label_map = label_map + label
            index = index + 1
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
    with open(image.split('.')[0] + ".txt") as boxes:
        image_data = cv.imread(image)
        image_format = b'jpg'
        fid = tf.gfile.GFile(image, 'rb')
        record["image_encoded"] = fid.read()
        for box in boxes.readlines():
            box = box.split(" ")
            record["classes_text"].append(classes[int(box[0])].encode("utf8"))
            record["classes_index"].append(int(box[0]) + 1)
            limits = convert_axis(image_data.shape, box[1:])
            record["xmin"].append(limits[0] / image_data.shape[1])
            record["xmax"].append(limits[1] / image_data.shape[1])
            record["ymin"].append(limits[2] / image_data.shape[0])
            record["ymax"].append(limits[3] / image_data.shape[0])
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(image_data.shape[0]),
            'image/width': dataset_util.int64_feature(image_data.shape[1]),
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


def create_csv(image_path, classes, image_ext=".jpg"):
    lines = []
    # filename,width,height,class,xmin,ymin,xmax,ymax
    with open(image_path.split('.')[0] + ".txt") as boxes:
        image = cv.imread(image_path)
        for box in boxes.readlines():
            box = box.split(" ")
            limits = convert_axis(image.shape, box[1:])
            lines.append("{},{},{},{},{},{},{},{}".format(image_path, image.shape[1], image.shape[0],
                                                          classes[int(
                                                              box[0])], limits[0], limits[2],
                                                          limits[1], limits[3]))
    return lines


def generate_tf_records(output, files, classes, shards=1, csv=None):
    #writer = tf.python_io.TFRecordWriter(output)
    index = 0
    csv_file = []
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output, shards)
        index = 0
        for filename in files:
            tf_example = create_tfrecord_from_yolo(filename, classes)
            output_shard_index = index % shards
            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
            index = index + 1
            if csv:
                csv_file = csv_file + create_csv(filename, classes)
                save_file(csv ,csv_file)

        print("Se escribio ", str(index), "datos de", output)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def run_inference_multiple_images(images, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            output = []
            for data in images:
                image_np = cv.imread(data['image'])
                image_np = cv.cvtColor(image_np, cv.COLOR_BGR2RGB)
                output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image_np, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(
                    output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                output_dict['path'] = data['image']
                output_dict['image'] = image_np
                output_dict['boxes'] = data['boxes']
                output.append(output_dict)

    return output

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def load_graph(path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def aumentation_properties():
    sometimes = lambda aug: iaa.Sometimes(0.8, aug)
    return iaa.Sequential([
        iaa.Fliplr(0.5), # horizontally flip 50% of all images

        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.7))
        ),
     # Either drop randomly 1 to 10% of all pixels (i.e. set
                # them to black) or drop them on an image with 2-5% percent
                # of the original size, leading to large dropped
                # rectangles.
        iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.2
                    ),
                ]),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.6), "y": (0.6)},
            rotate=(-45, 45),
            shear=(-8, 8)
        )
    ], random_order=True)


def generate_augmentations(paths, output_dir, repeats, maximun=-1):
    images = read_yolo_paths(paths)
    seq = aumentation_properties()
    print("Aumentanto {} x {}".format(len(images), repeats))
    results = []
    check_max = 0
    for image in images:
        check_max = check_max + 1
        if check_max >= maximun and maximun != -1: break
        filename = image["image"].split("/")[-1]
        data = cv.imread(image["image"])
        boxes = []
        names = []
        for line in image["boxes"]:
            if len(line) != 5 : continue
            name, x1, y1, x2, y2 = line
            x1, x2, y1, y2 = convert_axis(data.shape,
            (float(x1), float(y1), float(x2), float(y2)))
            boxes.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
            names.append(name)
        boxes = ia.BoundingBoxesOnImage(boxes, shape=data.shape)
        seq_det = seq.to_deterministic()
        images_aug = seq_det.augment_images([data] * repeats)
        boxes_aug = seq_det.augment_bounding_boxes([boxes] * repeats)
        for i in range(0, len(images_aug)):
            image_aug = images_aug[i]
            box_aug = boxes_aug[i]
            result_filename = "aug-" + str(i) + filename
            cv.imwrite(os.path.join(output_dir, result_filename), image_aug)
            with open(os.path.join(output_dir, result_filename.split(".")[0] + ".txt"), "w") as yolo_txt:
                j = 0
                box_aug = box_aug.cut_out_of_image()
                box = box_aug.bounding_boxes
                for name in names:
                    #print(box[j].is_fully_within_image(data.shape))
                    if box[j].is_fully_within_image(data.shape):
                        x1, y1, x2, y2 = convert_yolo(data.shape, 
                        (box[j].x1, box[j].x2, box[j].y1 ,box[j].y2))
                        yolo_txt.write("%s %.4f %.4f %.4f %.4f\n" %
                        (name, x1, y1, x2, y2) )
                    j = j + 1
            i = i + 1
            results.append(os.path.join(output_dir, result_filename))

    shuffle_array(results)
    print("Datos generados ", len(results))
    return results 
        
        