import tensorflow as tf
import os
import numpy as np
import cv2 as cv
from utilities import generate_classes_pbtxt, load_image_into_numpy_array, run_inference_for_single_image
from utilities import vis_util, load_graph, label_map_util, read_file
from PIL import Image


flags = tf.app.flags
flags.DEFINE_string('images', '', 'Path del archivos para verificar')
flags.DEFINE_string('labels', '', 'Path del con las clases')
flags.DEFINE_string('graph', '', 'Path del checkpoint del grafo')
FLAGS = flags.FLAGS


def main(_):
    detection_graph = load_graph(FLAGS.graph)
    paths = read_file(FLAGS.images)
    category_index = label_map_util.create_category_index_from_labelmap(FLAGS.labels, use_display_name=True)
    for image_path in paths:
        image_np = cv.imread(image_path)
        image_np = cv.cvtColor(image_np, cv.COLOR_BGR2RGB)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        #image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        #image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        print(image_path, "detections {}".format(output_dict['num_detections']))
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
        name = image_path.split("/")[-1]
        image_np = cv.cvtColor(image_np, cv.COLOR_RGB2BGR)
        cv.imwrite("./predicts/{}".format(name), image_np)


  
if __name__ == '__main__':
    tf.app.run()
