import tensorflow as tf
import os
import numpy as np
import cv2 as cv
from utilities import generate_classes_pbtxt, load_image_into_numpy_array, run_inference_multiple_images
from utilities import vis_util, load_graph, label_map_util, read_file, read_yolo_images
from PIL import Image


flags = tf.app.flags
flags.DEFINE_string('images', '', 'Path del archivos para verificar')
flags.DEFINE_string('labels', '', 'Path del con las clases')
flags.DEFINE_string('graph', '', 'Path del checkpoint del grafo')
FLAGS = flags.FLAGS


def main(_):
    detection_graph = load_graph(FLAGS.graph)
    paths = read_yolo_images(FLAGS.images)
    category_index = label_map_util.create_category_index_from_labelmap(FLAGS.labels, use_display_name=True)
    with detection_graph.as_default():
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

            ## Start predictions
            positives = 0
            total = 0
            for data in paths:
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
                print(output_dict['path'])
                vis_util.visualize_boxes_and_labels_on_image_array(
                output_dict['image'],
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
                
                best_results = np.where(output_dict['detection_scores'] > 0.5)[0]

                for box in output_dict['boxes']:
                    if len(best_results) == 0:
                        total = total + 1
                    for pred in best_results:
                        total = total + 1
                        if output_dict['detection_classes'][pred] - 1 == int(box[0]):
                            positives = positives + 1
                            break

                name = output_dict['path'].split("/")[-1]
                image_np = cv.cvtColor(output_dict['image'], cv.COLOR_RGB2BGR)
                cv.imwrite("./predicts/{}".format(name), image_np)

            print("Acertados {} de {}".format(positives, total))
            print("Porcentage {}".format(positives / total))



  
if __name__ == '__main__':
    tf.app.run()
