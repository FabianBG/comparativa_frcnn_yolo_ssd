import tensorflow as tf
import os
import numpy as np
import cv2 as cv
import time
from utilities import generate_classes_pbtxt, load_image_into_numpy_array, run_inference_multiple_images
from utilities import vis_util, load_graph, label_map_util, read_file, read_yolo_images, calcualte_iou
from utilities import desnormalize, convert_axis, draw_box, coco_eval
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
    classes = len(category_index)
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
            true_class = [0] * classes
            total_class = [0] * classes
            times = []
            ious = []
            ann_json = {
                "annotations":[],
                "categories":[],
                "images": []
            }
            pred_json = []
            image_id = 0
            for data in paths:
                image_np = cv.imread(data['image'])
                image_np = cv.cvtColor(image_np, cv.COLOR_BGR2RGB)
                image_id = image_id + 1

                ann_json["images"].append({
                    "id" : image_id, 
                    "width" : image_np.shape[1], 
                    "height" : image_np.shape[0], 
                    "file_name" : data['image']
                })

                start_time = time.time()
                output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image_np, 0)})
                times.append(time.time() - start_time)

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
                total = total + len(data['boxes'])
                for box in data['boxes']:
                    box_id = int(box[0])
                    total_class[box_id] = total_class[box_id] + 1
                    real_box = np.array(box[1:], dtype=float)
                    real_box = desnormalize(real_box, image_np.shape)
                    coco_box = convert_axis(image_np.shape, np.array(box[1:], dtype=float))
                    ann_json["annotations"].append({
                        "id" : image_id,
                        "image_id" : image_id,
                        "category_id" : box_id + 1, 
                        #"segmentation" : RLE or [polygon], 
                        #"area" : float, 
                        "bbox" : [coco_box[0],coco_box[2],
                                            coco_box[1] - coco_box[0],coco_box[3] - coco_box[2]], 
                        "iscrowd" : 0,
                        "area": (coco_box[1] - coco_box[0]) * (coco_box[3] - coco_box[2])
                    })
                    
                    for pred in best_results:
                        if output_dict['detection_classes'][pred] - 1 == int(box[0]):
                            positives = positives + 1
                            true_class[box_id] = true_class[box_id] + 1
                            pred_box = desnormalize(output_dict['detection_boxes'][pred], image_np.shape)
                            iou = calcualte_iou(real_box, pred_box, image_np.shape)
                            print("IOU", iou)
                            draw_box(image_np, "real", real_box, 100, 
                            class_index=20)   
                            ious.append(iou)
                            pred_json.append({
                                "image_id" : image_id, 
                                "category_id" : int(output_dict['detection_classes'][pred]), 
                                "bbox" : [pred_box[0],pred_box[1],
                                            pred_box[2] - pred_box[0],pred_box[3] - pred_box[1]], 
                                "score" : float(output_dict['detection_scores'][pred])
                            })
                            break

                name = output_dict['path'].split("/")[-1]
                image_np = cv.cvtColor(output_dict['image'], cv.COLOR_RGB2BGR)
                  
                cv.imwrite("./predicts/{}".format(name), image_np)

            print("Acertados {} de {}".format(positives, total))
            print("Porcentage {}".format(positives / total))
            print("Tiempo de inferencia promedio {} segundos".format(np.sum(times)/len(times)))
            print("IOU promedio ~ {}".format(np.sum(ious)/len(ious)))
            ap = []
            for key in category_index:
                i = category_index[key]['id'] - 1
                v = category_index[key]['name']
                ann_json['categories'].append(category_index[key])
                if total_class[i] > 0:
                    ap.append(true_class[i] / total_class[i])
                    print(v, true_class[i], "de", total_class[i], true_class[i] / total_class[i])
                else:
                    print(v, "SIN DATOS")
            print("Promedio de precisiones", np.sum(ap)/len(ap))
            coco_eval(ann_json, pred_json)



  
if __name__ == '__main__':
    tf.app.run()
