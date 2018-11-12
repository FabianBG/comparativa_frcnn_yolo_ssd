# comparativa_frcnn_yolo_ssd
Comparativa de algoritmos de detección de objetos.

### YOLO

v2=num*(classes+5) v3=num/3*(classes+5)

revisar archivos de meta y config por caracteres invisibles escribir de nuevo en blanco

python3 generate_yolo_data.py ~/git/comparativa_frcnn_yolo_ssd/dataset/ ~/git/comparativa_frcnn_yolo_ssd/yolo-model/ 0.9 0.8

python3 test_yolo.py ~/git/darknet/weigths/yolov2-tiny-plantas_10000.weights ~/git/comparativa_frcnn_yolo_ssd/yolo-model/yolov2-tiny-plantas.cfg ~/git/comparativa_frcnn_yolo_ssd/yolo-model/plantas.data  ~/git/comparativa_frcnn_yolo_ssd/yolo-model/plantas.names /home/mbastidas/git/comparativa_frcnn_yolo_ssd/yolo-model/validation.txt 

## TF

python3 generate_tf_data.py --tr=/home/mbastidas/git/comparativa_frcnn_yolo_ssd/ssd-model/data/ --ds=/home/mbastidas/git/comparativa_frcnn_yolo_ssd/dataset/ --names=/home/mbastidas/git/comparativa_frcnn_yolo_ssd/dataset/classes.txt --test=0.9 --val=0.8

sh export_graph.sh /home/mbastidas/git/comparativa_frcnn_yolo_ssd/ssd-model/models/ssd_mobilnetv2/pipeline.config /home/mbastidas/git/comparativa_frcnn_yolo_ssd/ssd-model/models/ssd_mobilnetv2/train/model.ckpt-2226 ./graph

python3 test_tf.py --images=/home/mbastidas/git/comparativa_frcnn_yolo_ssd/ssd-model/data/test.txt --labels=/home/mbastidas/git/comparativa_frcnn_yolo_ssd/ssd-model/data/classes.pbtxt --graph=/home/mbastidas/git/comparativa_frcnn_yolo_ssd/ssd-model/graph/frozen_inference_graph.pb
