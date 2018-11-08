# comparativa_frcnn_yolo_ssd
Comparativa de algoritmos de detección de objetos.

### YOLO

v2=num*(classes+5) v3=num/3*(classes+5)

revisar archivos de meta y config por caracteres invisibles escribir de nuenvo en blanco

 python3 test_yolo.py ~/git/darknet/weigths/yolov2-tiny-plantas_10000.weights ~/git/comparativa_frcnn_yolo_ssd/yolo-model/yolov2-tiny-plantas.cfg ~/git/comparativa_frcnn_yolo_ssd/yolo-model/plantas.data  ~/git/comparativa_frcnn_yolo_ssd/yolo-model/plantas.names /home/mbastidas/git/comparativa_frcnn_yolo_ssd/yolo-model/validation.txt