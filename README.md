
# Comparativa de modelos SSD, FRCNN y YOLO

El presente trabajo realizó una comparativa de modelos para detección de objetos, se uso un dataset que consta de imágenes de 14 clases de plantas debidamente labelizado y clasificado.

El presente trabajo esta licenciado bajo GNUv3.

### REQUISITOS
 * Tensorflow
 * Tensorflow Object Detector API 
 * LabelImg
 *  ImgAug 
 * Darknet 
 * CUDA

### DARKNET

Para el entrenamiento de modelos YOLO se uso Darknet.
En los archivos de configuración tener en cuenta las capas de filters para en base al numero de clases que se desea entrenar realizar el sigueinte cálculo:

* YOLOv2
`filters = num*(classes+5)`

* YOLOv3
`filters = num/3*(classes+5)`

Para generar los conjuntos de datos(entrenamiento, validación y pruebas) se usa el siguiente comando

`python3 generate_yolo_data.py [dataset] [output] [validation_split] [test_split] ~/git/comparativa_frcnn_yolo_ssd/yolo-model/ 0.9 0.8`
En donde el primer parámetro apunta a la carpeta con el conjunto de datos, el segundo la salida donde se almacenaran las imágenes y los 2 últimos el porcentaje de división del conjunto de datos. 
Por ejemplo:
`python3 generate_yolo_data.py ~/git/comparativa_frcnn_yolo_ssd/dataset/ ~/git/comparativa_frcnn_yolo_ssd/yolo-model/ 0.9 0.8`
Este script se encarga de generar automáticamente transformaciones para aumentar el tamaño del conjunto de datos.
dentro de la carpeta de cada modelo se encuentra un bash script **train.sh** con el comando para iniciar el entrenamiento.
Para ejecutar la evaluación del modelo se ejecuta el siguiente comando:
`python3 test_yolo.py [weights] [config_file] [data_file] [names_file] [images_paths]`
Por ejemplo: 
`python3 test_yolo.py ~/git/darknet/weigths/yolov2-tiny-plantas_10000.weights ~/git/comparativa_frcnn_yolo_ssd/yolo-model/yolov2-tiny-plantas.cfg ~/git/comparativa_frcnn_yolo_ssd/yolo-model/plantas.data ~/git/comparativa_frcnn_yolo_ssd/yolo-model/plantas.names /home/mbastidas/git/comparativa_frcnn_yolo_ssd/yolo-model/validation.txt`

  
## Tensorflow
  Para los modelos que hacen uso de tensorflow existe otra serie de scripts que permiten realizar tareas similares a la de Darknet.
Para generar los conjuntos de datos se aplica el siguiente comando:
`python3 generate_tf_data.py --tr=[output] --ds=[dataset] --names=[names_file] --test=[test_percentaje] --val=[validation_percentaje]`
Por ejemplo:
`python3 generate_tf_data.py --tr=/home/mbastidas/git/comparativa_frcnn_yolo_ssd/ssd-model/data/ --ds=/home/mbastidas/git/comparativa_frcnn_yolo_ssd/dataset/ --names=/home/mbastidas/git/comparativa_frcnn_yolo_ssd/dataset/classes.txt --test=0.9 --val=0.8`
Para el entrenamiento al igual que Darknet cada modelo dentro de su carpeta presenta un bash script que permite iniciar el entrenamiento de dicho modelo.

Para poder probar el modelo es necesario exportarlo como grafo congelado, para esto se aplica el siguiente comando:
`sh export_graph.sh [pipeline_config] [checkpoint] [output]`
Por ejemplo:
`sh export_graph.sh /home/mbastidas/git/comparativa_frcnn_yolo_ssd/ssd-model/models/ssd_mobilnetv2/pipeline.config /home/mbastidas/git/comparativa_frcnn_yolo_ssd/ssd-model/modes/ssd_mobilnetv2/train/model.ckpt-2226 ./graph`
Finalmente para realizar la evaluación del modelo se ejecuta el siguiente comando:
`python3 test_tf.py --images=[images_paths] --labels=[labels_pbtxt] --graph=[graph_pb]`
Por ejemplo:
`python3 test_tf.py --images=/home/mbastidas/git/comparativa_frcnn_yolo_ssd/ssd-model/data/test.txt --labels=/home/mbastidas/git/comparativa_frcnn_yolo_ssd/ssd-model/data/classes.pbtxt --graph=/home/mbastidas/git/comparativa_frcnn_yolo_ssd/ssd-model/graph/frozen_inference_graph.pb`