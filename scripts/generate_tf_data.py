
import tensorflow as tf
import os
from utilities import generate_classes_pbtxt, generate_tf_records, read_yolo_images
from utilities import generate_datasets, save_file




flags = tf.app.flags
flags.DEFINE_string('tr', '', 'Path del archivos a generar de entrenamiento, validación y pruebas')
flags.DEFINE_string('ds', '', 'Path del conjunto de imagenes')
flags.DEFINE_string('names', '', 'Archivo con las clases de los objetos')
flags.DEFINE_float('test', 0.7, 'Porcenjaje de division de datos')
flags.DEFINE_float('val', 0.7, 'Porcenjaje de division de datos')
FLAGS = flags.FLAGS

def main(_):
    classes = generate_classes_pbtxt(FLAGS.names, FLAGS.tr)
    train, validate, test, _ = generate_datasets(FLAGS.ds,
        FLAGS.val, FLAGS.test)

    generate_tf_records(os.path.join(FLAGS.tr, "train.record") ,train, classes)
    generate_tf_records(os.path.join(FLAGS.tr, "validate.record"), validate, classes, csv='./test.csv')
    save_file(os.path.join(FLAGS.tr, "test.txt"), test)
    print("Se escribio  {} datos de {}".format(len(test), os.path.join(FLAGS.tr, "test.txt")))


  
if __name__ == '__main__':
    tf.app.run()

