
import tensorflow as tf
import os
from utilities import generate_classes_pbtxt, generate_tf_records, read_yolo_images
from utilities import generate_datasets, save_file




flags = tf.app.flags
flags.DEFINE_string('tf_path', '', 'Path del archivos a generar de entrenamiento, validación y pruebas')
flags.DEFINE_string('ds_path', '', 'Path del conjunto de imagenes')
flags.DEFINE_string('test_path', '', 'Path del archivo a generar de pruebas')
flags.DEFINE_string('classes_file', '', 'Archivo con las clases de los objetos')
flags.DEFINE_float('divide_ratio', 0.7, 'Porcenjaje de division de datos')
FLAGS = flags.FLAGS

def main(_):
    classes = generate_classes_pbtxt(FLAGS.classes_file, FLAGS.tf_path)
    train, validate, test, _ = generate_datasets(FLAGS.ds_path,
        FLAGS.divide_ratio)

    generate_tf_records(FLAGS.tf_path ,train, classes)
    generate_tf_records(FLAGS.tf_path, test, classes)
    save_file(os.path.join(FLAGS.ds_path, "test.txt"), validate)


  
if __name__ == '__main__':
    tf.app.run()

