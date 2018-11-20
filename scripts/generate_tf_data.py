
import tensorflow as tf
import os
from utilities import generate_classes_pbtxt, generate_tf_records
from utilities import generate_datasets, save_file, generate_augmentations




flags = tf.app.flags
flags.DEFINE_string('tr', '', 'Path del archivos a generar de entrenamiento, validación y pruebas')
flags.DEFINE_string('ds', '', 'Path del conjunto de imagenes')
flags.DEFINE_string('names', '', 'Archivo con las clases de los objetos')
flags.DEFINE_float('test', 0.7, 'Porcenjaje de division de datos')
flags.DEFINE_float('val', 0.7, 'Porcenjaje de division de datos')
flags.DEFINE_integer('aug', 0, 'Numero de veces a aumentar una imagen')
FLAGS = flags.FLAGS

shards_train = 20
shards_validate = 10
maximun_aug = -1 #-1 para aumentar completo

def main(_):
    classes = generate_classes_pbtxt(FLAGS.names, FLAGS.tr)
    train, validate, test, _ = generate_datasets(FLAGS.ds,
        FLAGS.val, FLAGS.test)

    if FLAGS.aug != 0:
        print("Iniciando aumentacion de datos train")
        train = train + generate_augmentations(train, 'augdata', FLAGS.aug, maximun_aug)
        print("Iniciando aumentacion de datos validate")
        validate = validate + generate_augmentations(validate, 'augdata', FLAGS.aug , maximun_aug)
        print("Fin aumentacion de datos")

    generate_tf_records(os.path.join(FLAGS.tr, "train.record") ,train, classes, shards=shards_train)
    generate_tf_records(os.path.join(FLAGS.tr, "validate.record"), 
        validate, classes, csv='./test.csv', shards=shards_validate)
    save_file(os.path.join(FLAGS.tr, "test.txt"), test)
    print("Se escribio  {} datos de {}".format(len(test), os.path.join(FLAGS.tr, "test.txt")))


  
if __name__ == '__main__':
    tf.app.run()

