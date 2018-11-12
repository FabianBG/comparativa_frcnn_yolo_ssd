export PYTHONPATH=:/home/mbastidas/git/tensorflow_models/research:/home/mbastidas/git/tensorflow_models/research/slim

INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=$1
TRAINED_CKPT_PREFIX=$2
EXPORT_DIR=$3
python3 /home/mbastidas/git/tensorflow_models/research/object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
