module load cuda/8.0
module load cudnn/8.0.0
module load python/3.5.3
module load openblas/0.2.19
#alias python=python3

export PYTHONPATH=:/home/mbastidas/git/models_tf/research:/home/mbastidas/git/models_tf/research/slim


if [ $# -gt 0 ]
then
    echo CLUSTER MODE
    export TF_CONFIG=$(cat $1)
fi

#echo $TF_CONFIG > /home/mbastidas/$(hostname).txt
#rm $1


cd /home/mbastidas/git/models_tf/research/
PIPELINE_CONFIG_PATH=/home/mbastidas/git/comparativa_frcnn_yolo_ssd/ssd-model/models/ssd_resnet/pipeline.config
MODEL_DIR=/home/mbastidas/git/comparativa_frcnn_yolo_ssd/ssd-model/models/ssd_resnet/train
NUM_TRAIN_STEPS=200000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python3 object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR}  \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
