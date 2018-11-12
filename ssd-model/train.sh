module load cuda/8.0
module load cudnn/8.0.0
module load python/3.5.3
module load openblas/0.2.19
#alias python=python3
export PYTHONPATH=:/home/mbastidas/git/models_tf/research:/home/mbastidas/git/models_tf/research/slim


cd /home/mbastidas/git/models_tf/research/ 
python3 object_detection/model_main.py  --pipeline_config_path=/home/mbastidas/git/tf-models/waldo/models/ssd_mobilnetv2/ssd_mobilenet_v2_coco.config --model_dir=/home/mbastidas/git/tf-models/waldo/models/ssd_mobilnetv2/ --num_train_steps=50000 --sample_1_of_n_eval_examples=1 --alsologtostderr
