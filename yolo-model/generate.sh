module load cuda/8.0
module load cudnn/8.0.0

/home/mbastidas/git/darknet/darknet detector train ./plantas.data ./yolov2-tiny-plantas.cfg ./darknet19_448.conv.23 -gpus 0,1,2,3
