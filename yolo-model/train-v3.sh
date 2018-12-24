module load cuda/8.0
module load cudnn/8.0.0

/home/mbastidas/git/darknet/darknet detector train ./plantas.data ./yolov3-tiny-plantas.cfg ./darknet53.conv.74 -gpus 0,1,2,3
