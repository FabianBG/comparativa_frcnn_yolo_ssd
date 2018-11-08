#!/bin/bash
#
#BSUB -J object-detector      # job name
#BSUB -n 4                   # number of tasks in job
#BSUB -q normal              # queue
#BSUB -oe /home/mbastidas/bonsai.err     # error file name in which %J is replaced by the job ID
#BSUB -oo /home/mbastidas/bonsai.out     # output file name in which %J is replaced by the job ID

#BSUB -R "span[ptile=4]"      # run four MPI tasks per node
#BSUB -P  yachay-ep                    # project code
 
./generate.sh 
