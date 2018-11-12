#!/bin/bash
#
#BSUB -J object-detector      # job name
#BSUB -n 1                   # number of tasks in job
#BSUB -q normal              # queue
#BSUB -e /home/mbastidas/waldo.err     # error file name in which %J is replaced by the job ID
#BSUB -o /home/mbastidas/waldo.out     # output file name in which %J is replaced by the job ID

#BSUB -R "span[ptile=4]"      # run four MPI tasks per node
#BSUB -P  yachay-ep                    # project code
 
./generate.sh 
