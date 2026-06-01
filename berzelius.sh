#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 1:00:00
#SBATCH --output /proj/chalmers-cv/users/%u/logs/%A_%a.out
#SBATCH -A berzelius-2026-138
#

assignment=${1:?"No assignment specified."}
cd $assignment
echo "Running $assignment with arguments: ${@:2}"

singularity exec --nv /proj/chalmers-cv/users/x_tiawu/containers/dl4nlp.sif ${@:2}

#
#EOF
