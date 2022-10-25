#!/bin/bash --login

#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J PRET_MIX_C1
#SBATCH -o /home/slimhy/logs/shape_classifier/PRET_MIX_C1.out
#SBATCH -e /home/slimhy/logs/shape_classifier/PRET_MIX_C1.err
#SBATCH --mail-user=habib.slim@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:2


source /home/slimhy/miniconda3/bin/activate

PROJECT_DIR=/home/slimhy/shape_classifier/
cd "$PROJECT_DIR"

# run the application:
python main.py --num-workers 4 \
    --use-tmp \
    --exp-name PRET_MIX_C1 \
    --exp-tag PRET-MIX \
    --root-url /var/remote/lustre/scratch/project/k1546/shards/ \
    --models-dir /ibex/scratch/slimhy/models/shape_classifier/PRET_MIX_C1/ \
    --batch-size 64 \
    --nbatches 10000 \
    --num-classes 41 \
    --resnet-type resnet50 \
    --seed 222 \
    --n-comp 1 \
    --view-type all \
    --use-pretrained \
