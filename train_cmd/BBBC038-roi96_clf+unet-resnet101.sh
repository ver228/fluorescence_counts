#!/bin/bash
#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1


export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/cell_localization/scripts

python -W ignore train_locmax.py \
--batch_size 128  \
--data_type 'BBBC038-fluorescence' \
--flow_type 'lymphocytes' \
--roi_size 96 \
--model_name 'clf+unet-resnet101' \
--loss_type 'maxlikelihood' \
--lr 128e-6 \
--num_workers 4  \
--n_epochs 500 \
--val_dist 10