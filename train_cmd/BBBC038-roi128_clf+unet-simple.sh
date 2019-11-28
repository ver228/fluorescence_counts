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
--roi_size 112 \
--model_name 'clf+unet-simple' \
--loss_type 'maxlikelihood' \
--lr 128e-6 \
--num_workers 1  \
--n_epochs 500 \
--val_dist 10
