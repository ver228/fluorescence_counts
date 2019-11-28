
CUDA_VISIBLE_DEVICES=0 

python train.py --model_name 'seg+unet-simple+n2n' --coord_type 'segmentation+noise2noise' --loss_type crossentropy+W1-1-10 --crops_same_image  True --ignore_bad_crops True
python train.py --model_name 'seg+unet-simple+n2n' --coord_type 'segmentation+noise2noise' --loss_type crossentropy+W1-1-10 --crops_same_image  False --ignore_bad_crops False

python train.py --model_name 'clf+unet-simple+n2n' --coord_type 'centroids+noise2noise' --loss_type maxlikelihood --crops_same_image  True --ignore_bad_crops True
python train.py --model_name 'clf+unet-simple+n2n' --coord_type 'centroids+noise2noise' --loss_type maxlikelihood --crops_same_image  False --ignore_bad_crops False
