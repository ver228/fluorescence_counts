#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:12:10 2018

@author: avelinojaver
"""
from pathlib import Path
import random
import cv2
import numpy as np

from torch.utils.data import Dataset 
import tqdm
import pickle


from .utils import OverlapsTracker

from .transforms_image import AddRandomRectangle, RandomBlur, AddGaussNoise, ConvertToPoissonNoise
from .transforms_outputs import Contours2Outputs

from cell_localization.flow.transforms import ( crop_contour, RandomCrop, AffineTransform, RandomVerticalFlip, 
RandomHorizontalFlip, ToTensor, Compose )


class AffineTransformBounded(AffineTransform):
    def __init__(self, *args, min_crop_size = 128, **argkws):
        super().__init__( *args, **argkws)
        self.min_crop_size =  min_crop_size
    
    def __call__(self, image, target):
        theta = np.random.uniform(*self.rotation_range)
        
        _zoom = random.uniform(*self.zoom_range)
#        if self.max_crop_size is not None:
#            _max_zoom = self.max_crop_size/max(image.shape)
#            _zoom = min(_max_zoom,_zoom)
        
        if self.min_crop_size is not None:
            _min_zoom = self.min_crop_size/max(image.shape)
            _zoom = max(_min_zoom,_zoom)
        
        #https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
     
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), theta, _zoom)
        cos_val = np.abs(M[0, 0])
        sin_val = np.abs(M[0, 1]) #this value is already scaled (zoomed) in the rotation matrix
     
        # compute the new bounding dimensions of the image
        nW = int(((h * sin_val) + (w * cos_val)))
        nH = int(((h * cos_val) + (w * sin_val)))
     
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        
        return self._apply_transform(image, target, M, img_shape = (nW, nH))
    

#%%

class MaskedBase():
    def __init__(self, 
                 crop_data, 
                 q_range, 
                 int_scale = (0, 255),
                 null_value = 0,  
                 merge_by_prod = False,
                 
                 ):        
        assert len(q_range) == 2
        
        
        #mask image
        img = crop_data['image'].astype(np.float32)
        img = np.ma.masked_equal(img, null_value)
        
        #scale
        img = (img - int_scale[0])/(int_scale[1] - int_scale[0])
        
        if img.ndim == 2:
            base_range = np.percentile(img.compressed(), q_range)
        elif img.ndim == 3:
            base_range = [np.percentile(img[..., i].compressed(), q_range) for i in range(3)]
                
        else:
            raise ValueError(f'Invalid mumber of dimensions `{img.ndim}` in image')
        q_max = np.max(img, axis = (0,1))
        
        self.target = {k:v for k,v in crop_data.items() if k != 'image'}
        
        if 'contours' in self.target:
            self.target['contours'] = [x.astype(np.float32) for x in self.target['contours']]
        
        
        self.image = img
        self.intensity_base_range = base_range
        self.intensity_max = q_max
        
        self.merge_by_prod = merge_by_prod
        
        

    def _get_random_base(self):
        #obtain parameters...
        if len(self.intensity_base_range) == 2:
            base_val = random.uniform(*self.intensity_base_range)
        else:
            #in three dimensions i want a random value along the ranges for each color
            base_val = [random.uniform(*x) for x in self.intensity_base_range]
        return base_val
            
class MaskedCrop(MaskedBase):
    def __init__(self, 
                 *args, 
                 zoom_range = (0.9, 1.1), 
                 min_crop_size = 5,
                 max_crop_size = 128,
                 crop_intensity_range = (0.025, 1.),
                 **argkws):
        
        
        super().__init__(*args, **argkws)
        self.crop_intensity_range = crop_intensity_range
        transforms = [AffineTransformBounded(zoom_range = zoom_range, min_crop_size = min_crop_size),
                        RandomVerticalFlip(), 
                        RandomHorizontalFlip(),
                        ]
        self.transforms_random = Compose(transforms)
        
        
        
    
    def augment(self):
        img = self.image.copy()
        target = self.target.copy()
        
        base_val = self._get_random_base()
        
        if base_val == self.intensity_max:
            base_val = 0
        
        int_range = [x/(self.intensity_max - base_val) for x in self.crop_intensity_range]
        _intensity = random.uniform(*int_range)
        
        if self.merge_by_prod:
            
            img /= base_val
            img = np.power(img, _intensity)
            
        else:
            img -= base_val
            img *= _intensity
        
        assert not np.isnan(img).any()
        
        return self.transforms_random(img, target)
    
    
class MaskedBackground(MaskedBase):
    def __init__(self, 
                 *args, 
                 crop_size = (128, 128),
                 zoom_range = (0.9, 1.1), 
                 **argkws):
        
        super().__init__(*args, **argkws)
        self.random_crop = RandomCrop(crop_size = crop_size)
    
        transforms = [
                AffineTransform(zoom_range = zoom_range, border_mode = cv2.BORDER_REFLECT_101),
                RandomVerticalFlip(),
                RandomHorizontalFlip(),
                AddRandomRectangle(img_shape = crop_size)
                ]
        self.transforms_random = Compose(transforms)
    
    def augment(self):
        img_bgnd = self.image
        crop_bgnd = self.random_crop(img_bgnd, {})[0]
        
        
        
        base_val = self._get_random_base()
        if crop_bgnd.ndim == 2:
            crop_bgnd = crop_bgnd.filled(base_val)
        else:
            crop_bgnd = [crop_bgnd[..., ii].filled(v)[..., None] for ii, v in enumerate(base_val)]
            crop_bgnd = np.concatenate(crop_bgnd, axis=2)
        
        crop_bgnd = self.transforms_random(crop_bgnd, {})[0]
        
        return crop_bgnd

#%%
        
class FluoMergedFlow(Dataset):
    _valid_output_types = [ 'contours', 
                           'bboxes', 
                           'centroids', 
                           'point-inside-contour', 
                           'separated-channels', 
                           'noise2noise', 
                           'segmentation'
                           ]
    
    def __init__(self, 
                 cell_crops = [],
                 bad_crops = [],
                 backgrounds = [],
                         
                 img_size = (128, 128),
                 n_cells_per_crop = 4,
                 n_bgnd_per_crop = None,
                 crop_intensity_range = (0.1, 1.),
                 
                 epoch_size = 1000,
                 
                 int_scale = (0,  255),
                 fg_quantile_range = (0, 10),
                 bg_quantile_range = (25, 75),
                 output_type = 'contours',
                 
                 frac_crop_valid = 0.9,
                 zoom_range = None,
                 rotate_range = None,
                 max_overlap = 1.,
                 
                 
                 null_value = 0,
                 merge_by_prod = False,
                 _debug = False,
                 
                 min_crop_size = None,
                 max_crop_size = None
                 ):
        
        
        _dum = set(dir(self))
        self.img_size = img_size
        
        self.n_cells_per_crop = self.num2range(n_cells_per_crop)
        if n_bgnd_per_crop is None:
            self.n_bgnd_per_crop = self.n_cells_per_crop
        else:
            self.n_bgnd_per_crop = self.num2range(n_bgnd_per_crop)
        
        self.crop_intensity_range = crop_intensity_range
        self.frac_crop_valid = frac_crop_valid
        
        self.epoch_size = epoch_size
        
        self.int_scale = int_scale #range how the images will be scaled
        
        self.bg_quantile_range = bg_quantile_range
        self.fg_quantile_range = fg_quantile_range
        
        
        
        
        self.output_type = output_type
        
        
        self.zoom_range = zoom_range
        self.rotate_range = rotate_range
        self.max_overlap = max_overlap
        
        self.null_value = null_value
        self.merge_by_prod = merge_by_prod
        
        self.min_crop_size = 1 if min_crop_size is None else min_crop_size
        
        
#        if max_crop_size is None:
#            max(self.img_size)*(1 + )
#        else:
#            self.max_crop_size =  max_crop_size
        
        self._input_names = list(set(dir(self)) - _dum) #i want the name of this fields so i can access them if necessary
        
        
        
        fg_args = dict(
                   q_range = self.fg_quantile_range, 
                   int_scale = self.int_scale, 
                   null_value = self.null_value,  
                   merge_by_prod = self.merge_by_prod,
                   crop_intensity_range = self.crop_intensity_range,
                   
                   zoom_range = self.zoom_range,
                   min_crop_size = self.min_crop_size,
                   #max_crop_size = self.max_crop_size,
                   )
        
        bg_args = dict(
                   q_range = self.bg_quantile_range, 
                   int_scale = self.int_scale, 
                   null_value = self.null_value,  
                   merge_by_prod = self.merge_by_prod,
                   
                   crop_size = self.img_size,
                   zoom_range = self.zoom_range
                   )
        
        
        
        
        self.cell_crops = self._prepare_data(cell_crops, MaskedCrop, **fg_args)
        self.bad_crops = self._prepare_data(bad_crops, MaskedCrop, **fg_args)
        self.backgrounds = self._prepare_data(backgrounds, MaskedBackground, **bg_args)
        
        
        cnt_output_type = output_type.split('+')
        for c in cnt_output_type:
            assert c in self._valid_output_types, f'Invalid output_type `{c}`. Valid types are : `{self._valid_output_types}`'
        try:
            cnt_output_type.remove('noise2noise')
            self.is_return_n2n = True
        except ValueError:
            self.is_return_n2n = False
        self.contour_outputs = Contours2Outputs(cnt_output_type, channel_lasts = False)
        
        transforms = [
                RandomBlur(), 
                AddGaussNoise(), 
                ConvertToPoissonNoise(int_scale = self.int_scale)
                ]
        self.apply_noise = Compose(transforms)
        self.random_crop = RandomCrop(img_size)
        
    
    @property
    def input_parameters(self):
        return {x:getattr(self, x) for x in self._input_names}
    
    @staticmethod
    def num2range(x):
        if isinstance(x, (list, tuple)):
            assert len(x) == 2
            return x
        else:
            return (0, x)
    
    def _prepare_data(self, data, obj, *args, **argkws):
        if isinstance(data, dict):
            data_prepared = {}
            for k, val in data.items():
                data_prepared[k] = self._prepare_data(val, obj, *args, **argkws)
        elif isinstance(data, (list, tuple)):
            data_prepared = [obj(x, *args, **argkws) for x in data]
        else:
            raise TypeError(type(data))
        
        return data_prepared
    
    def _get_n2n_complement(self, fngd_p1, overlap_tracker):
        bgnd1_p2 = self._get_random_bgnd()
        bgnd2_p2 = self.get_cell_pairs(self.bad_crops, self.n_bgnd_per_crop, overlap_tracker)[0]
        out2 = self._merge(fngd_p1, bgnd1_p2, bgnd2_p2)
        return out2
    
    
         
    def _get_random_bgnd(self):
        if not self.backgrounds:
            return np.full(self.img_size, self.null_value , np.float32)
        bgnd_data = random.choice( self.backgrounds)
        crop_bgnd = bgnd_data.augment()
        return crop_bgnd
    
    def _sample(self):
        fngd_p1, contours, overlap_tracker = self.get_cell_pairs(self.cell_crops, self.n_cells_per_crop)
        bgnd1_p1 = self._get_random_bgnd()
        bgnd2_p1 = self.get_cell_pairs(self.bad_crops, self.n_bgnd_per_crop, overlap_tracker)[0]
        
        
        out1 = self._merge(fngd_p1, bgnd1_p1, bgnd2_p1)
        if out1.ndim == 2:
            out1 = out1[None]
        
        out1, out2 = self.contour_outputs(out1, {'contours': contours})
        if self.is_return_n2n:
            out2['n2n_target'] = self._get_n2n_complement(fngd_p1, overlap_tracker)
        
        out1, out2 = self.apply_noise(out1, out2)
        return out1, out2
        
        
    def _merge(self, fgnd, bgnd1, bgnd2):
        
        if self.merge_by_prod:
            bgnd1 = np.clip(bgnd1, 0, 1)
            bgnd2 = np.clip(bgnd2, 0, 1)
            fgnd = np.clip(fgnd, 0, 1)
            _out = bgnd1*bgnd2*fgnd
            
        else:
            _out = bgnd1 + bgnd2 + fgnd
            _out = np.clip(_out, 0, 1)
        
        
        #TODO there shouldn't be any nan here, but it some situation it does happens... this is a quick fix
        _out[np.isnan(_out)] = 1.
        
        return _out
            
    
    def get_cell_pairs(self, crops, n_images_range, overlap_tracker = None):
        n_rois = random.randint(*n_images_range)
        raw_cell_imgs = self._read_random_imgs(crops, n_rois)
        synthetic_image, contours, overlap_tracker = self.synthetize_image(raw_cell_imgs, overlap_tracker)
        
        synthetic_image = self._adjust_channels(synthetic_image)
        
        
        return synthetic_image, contours, overlap_tracker
        
    def _adjust_channels(self, x):
        if x.ndim == 2:
            return x[None]
        else:
            return np.rollaxis(x, 2, 0)
            
    
    
    def _read_random_imgs(self, crops, n_crops):
        if isinstance(crops, dict):
            crops = random.choice(list(crops.values()))
        
        random_images = []
        if not crops:
            return random_images
        
        for _ in range(n_crops):
            crop_data = random.choice(crops)
            
            random_images.append(crop_data)
            
        
        return random_images
    
    
    def synthetize_image(self, raw_cell_imgs, overlap_tracker = None):
        
        if overlap_tracker is None:
            overlap_tracker = OverlapsTracker(self.img_size[:2], self.max_overlap, self.null_value)
        
        
        #synthetize image
        synthetic_image = np.full(self.img_size, self.null_value, dtype = np.float32)
        contours_out = []
        for imgs_data in raw_cell_imgs:
            img, target = imgs_data.augment()
            _out = self._random_locate_imgs(img, target, overlap_tracker)
            
            if not _out:
                continue
            
            (xi,yi), (crop, contours), overlap_mask = _out
            
            if self.merge_by_prod:
                synthetic_image[xi:xi+crop.shape[0], yi:yi+crop.shape[1]] *= crop
            else:
                synthetic_image[xi:xi+crop.shape[0], yi:yi+crop.shape[1]] += crop
                
            contours_out +=  contours
        
        
          
        return synthetic_image, contours_out, overlap_tracker
    
    def _random_locate_imgs(self, crop, target, overlap_tracker):
        #randomly located a pair in the final image. If part of the pair is 
        #located outside the final image, the the pair is cropped.        
        
        crop_shape = crop.shape[:2]
        
        #crop if the x,y coordinate is outside the expected image 
        frac_cc = [int(round(x*self.frac_crop_valid)) for x in crop_shape]
        
        
        
        if self.img_size[0] > crop_shape[0] :
            max_ind_x = self.img_size[0] - crop_shape[0] 
            xlims = -frac_cc[0], max_ind_x + frac_cc[0] + 1 #randint upper bound is not inclusive
        else:
            max_ind_x = 0
            xlims = (0, 1)
        
        if self.img_size[1] > crop_shape[1] :
            max_ind_y = self.img_size[1] - crop_shape[1]
            ylims = -frac_cc[1], max_ind_y + frac_cc[1] + 1
        else:
            max_ind_y = 0
            ylims = (0, 1)
        
        if max_ind_x == 0 or max_ind_y == 0:
            crop, target = self.random_crop(crop, target)
            
        
        
        roi_lims = (0, self.img_size[1] - 1, 0, self.img_size[0] - 1)
        BREAK_NUM = 3 #number of times to try to add an image before giving up
        for _ in range(BREAK_NUM):
            xi = random.randint(*xlims)
            yi = random.randint(*ylims)
            
            cnt_corner = np.array((yi, xi))[None]
            
            contours = [crop_contour(x + cnt_corner, *roi_lims) for x in target['contours']]
            #contours = [x + cnt_corner for x in target['contours']]
            contours = [x for x in contours if x.size]
            
            if xi < 0:
                crop = crop[abs(xi):]
                xi = 0
            
            if yi < 0:
                crop = crop[:, abs(yi):]
                yi = 0
            
            
            if xi + crop.shape[0] > self.img_size[0]:
                ii = self.img_size[0] - xi
                crop = crop[:ii]
                
                assert xi + crop.shape[0] == self.img_size[0]
            
            if yi + crop.shape[1] > self.img_size[1]:
                ii = self.img_size[1] - yi
                crop = crop[:, :ii]
                
                assert yi + crop.shape[1] == self.img_size[1]
                
            
            if not overlap_tracker.add(xi, yi, crop):
                continue
            
            return (xi, yi), (crop, contours), overlap_tracker
        else:
            return
    
    
    def __len__(self):
        return self.epoch_size
    
    
    def __getitem__(self, ind):
        
        return self._sample()
        #return [(x[None], t) if x.ndim == 2 else (x, t) for x, t in self._sample()]
    
    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n <= len(self):
             
            self.n += 1
            return self[self.n-1]
            
        else:
            raise StopIteration

#dummy flow use for compatibility with otherfunctions for testing
class FluoMergedFlowTest(FluoMergedFlow):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.num_test_samples = num_test_samples
        #num_test_samples = 200,
        
    @property
    def data_indexes(self):
        return [None]*self.epoch_size
    
    def read_full(self, ind):
        return ToTensor()(*self._sample())



def load_data(src_file, min_size = 2, crops_same_image = False, ignore_bad_crops = False):
    #%%
    with open(src_file, 'rb') as fid:
        data = pickle.load(fid)
        

    if crops_same_image:
        cell_crops = {}
        for cc in data['cell_crops']:
            _key = cc['img_id']
            if not _key in cell_crops:
                cell_crops[_key] = []
            cell_crops[_key].append(cc)
    else:
        cell_crops = [x for x in data['cell_crops'] if min(x['image'].shape)>=min_size]
    
    if ignore_bad_crops:
        bad_crops = []
    else:
        bad_crops = [x for x in data['bad_crops'] if min(x['image'].shape)>=min_size]
        
        
    backgrounds = data['backgrounds']
    #%%
    return cell_crops, bad_crops, backgrounds

def flow_BBBC38_crops():
    #%%
    #src_file = '/Users/avelinojaver/Desktop/nuclei_datasets/crops/BBBC038_Kaggle_2018_Data_Science_Bowl.p'
    #src_file = Path.home() / 'workspace/localization/data/crops4synthetic/BBBC038_Kaggle_2018_Data_Science_Bowl.p'
    src_file = '/Users/avelinojaver/Desktop/nuclei_datasets/crops/BBBC038_Kaggle_2018_Data_Science_Bowl_v2.p'
    src_file = Path(src_file)
    
    cell_crops, bad_crops, backgrounds = load_data(src_file, min_size = 2, crops_same_image = False, ignore_bad_crops = False)
    
    #%%
    gen = FluoMergedFlowTest(cell_crops = cell_crops,
                         bad_crops = bad_crops,
                         backgrounds = backgrounds,
                         
                         output_type =  'noise2noise+segmentation',#'centroids+noise2noise+segmentation',#'centroids+noise2noise',#'segmentation+noise2noise',#'point-inside-contour',#'centroids',#
                         
                         img_size = (128, 128),
                         int_scale = (0, 255),
                         
                         n_cells_per_crop = (15, 15),
                         n_bgnd_per_crop = (0, 0),
                         
                         crop_intensity_range = (0.2, 1.),
                         fg_quantile_range = (0, 10),
                         bg_quantile_range = (25, 75),
                         
                         frac_crop_valid = 0.1,
                         zoom_range = (0.9, 1.1),
                         rotate_range = (0, 90),
                         max_overlap = 0.4,
                         
                         null_value = 0.,
                         merge_by_prod = False,
                         max_crop_size = 128
                 
                         )  
    return gen


if __name__ == '__main__':
    from pathlib import Path
    from torch.utils.data import DataLoader
    import matplotlib.pylab as plt
    import matplotlib.patches as patches
    import tqdm

    #%%
    gen = flow_BBBC38_crops()
    #gen = _test_load_BBBC42(True)
    #gen = _test_load_BBBC26(True)
    #gen = _test_load_microglia(True)
    #%%
    #for _ in tqdm.tqdm(range(10000)):
    #   xin, xout = gen[0]
   
   
    for _ in tqdm.tqdm(range(10)):
        xin, xout = gen._sample()
        
        n_per_channel = xin.shape[0]
        
        if gen.output_type == 'separated-channels':
            fig, axs = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(15, 5))
            xout = [xout[ii:ii+n_per_channel] for ii in range(0, 3*n_per_channel, n_per_channel)]
            for ax, x in zip(axs, [xin] + xout):
                if n_per_channel == 1:
                    x = x[0]
                else:
                    x = np.rollaxis(x, 0, 3)
                
        
        n_plots = 3 if 'n2n_target' in xout else 2
            
        fig, axs = plt.subplots(1, n_plots, sharex=True, sharey=True)
        for ax in axs[:2]:
            ax.imshow(xin[0], cmap='gray', vmin = 0, vmax = 1)
        
        if 'segmentation_mask' in xout:
            axs[1].imshow(xout['segmentation_mask'])
           
        
        if 'n2n_target' in xout:
            axs[2].imshow(xout['n2n_target'][0], cmap='gray', vmin = 0, vmax = 1)
            
        if 'contours' in xout:
            for cc in xout['contours']:
                axs[1].plot(cc[:, 0], cc[:, 1], 'r')
        
        if 'bboxes' in xout:
            for cc in xout['bboxes']:
                x, y, w, h = cc[0], cc[1], cc[2] - cc[0], cc[3] - cc[1]
                rect = patches.Rectangle((x, y), w, h,linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
                
        
        if 'coordinates' in xout:
            cc = xout['coordinates']
            axs[1].plot(cc[:, 0], cc[:, 1], 'r.')
                
        
                
                
    #%%
    
    