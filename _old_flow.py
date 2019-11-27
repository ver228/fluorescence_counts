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
from cell_localization.flow import ToTensor

def rotate_bound(image, cnt, angle, border_value = 0):
    #https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    if image.ndim == 3:
        border_value = 3*[border_value]
    
    
    cnt_rot = np.dot(M[:2, :2], cnt.T)  +  M[:, -1][:, None]
    #cnt_rot = np.round(cnt_rot)
    
    image_rot = cv2.warpAffine(image, M, (nW, nH),
                          borderValue = border_value)
    
    # perform the actual rotation and return the image
    return image_rot, cnt_rot.T

#%%
   
        
class FluoMergedFlow(Dataset):
    _valid_output_types = [ 'contours', 
                           'bboxes', 
                           'centroids', 
                           'point-inside-contour', 
                           'separated-channels', 
                           'noise2noise', 
                           'centroids+noise2noise',
                           'segmentation',
                           'segmentation+noise2noise'
                           ]
    
    def __init__(self, 
                 cell_crops = [],
                 bad_crops = [],
                 backgrounds = [],
                         
                 img_size = (128, 128),
                 n_cells_per_crop = 4,
                 n_bgnd_per_crop = None,
                 crop_intensity_range = (0.025, 1.),
                 
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
        
        
        assert output_type in self._valid_output_types, f'Invalid output_type `{output_type}`'
        
        self.output_type = output_type
        
        
        self.zoom_range = zoom_range
        self.rotate_range = rotate_range
        self.max_overlap = max_overlap
        
        self.null_value = null_value
        self.merge_by_prod = merge_by_prod
        
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        
        self._input_names = list(set(dir(self)) - _dum) #i want the name of this fields so i can access them if necessary
        
        
        self.cell_crops = self._prepare_data(cell_crops, self.fg_quantile_range)
        self.bad_crops = self._prepare_data(bad_crops, self.fg_quantile_range)
        self.backgrounds = self._prepare_data(backgrounds, self.bg_quantile_range)
        
    
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
            
    
    def _prepare_data(self, data, q_range):
        if isinstance(data, dict):
            data_prepared = {}
            for k, val in data.items():
                data_prepared[k] = self._prepare_data(val, q_range)
        else:
            data_prepared = [self._prepare_crop(x, q_range) for x in data]
        return data_prepared
    
    def _prepare_crop(self, data, q_range):
        assert len(q_range) == 2
        q_range = (*q_range, 50)
        
        img = data['image']
        
        img = self._scale(img)
        
        img = np.ma.masked_equal(img, self.null_value)
        
        if img.ndim == 2:
            base_range = np.percentile(img.compressed(), q_range)
            
            
            q_med = base_range[-1]
            base_range = base_range[:2]
            
        elif img.ndim == 3:
            base_range = [np.percentile(img[..., i].compressed(), q_range) for i in range(3)]
            q_med = [x[-1] for x in base_range]
            base_range = [x[:2] for x in base_range]
                
        else:
            raise ValueError(f'Invalid mumber of dimensions `{img.ndim}` in image')
        
        data['image'] = img
        data['intensity_base_range'] = base_range
        data['intensity_median'] = q_med
        return data
    
    
    
    
    def _get_random_bgnd(self):
        if not self.backgrounds:
            return np.full(self.img_size, self.null_value , np.float32)
        
        bgnd_data = random.choice( self.backgrounds)
        
        img_bgnd = bgnd_data['image']
        base_range = bgnd_data['intensity_base_range']
        
        xi = random.randint(0, img_bgnd.shape[0] - self.img_size[0])
        yi = random.randint(0, img_bgnd.shape[1] - self.img_size[1])
        crop_bgnd = img_bgnd[xi:xi + self.img_size[0], yi:yi + self.img_size[1]]
        
        base_val = self._get_random_base(base_range)
        
        if crop_bgnd.ndim == 2:
            crop_bgnd = crop_bgnd.filled(base_val)
        else:
            crop_bgnd = [crop_bgnd[..., ii].filled(v)[..., None] for ii, v in enumerate(base_val)]
            crop_bgnd = np.concatenate(crop_bgnd, axis=2)
           
        if self.rotate_range:
            angle = random.uniform(*self.rotate_range)
            (cX, cY) = (crop_bgnd.shape[0] // 2, crop_bgnd.shape[1] // 2)
            M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
            
            crop_bgnd = cv2.warpAffine(crop_bgnd, 
                                       M, 
                                       crop_bgnd.shape[:2], 
                                       borderMode = cv2.BORDER_REFLECT_101
                                       )

        #random flips
        if random.random() >= 0.5:
            crop_bgnd = crop_bgnd[::-1]
        if random.random() >= 0.5:
            crop_bgnd = crop_bgnd[:, ::-1]
        
        if random.random() >= 0.5:
            assert not self.merge_by_prod
            shape_noise = self._get_shape_noise(crop_bgnd.shape)
            crop_bgnd += shape_noise
            
            
        
        
        crop_bgnd = self._adjust_channels(crop_bgnd)
        return crop_bgnd
    
    def _get_shape_noise(self, img_shape):
        _rr = [(x//4, x) for x in img_shape]
        
        box_size  = [random.randint(*x) for x in _rr]
        corner = [random.randint(0, s-1) for s in img_shape]
        angle = random.randint(-180, 180)
        int_level = random.uniform(0.1, 0.5)
        
        box = cv2.boxPoints((corner, box_size, angle))
        box = np.int0(box)
        
        shape_noise = np.zeros(img_shape, np.float32)
        cv2.drawContours(shape_noise, [box], 0, int_level, -1)
        
        shape_noise = cv2.GaussianBlur(shape_noise, (25, 25), 0)
        return shape_noise
    
    
    def _scale(self, x):
        x[x<self.int_scale[0]] = self.int_scale[0]
        x = (x-self.int_scale[0])/(self.int_scale[1]-self.int_scale[0])
        
        return x.astype(np.float32)
    
    @staticmethod
    def _contour2centroids(contours):
        cms = [np.mean(x, axis=0) for x in contours]
        
        if cms:
            coordinates = np.array(cms, dtype = np.int)
        else:
            coordinates = np.zeros((0, 2), dtype = np.int)
        
        out2 = dict(
                coordinates = coordinates,
                labels = np.ones(coordinates.shape[0])
                )
        return out2
    
    @staticmethod
    def _contour2bboxes(contours):
        bboxes = [(x[:, 0].min(), x[:, 1].min(), x[:, 0].max(), x[:, 1].max()) for x in contours]
        
        if bboxes:
            bboxes = np.array(bboxes, dtype = np.float32)
        else:
            bboxes = np.zeros((0, 4), dtype = np.int)
        out2 = dict(bboxes = bboxes, 
                    labels = np.ones(bboxes.shape[0])
                    )
        return out2
    
    
    def _get_n2n_complement(self, fngd_p1, overlap_tracker):
        bgnd1_p2 = self._get_random_bgnd()
        bgnd2_p2 = self.get_cell_pairs(self.bad_crops, self.n_bgnd_per_crop, overlap_tracker)[0]
        out2 = self._merge(fngd_p1, bgnd1_p2, bgnd2_p2)
        return out2
        
    @staticmethod
    def _contour2pointsinside(contours):
        #I am assuming that most of the time the points will be inside the contour. 
        #I think this is faster than getting all the points by drawing the contour
        points = [] 
        for cnt in contours:
            cnt = cnt.round().astype(np.int)
            xlims = cnt[:, 0].min(), cnt[:, 0].max()
            ylims = cnt[:, 1].min(), cnt[:, 1].max()
            for _ in range(100):
                x = random.randint(*xlims)
                y = random.randint(*ylims)
                
                is_inside = cv2.pointPolygonTest(cnt, (x, y), False)
                if is_inside == 1:
                    break
            else:
                x, y = np.mean(cnt, axis=0)
        
            points.append((x,y))
                
        if points:
            coordinates = np.array(points, dtype = np.int)
        else:
            coordinates = np.zeros((0, 2), dtype = np.int)
        out2 = dict(
                coordinates = coordinates,
                labels = np.ones(coordinates.shape[0])
                )
        return out2
    @staticmethod
    def _contour2segmask(contours, img_shape):
        contours_i = [np.floor(x).astype(np.int) for x in contours]
        
        seg_mask = np.zeros(img_shape, dtype = np.uint8)
        #cv2.drawContours(seg_mask, contours_i, -1, 1, -1)
        
        contours_i = sorted(contours_i, key = cv2.contourArea)
        for cnt in contours_i:
            cv2.drawContours(seg_mask, [cnt], 0, 1, -1)
            
        for cnt in contours_i:
            rect = cv2.minAreaRect(cnt)
            max_length = min(rect[1])
            
            thickness = max(1, int(np.log10(max_length + 1) + 1) )
            
            cv2.drawContours(seg_mask, [cnt], 0, 2, thickness)
        
        return dict(segmentation_mask = seg_mask.astype(np.int))
        
    def _sample(self):
        fngd_p1, contours, overlap_tracker = self.get_cell_pairs(self.cell_crops, self.n_cells_per_crop)
        bgnd1_p1 = self._get_random_bgnd()
        bgnd2_p1 = self.get_cell_pairs(self.bad_crops, self.n_bgnd_per_crop, overlap_tracker)[0]
        
        
        out1 = self._merge(fngd_p1, bgnd1_p1, bgnd2_p1)
        if out1.ndim == 2:
            out1 = out1[None]
        
        if self.output_type == 'contours':
            out2 = dict(contours = contours)
            
        elif self.output_type == 'bboxes':
            out2 = self._contour2bboxes(contours)
            
        elif self.output_type == 'centroids':
            out2 = self._contour2centroids(contours)
        
            
        elif self.output_type == 'point-inside-contour':
            out2 = self._contour2pointsinside(contours)
            
        
        elif self.output_type == 'separated-channels':
            out2 = [fngd_p1, bgnd1_p1, bgnd2_p1]
            out2 = np.concatenate(out2)
        
        elif self.output_type == 'noise2noise':
            out2 = self._get_n2n_complement(fngd_p1, overlap_tracker)
        
        elif self.output_type == 'segmentation':
            out2 = self._contour2segmask(contours, self.img_size)
            
        elif self.output_type == 'centroids+noise2noise':
            out2 = self._contour2centroids(contours)
            out2['n2n_target'] = self._get_n2n_complement(fngd_p1, overlap_tracker)
        
        elif self.output_type == 'segmentation+noise2noise':
            out2 = self._contour2segmask(contours, self.img_size)
            out2['n2n_target'] = self._get_n2n_complement(fngd_p1, overlap_tracker)
            
           
        out1, n2n_target = self.apply_noise(out1, out2)
        if n2n_target is not None:
            out2['n2n_target'] = n2n_target
        
        return out1, out2
        
        
    def apply_noise(self, out1, out2):
        rand_blur, rand_noise = random.random(), random.random()
        gauss_scale = np.random.uniform(0.01, 0.1)
        _scale = self.int_scale[1] - self.int_scale[0]
        
        def _apply_noise(img):
            if rand_blur < rand_blur:
                img = cv2.GaussianBlur(img, (11, 11), 0)
            
            if rand_noise < 0.33:
                img += np.random.normal(0, gauss_scale, size = img.shape).astype(np.float32)
                
                
            elif rand_noise < 0.67:
                #replace data with poisson noise...
                
                img = np.random.poisson(img*_scale).astype(np.float32)/_scale
            
            return img
        
        
        out1 = _apply_noise(out1)     
        n2n_target = _apply_noise(out2['n2n_target']) if 'n2n_target' in out2 else None
        
        return out1, n2n_target
    
    
    def _merge(self, fgnd, bgnd1, bgnd2):
        
        if self.merge_by_prod:
            bgnd1 = np.clip(bgnd1, 0, 1)
            bgnd2 = np.clip(bgnd2, 0, 1)
            fgnd = np.clip(fgnd, 0, 1)
            _out = bgnd1*bgnd2*fgnd
            
        else:
            _out = bgnd1 + bgnd2 + fgnd
            _out = np.clip(_out, 0, 1)
            
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
            
    
    
    def synthetize_image(self, raw_cell_imgs, overlap_tracker = None):
        p_cell_imgs = []
        
        if overlap_tracker is None:
            overlap_tracker = OverlapsTracker(self.img_size[:2], self.max_overlap, self.null_value)
        
        
        
        for imgs_data in raw_cell_imgs:
            imgs_data = self._random_augment_imgs(imgs_data)
            _out = self._random_locate_imgs(imgs_data, overlap_tracker)
            
            if not _out:
                continue
            
            coords, located_imgs, overlap_mask = _out
            
            
            p_cell_imgs.append((coords, located_imgs))
        
        #synthetize images form the pairs
        synthetic_image, contours = self._cellcrops2image(p_cell_imgs)         
           
        return synthetic_image, contours, overlap_tracker
    
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
    
    
    @staticmethod
    def _get_random_base(base_range):
        
        if len(base_range) == 2:
            return random.uniform(*base_range)
        else:
            #in three dimensions i want a random value along the ranges for each color
            val = [random.uniform(*x) for x in base_range]
            return val
    
    
    
    def _random_augment_imgs(self, imgs_data):
        '''
        Randomly exectute the same transforms for each tuple the list.
        '''
        
        
        img =  imgs_data['image'].astype(np.float32).copy() # If it is already a float32 it will not return a copy...
        cnt = imgs_data['contour'].astype(np.float32).copy()
        base_range = imgs_data['intensity_base_range'] 
        med_intensity = imgs_data['intensity_median']
        
        #obtain parameters...
        base_val = self._get_random_base(base_range)
        
        if img.ndim == 3: #deal with the case of three dimensions where each color channel will have its own value
            base_val = np.array(base_val)[None, None]
        
        _flipv = random.random() >= 0.5
        _fliph = random.random() >= 0.5
        
        _zoom = None
        if self.zoom_range:
            _zoom = random.uniform(*self.zoom_range)
            if self.max_crop_size is not None:
                _max_zoom = self.max_crop_size/max(img.shape)
                _zoom = min(_max_zoom,_zoom)
                
            if self.min_crop_size is not None:
                _min_zoom = self.min_crop_size/max(img.shape)
                _zoom = max(_min_zoom,_zoom)
            
        
        _angle = random.uniform(*self.rotate_range) if self.rotate_range else None
        
        
        int_range = [x/med_intensity for x in self.crop_intensity_range]
        _intensity = random.uniform(*int_range)
        
        if self.merge_by_prod:
            
            img /= base_val
            img = np.power(img, _intensity)
            
        else:
            img -= base_val
            img *= _intensity
        
        #random rotation
        if _angle:
            img, cnt = rotate_bound(img, cnt, _angle, border_value = self.null_value)
            
        
        #random flips
        if _fliph:
            img = img[::-1]
            cnt[:, 1] = (img.shape[0] - 1 - cnt[:, 1])
            
            
        if _flipv:
            img = img[:, ::-1]
            cnt[:, 0] = (img.shape[1] - 1 - cnt[:, 0])
        
        #zoom
        if _zoom:
            img = cv2.resize(img, (0,0), fx=_zoom, fy=_zoom, interpolation = cv2.INTER_LINEAR)
            cnt *= _zoom
        
        #crop in case it is larger than the final image
        img = img[:self.img_size[0], :self.img_size[1]] 
        
        
        val = self.img_size[0] - 1
        bad = cnt[:, 1] > val
        self._clip_to_crop(cnt[:, 1], cnt[:, 0], bad, val)
        
        val = self.img_size[1] - 1
        bad = cnt[:, 0] > val
        self._clip_to_crop(cnt[:, 0], cnt[:, 1], bad, val)
        
        
        img = np.clip(img, 0, 1) # here I am assuming the image was originally scaled from 0 to 1
        
        
        new_data = {}
        new_data['image'] = img
        new_data['contour'] = cnt
        
        return new_data
        
    @staticmethod
    def _clip_to_crop(p1, p2, bad, val):
            p1[bad] = val
            vv = p2[~bad]
            if not vv.size:
                return np.zeros((2, 0))
            
            
            p2[bad] = np.clip(p2[bad], vv.min(), vv.max())
            return p1, p2
    
    
    def _random_locate_imgs(self, _imgs_data, overlap_tracker):
        #randomly located a pair in the final image. If part of the pair is 
        #located outside the final image, the the pair is cropped.        
        
        crop_shape = _imgs_data['image'].shape
        
        #crop if the x,y coordinate is outside the expected image 
        frac_cc = [int(round(x*self.frac_crop_valid)) for x in crop_shape]
        
        
        max_ind_x = self.img_size[0] - crop_shape[0]
        max_ind_y = self.img_size[1] - crop_shape[1]
        
        
        BREAK_NUM = 3 #number of times to try to add an image before giving up
        
        new_data = dict(
                image = _imgs_data['image'],
                contour = _imgs_data['contour']
                )
        
        
        for _ in range(BREAK_NUM):
            xi = random.randint(-frac_cc[0], max_ind_x + frac_cc[0])
            yi = random.randint(-frac_cc[1], max_ind_y + frac_cc[1])
            
            if xi < 0:
                new_data['image'] = new_data['image'][abs(xi):]
                
                if not new_data['image'].size:
                    cc = np.zeros((0, 2))
                else:
                    cc = new_data['contour']
                    cc[:, 1] += xi
                    bad = cc[:, 1] < 0
                    self._clip_to_crop(cc[:, 1], cc[:, 0], bad, 0)
                
                xi = 0
            
            if yi < 0:
                new_data['image'] = new_data['image'][:, abs(yi):]
                
                if not new_data['image'].size:
                    cc = np.zeros((0, 2))
                else:
                    cc = new_data['contour']
                    cc[:, 0] += yi
                    bad = cc[:, 0] < 0
                    self._clip_to_crop(cc[:, 0], cc[:, 1], bad, 0)
                    
                new_data['contour'] = cc
                yi = 0
            
            
            if xi > max_ind_x:
                ii = max_ind_x-xi
                new_data['image'] = new_data['image'][:ii]
                
                if not new_data['image'].size:
                    cc = np.zeros((0, 2))
                else:
                    cc = new_data['contour']
                    val = new_data['image'].shape[0] -1
                    bad = cc[:, 1] > val
                    self._clip_to_crop(cc[:, 1], cc[:, 0], bad, val)
                
                new_data['contour'] = cc
                
            
            if yi > max_ind_y:
                ii = max_ind_y - yi
                
                new_data['image'] = new_data['image'][:, :ii] 
                
                if not new_data['image'].size:
                    cc = np.zeros((0, 2))
                else:
                    cc = new_data['contour']
                    val = new_data['image'].shape[1] -1
                    bad = cc[:, 0] > val
                    self._clip_to_crop(cc[:, 0], cc[:, 1], bad, val)
                
                new_data['contour'] = cc
                

            
            if not overlap_tracker.add(xi, yi, new_data['image']):
                continue
            
            return (xi, yi), new_data, overlap_tracker
        else:
            return
    
    
    
    def _cellcrops2image(self, cells_data):
        
        image_out = np.full(self.img_size, self.null_value, dtype = np.float32)
        contours_out = []
        for (xi,yi), cc in cells_data:
            crop = cc['image']
            
            if self.merge_by_prod:
                image_out[xi:xi+crop.shape[0], yi:yi+crop.shape[1]] *= crop
            else:
                image_out[xi:xi+crop.shape[0], yi:yi+crop.shape[1]] += crop
                
                
            cnt = cc['contour']
            cnt = np.array(((cnt[:, 0] + yi, cnt[:, 1] + xi))).T
            contours_out.append(cnt)
        
        return image_out, contours_out
        
    
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



def load_data(src_file, min_size = 2, crops_same_image = True, ignore_bad_crops = True):
    
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
    
    return cell_crops, bad_crops, backgrounds

def flow_BBBC38_crops():
    #%%
    #src_file = '/Users/avelinojaver/Desktop/nuclei_datasets/crops/BBBC038_Kaggle_2018_Data_Science_Bowl.p'
    #src_file = Path.home() / 'workspace/localization/data/crops4synthetic/BBBC038_Kaggle_2018_Data_Science_Bowl.p'
    src_file = '/Users/avelinojaver/Desktop/nuclei_datasets/crops/BBBC038_Kaggle_2018_Data_Science_Bowl_v2.p'
    src_file = Path(src_file)
    
    cell_crops, bad_crops, backgrounds = load_data(src_file, min_size = 2, crops_same_image = True)
    
    #%%
    gen = FluoMergedFlowTest(cell_crops = cell_crops,
                         bad_crops = bad_crops,
                         backgrounds = backgrounds,
                         
                         output_type =  'segmentation+noise2noise',#'point-inside-contour',#'centroids',#'centroids+noise2noise',
                         
                         img_size = (128, 128),
                         int_scale = (0, 255),
                         
                         n_cells_per_crop = (0, 15),
                         n_bgnd_per_crop = (0, 15),
                         
                         crop_intensity_range = (0.025, 0.5),
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
                
                ax.imshow(x)
        elif gen.output_type == 'noise2noise':
            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
            axs[0].imshow(xin[0])
            axs[1].imshow(xout[0])
            
        elif gen.output_type == 'segmentation':
            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
            axs[0].imshow(xin[0], cmap='gray', vmin = 0, vmax = 1)
            axs[1].imshow(xout['segmentation_mask'])
            
        elif gen.output_type == 'segmentation+noise2noise':
            fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
            axs[0].imshow(xin[0], cmap='gray', vmin = 0, vmax = 1)
            axs[1].imshow(xout['segmentation_mask'])
            axs[2].imshow(xout['n2n_target'][0], cmap='gray', vmin = 0, vmax = 1)
        
        
        
        elif  gen.output_type == 'centroids+noise2noise':
            fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
            axs[0].imshow(xin[0], cmap='gray', vmin = 0, vmax = 1)
            axs[1].imshow(xin[0], cmap='gray', vmin = 0, vmax = 1)
            cc = xout['coordinates']
            axs[1].plot(cc[:, 0], cc[:, 1], 'r.')
            axs[2].imshow(xout['n2n_target'][0], cmap='gray', vmin = 0, vmax = 1)
        
        else:
            
            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
            for ax in axs:
                ax.imshow(xin[0], cmap='gray', vmin = 0, vmax = 1)
            
            
            contours = xout['contours']
            img_shape = xin[0].shape
            
            contours_i = [np.floor(x).astype(np.int) for x in contours]
            
            seg_mask = np.zeros(img_shape, dtype = np.uint8)
            #cv2.drawContours(seg_mask, contours_i, -1, 1, -1)
            
            contours_i = sorted(contours_i, key = cv2.contourArea)
            for cnt in contours_i:
                cv2.drawContours(seg_mask, [cnt], 0, 1, -1)
                
            for cnt in contours_i:
                rect = cv2.minAreaRect(cnt)
                max_length = max(rect[1])
                
                thickness = max(1, int(np.log10(max_length) + 1) )
                
                cv2.drawContours(seg_mask, [cnt], 0, 2, thickness)
            
            if gen.output_type == 'contours':
                for cc in xout['contours']:
                    axs[1].plot(cc[:, 0], cc[:, 1], 'r')
            elif gen.output_type == 'bboxes':
                for cc in xout['bboxes']:
                    x, y, w, h = cc[0], cc[1], cc[2] - cc[0], cc[3] - cc[1]
                    rect = patches.Rectangle((x, y), w, h,linewidth=1,edgecolor='r',facecolor='none')
                    ax.add_patch(rect)
            elif (gen.output_type == 'centroids') or (gen.output_type == 'point-inside-contour'):
                cc = xout['coordinates']
                axs[1].plot(cc[:, 0], cc[:, 1], 'r.')
                
            
                
                
    #%%
    
    