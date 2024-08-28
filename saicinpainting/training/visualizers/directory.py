import os

import cv2
import numpy as np

from saicinpainting.training.visualizers.base import BaseVisualizer, visualize_mask_and_images_batch
from saicinpainting.utils import check_and_warn_input_range
import pdb


class DirectoryVisualizer(BaseVisualizer):
    DEFAULT_KEY_ORDER = 'image predicted_image inpainted'.split(' ')

    def __init__(self, outdir, key_order=DEFAULT_KEY_ORDER, max_items_in_batch=10,
                 last_without_mask=True, rescale_keys=None):
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.key_order = key_order
        self.max_items_in_batch = max_items_in_batch
        self.last_without_mask = last_without_mask
        self.rescale_keys = rescale_keys

    def __call__(self, epoch_i, batch_i, batch, suffix='', rank=None):
        #pdb.set_trace()
        check_and_warn_input_range(batch['image'], 0, 1, 'DirectoryVisualizer target image')
        vis_img = visualize_mask_and_images_batch(batch, self.key_order, max_items=self.max_items_in_batch,
                                                  last_without_mask=self.last_without_mask,
                                                  rescale_keys=self.rescale_keys)

        vis_img = np.clip(vis_img * 255, 0, 255).astype('uint8')

        curoutdir = os.path.join(self.outdir, f'epoch{epoch_i:04d}{suffix}')
        os.makedirs(curoutdir, exist_ok=True)
        #if 'epoch2000_train' in curoutdir:
           #pdb.set_trace()
        
        #---------------------mines---------------------------
        name=''
        if len(batch['name'])==1:
            name=batch['name'][0].split('/')[-1]
        else:
            names=''
            #pdb.set_trace()
            for i in range(len(batch['name'])):
                name=batch['name'][i].split('/')[-1][:-4]
                names=names+'|'+name
            name=names
        #------------------------------------------------------
        
        rank_suffix = f'_r{rank}' if rank is not None else ''
        out_fname = os.path.join(curoutdir, f'{name}_batch{batch_i:07d}{rank_suffix}.jpg')
        
        if '_train/' in out_fname:
           #pdb.set_trace()
           root=out_fname.split('/|')[0]     
           name=out_fname.split('/|')[-1].replace('COCO_train2014_','')  
           name=name.replace('_batch0000000.jpg','.jpg')
           name=name.replace('000000','')
           name=name.replace('|','_')
           out_fname=root+'/'+name
           #out_fname.replace('COCO_train2014_','')
        #out_fname=out_fname.replace('COCO_train2014_','')

        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_fname, vis_img)      
