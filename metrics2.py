import os
import torch
import dutils
import colorful
import numpy as np
from benchmark.benchmark_utils import ChangeDir,AddPath
import skimage.transform
from benchmark.ground_truth_handler import get_gt,get_classname_and_target_id_from_obj
from cnn_utils import (normalize_tensor,denormalize_tensor,pascal_mean,pascal_std,)
IMAGENET_ROOT = "/root/bigfiles/dataset/imagenet"
PASCAL_ROOT = "/root/bigfiles/dataset/VOCdevkit"
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
class PointingGame():
    def __init__(self,model,dataset,tolerance=15,imagenet_root=None,):
        self.dataset = dataset
        self.tolerance = tolerance
        pass
    def run(self,input_tensor,grayscale_cams,target_id,bbox_info):
        #TODO = None
        class TODO():pass
        scale_size = TODO
        assert grayscale_cams.ndim == 4
        assert grayscale_cams.shape[:2] == (1,1)
        scale_size = grayscale_cams.shape[-2:]
        if False:
            import imagenet_localization_parser
            bbox_info = imagenet_localization_parser.get_voc_label(
                        root_dir = os.path.join(imagenet_root,'bboxes','val'),
                        x = imroot)
        print(bbox_info)
        imsize_WH = [int(bbox_info['annotation']['size'][k]) for k in ['width','height']]
        W,H = imsize_WH
        # bbox = bbox_info['annotation']['object'][0]['bndbox']
        bbox_objs = bbox_info['annotation']['object']
        saliency = grayscale_cams[0,0]
        assert saliency.ndim == 2
        saliency_original_size  = skimage.transform.resize(saliency,(H,W))
        highest_saliency = saliency_original_size.max()
        # print(highest_point)
        YX_highest = (saliency_original_size == highest_saliency).nonzero()
        if False:
            dutils.img_save(saliency_original_size,'saliency.png')
            dutils.img_save(input_tensor,'ref.png')
        if len(YX_highest[0]) > 1:
            # import ipdb;ipdb.set_trace()
            # YX_highest = (int(YX_highest[0].mean()),int(YX_highest[1].mean()))
            YX_highest = (int(YX_highest[0][0]),int(YX_highest[1][0]))
            
        ONE_BAD = [False]
        for bbox_obj in bbox_objs:
            bbox = bbox_obj['bndbox']
            classname,bbox_target_id = get_classname_and_target_id_from_obj(bbox_obj,self.dataset,pascal_root=PASCAL_ROOT,imagenet_root=IMAGENET_ROOT)
            if bbox_target_id != target_id:
                continue
            print(imsize_WH)
            print(scale_size)
            print(bbox)
            get_bbox_x0x1y0y1 = lambda bbox:[int(bbox[k]) for k in ['xmin','xmax','ymin','ymax']]
            bbox_x0x1y0y1 = get_bbox_x0x1y0y1(bbox)
            print(bbox_x0x1y0y1)
            
            W_224,H_224 = scale_size
            bbox_x0x1y0y1_224 = [loc * 1. * nsiz/siz for loc,siz,nsiz in zip(bbox_x0x1y0y1,[W,W,H,H],[W_224,W_224,H_224,H_224])]
            # print(bbox_x0x1y0y1_224)
            # print(grayscale_cams.shape)

            # print(YX_highest)

            """
            def is_in(YX_highest,bbox_x0x1y0y1_224):
                Ytest,Xtest = YX_highest
                left,right,top,bottom = bbox_x0x1y0y1_224
                assert left < right
                assert top < bottom
                return all([Ytest > top, Ytest < bottom, Xtest > left, Xtest < right])
            """
            def is_in(H,W,YX_highest,bbox_x0x1y0y1_224,saliency):
                # create a neighborhood mask around YX_highest
                v, u = torch.meshgrid((
                    (torch.arange(H,
                                dtype=torch.float32) - YX_highest[0])**2,
                    (torch.arange(W,
                                dtype=torch.float32) - YX_highest[1])**2,
                ))        
                nbd_mask = (v + u) < self.tolerance**2
                x0,x1,y0,y1 = bbox_x0x1y0y1_224
                obj_mask=torch.zeros(H,W,dtype=torch.bool)
                
                obj_mask[int(y0):int(y1),int(x0):int(x1)] = True
                hit = (obj_mask & nbd_mask).any()
                hit = hit.item()
                # import ipdb;ipdb.set_trace()
                
                if os.environ.get('DBG_POINTING_GAME',False) == '1':
                    # dutils.img_save(nbd_mask,'nbd_mask.png')
                    # dutils.img_save(obj_mask,'obj_mask.png')
                    # dutils.img_save(input_tensor,'ref.png')
                    # print(hit)
                    if not hit or ONE_BAD[0]:
                        ONE_BAD[0] = True
                        dutils.img_save(nbd_mask,'nbd_mask.png')
                        dutils.img_save(obj_mask,'obj_mask.png')
                        dutils.img_save(input_tensor,'ref.png')            
                        import ipdb;ipdb.set_trace()
                dutils.cipdb('DBG_POINTING_GAME')
                return hit
                # Ytest,Xtest = YX_highest
                # left,right,top,bottom = bbox_x0x1y0y1_224
                # assert left < right
                # assert top < bottom
                # return all([Ytest > top, Ytest < bottom, Xtest > left, Xtest < right])    
            hit = is_in(H,W,YX_highest,bbox_x0x1y0y1,saliency_original_size)
            if hit:
                """
                as long as 1 of the gt objects is hit, we consider it a hit
                """
                break
        # print(hit)
        # import ipdb;ipdb.set_trace()
        # def overlay_bbox_on_image_tensor(input_tensor,bbox_x0x1y0y1):
        def overlay_bbox_on_image_tensor(input_tensor,bbox_x0x1y0y1_224):
            import cv2
            import numpy as np
            assert input_tensor.shape[0] == 1
            input_ = input_tensor[0].permute(1,2,0).cpu().numpy()        
            x0,x1,y0,y1 = bbox_x0x1y0y1_224
            x0,x1,y0,y1 = int(x0),int(x1),int(y0),int(y1)
            x0,x1,y0,y1 = max(0,x0),min(x1,input_.shape[1]-1),max(0,y0),min(y1,input_.shape[0]-1)
            input_[y0:y1,x0] = 1
            input_[y0:y1,x1] = 1
            input_[y0,x0:x1] = 1
            input_[y1,x0:x1] = 1
            return input_
        overlayed = overlay_bbox_on_image_tensor(input_tensor,bbox_x0x1y0y1_224)
        metric_data = {
            'hit':hit,
            # 'overlayed_bbox':overlayed,
            'metricname':'pointing-game',
            'target_id':target_id,
        }
        # import ipdb;ipdb.set_trace()
        dutils.cipdb('DBG_POINTING_GAME')
        #%run -i dbg_pointing.ipy
        return metric_data
        
