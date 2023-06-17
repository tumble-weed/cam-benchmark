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
def convert_keep_thresholds_to_delete_(d):
    new_d = {}
    for k,v in d.items():
        knew = 100 - k
        new_d[knew] = v
    # print('compare new_d and d');import pdb;pdb.set_trace()
    return new_d
def run_road(input_tensor,grayscale_cams,model,target_id,
                delete_percentiles=[20, 40, 60, 80]):
    # if delete_percentiles is None:
    #     delete_percentiles=[20, 40, 60, 80]
    import pytorch_grad_cam.metrics.perturbation_confidence as perturbation_confidence
    with AddPath('benchmark/pytorch_grad_cam') as ctx:
        #-----------------------------------------------------------------------
        from pytorch_grad_cam.metrics.road import ROADMostRelevantFirstAverage,ROADLeastRelevantFirstAverage
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        #-----------------------------------------------------------------------
        assert not isinstance(grayscale_cams,torch.Tensor)
        assert grayscale_cams.ndim == 4
        grayscale_cams = grayscale_cams[:,0,...]
        
        targets = [ClassifierOutputTarget(target_id)]
        mrf_per_threshold = {}
        mrf_cam_metric = ROADMostRelevantFirstAverage(percentiles=100 - np.array(delete_percentiles))

        mrf_scores = mrf_cam_metric(input_tensor, grayscale_cams, targets, model,extras=mrf_per_threshold)
        # original_scores_mrf =  perturbation_confidence.extras['original_scores'] 
        # scores_after_imputation_mrf =  perturbation_confidence.extras['scores_after_imputation'] 
        print(f"The average confidence increase with most relevant ROAD accross 4 thresholds: {mrf_scores[0]}")

        lrf_cam_metric = ROADLeastRelevantFirstAverage(percentiles=100 - np.array( delete_percentiles))
        lrf_per_threshold = {}
        lrf_scores = lrf_cam_metric(input_tensor, grayscale_cams, targets, model,extras=lrf_per_threshold)

        mrf_per_threshold = convert_keep_thresholds_to_delete_(mrf_per_threshold)
        lrf_per_threshold = convert_keep_thresholds_to_delete_(lrf_per_threshold)
        # print('check if lrf and mrf dict are different');import pdb;pdb.set_trace()    
                
        # original_scores_lrf =  perturbation_confidence.extras['original_scores'] 
        # scores_after_imputation_lrf = perturbation_confidence.extras['scores_after_imputation'] 
        print(f"The average confidence increase with least relevant ROAD accross 4 thresholds: {lrf_scores[0]}")
        metric_data = {'mrf_scores':mrf_scores,'lrf_scores':lrf_scores,'metricname':'road',
        'mrf_per_threshold':mrf_per_threshold,
        'lrf_per_threshold':lrf_per_threshold,
        # 'scores_after_imputation_mrf':scores_after_imputation_mrf,
        # 'original_scores_mrf':original_scores_mrf,
        # 'scores_after_imputation_lrf':scores_after_imputation_lrf,
        # 'original_scores_lrf':original_scores_lrf
        }
        return metric_data
        # return (mrf_cam_metric,lrf_cam_metric)
##############################################################################################
from benchmark.benchmark_utils import ChangeDir
with AddPath('benchmark/pytorch_grad_cam') as ctx:
    from pytorch_grad_cam.metrics.perturbation_confidence import PerturbationConfidenceMetric,AveragerAcrossThresholds
    from pytorch_grad_cam.metrics.perturbation_confidence import RemoveMostRelevantFirst,RemoveLeastRelevantFirst
    from pytorch_grad_cam.metrics.cam_mult_image import multiply_tensor_with_cam,CamMultImageConfidenceChange,IncreaseInConfidence
    class PerturbationMostRelevantFirst(PerturbationConfidenceMetric):
        def __init__(self, percentile=80):
            super(PerturbationMostRelevantFirst, self).__init__(
                RemoveMostRelevantFirst(percentile, 
                multiply_tensor_with_cam
                # IncreaseInConfidence()
                ))

    class PerturbationMostRelevantFirstAverage(AveragerAcrossThresholds):
        def __init__(self, percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]):
            super(PerturbationMostRelevantFirstAverage, self).__init__(
                PerturbationMostRelevantFirst, percentiles)

    class PerturbationLeastRelevantFirst(PerturbationConfidenceMetric):
        def __init__(self, percentile=80):
            super(PerturbationLeastRelevantFirst, self).__init__(
                RemoveLeastRelevantFirst(percentile, 
                multiply_tensor_with_cam
                # IncreaseInConfidence()
                ))

    class PerturbationLeastRelevantFirstAverage(AveragerAcrossThresholds):
        def __init__(self, percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]):
            super(PerturbationLeastRelevantFirstAverage, self).__init__(
                PerturbationLeastRelevantFirst, percentiles)


def run_pointing_game(input_tensor,grayscale_cams,model,target_id,imroot,bbox_info,dataset,imagenet_root=None,tolerance=15):
    
    #TODO = None
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
        classname,bbox_target_id = get_classname_and_target_id_from_obj(bbox_obj,dataset,pascal_root=PASCAL_ROOT,imagenet_root=IMAGENET_ROOT)
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
            nbd_mask = (v + u) < tolerance**2
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
                    input_tensor_for_viz = torch.nn.functional.interpolate(input_tensor,(H,W))
                    dutils.img_save(input_tensor_for_viz,'ref_point.png')            
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
    for k in ['hit']:
        print(colorful.magenta(f'{k}:{metric_data[k]}'))    
    # dutils.cipdb('DBG_POINTING_GAME')
    #%run -i dbg_pointing.ipy
    return metric_data



def run_chattopadhyay(input_tensor,grayscale_cams,model,target_id,dataset):
    # import ipdb;ipdb.set_trace()
    if not any([input_tensor.min() < 0,input_tensor.max() > 1]):
        print(colorful.red('input tensor seems not to be normalized'))
    # if not all([grayscale_cams.min() == 1,grayscale_cams.max() == 1]):
    #     print(colorful.red_on_cyan('grayscale_cam does not have full range?'))
    original_scores = model(input_tensor)
    
    original_scores = torch.softmax(original_scores,dim=1)
    assert original_scores.ndim in [2,4]
    if original_scores.ndim == 4:
        assert original_scores.shape[-2:] == (1,1)
        original_scores = original_scores.mean(dim=(-1,-2))
    is_max = original_scores.argmax(dim=1) == target_id
    
    device= input_tensor.device
    original_output = model(input_tensor)
    original_probs = torch.softmax(original_output,dim=1)
    #print(original_scores.shape)
    original_score = original_output[:,target_id]
    original_prob = original_probs[:,target_id]
    if original_score.ndim == 3:
        original_score = original_score.mean(dim=(-1,-2))
        original_prob = original_prob.mean(dim=(-1,-2))
    print(original_score)
    
    #dutils.array_info(grayscale_cams)
    assert grayscale_cams.min() >= 0
    assert grayscale_cams.shape == (1,1) + input_tensor.shape[-2:]
    grayscale_cams = torch.tensor(grayscale_cams,device=device)
    m = grayscale_cams.min()
    M = grayscale_cams.max()
    denom = (M - m)
    denom = denom + (denom == 0).float()
    stretched_cams  = (grayscale_cams - m)/denom
    # import ipdb; ipdb.set_trace()
    if dataset  == 'imagenet':
        masked = normalize_tensor(denormalize_tensor(input_tensor) * stretched_cams)
    elif dataset == 'pascal':
        masked = normalize_tensor(denormalize_tensor(input_tensor,
                                                     vgg_mean = pascal_mean,
                                                    vgg_std = pascal_std) * stretched_cams,
                                  vgg_mean = pascal_mean,
                                  vgg_std = pascal_std)
    else:
        assert False
    # import ipdb; ipdb.set_trace()
    print(colorful.magenta('denormalize mask multiply in chattopadhyay'))
    masked_output = model(masked)
    masked_probs = torch.softmax(masked_output,dim=1)
    masked_score  = masked_output[:,target_id]
    masked_prob = masked_probs[:,target_id]
    if masked_score.ndim  == 3:
        masked_score =masked_score.mean(dim=(-1,-2))
        masked_prob = masked_prob.mean(dim=(-1,-2))
    #print(original_score,masked_score)
    drop_in_prob = torch.clamp( original_prob - masked_prob,0,None)
    rise_in_prob = torch.clamp(masked_prob - original_prob,0,None)
    denom1 = original_prob
    denom1 = denom1 + (denom1 == 0).float()
    normalized_drop_in_prob = drop_in_prob/denom1
    # normalized_rise_in_prob = rise_in_prob/original_prob
    denom2 = (1 -original_prob) 
    denom2 = denom2 + (denom2 == 0).float()
    normalized_rise_in_prob = rise_in_prob/denom2
    #============================================
    denom3 = (original_prob) 
    denom3 = denom3 + (denom3 == 0).float()    
    normalized_rise_in_prob2 = rise_in_prob/denom3
    #============================================
    if (normalized_rise_in_prob > 1):
        import ipdb;ipdb.set_trace()
    if (normalized_drop_in_prob > 1):
        import ipdb;ipdb.set_trace()        
    increase_indicator = (rise_in_prob > 0).float()
    decrease_indicator = (drop_in_prob > 0).float()
    metric_data = {
        'normalized_drop_in_prob' : normalized_drop_in_prob,
        'normalized_rise_in_prob' : normalized_rise_in_prob,
        'increase_indicator' : increase_indicator,
        'decrease_indicator' : decrease_indicator,
        'normalized_rise_in_prob2':normalized_rise_in_prob2,
        'masked':tensor_to_numpy(masked),
        'metricname':'chattopadhyay',
        'is_max':is_max.item(),
        'original_prob':original_prob.item(),
        'masked_prob':masked_prob.item(),
        'saliency':tensor_to_numpy(stretched_cams),
    }
    for k,v in metric_data.items():
        if isinstance(v,torch.Tensor):
            metric_data[k] = tensor_to_numpy(v).squeeze()
    for k in ['increase_indicator','normalized_drop_in_prob']:
        print(colorful.magenta(f'{k}:{metric_data[k]}'))
    
    if os.environ.get('DBG_METRICS_VIZ',False)=='1':
        from matplotlib import cm
        import dutils
        # tensor_to_numpy = lambda t:t.detach().cpu().numpy()
        dutils.img_save(
            cm.jet(tensor_to_numpy((stretched_cams)[0,0])),
                        'saliency.png')
        
        print(metric_data['masked_prob'],metric_data['normalized_drop_in_prob'])
        dutils.img_save(tensor_to_numpy(denormalize_tensor(input_tensor*stretched_cams).permute(0,2,3,1)[0]),
                'masked.png')

    # import ipdb;ipdb.set_trace()
    return metric_data
    
def run_perturbation(input_tensor,grayscale_cams,model,target_id,delete_percentiles=None):
    with AddPath('benchmark/pytorch_grad_cam') as ctx:
        # percentiles=[0,20, 40, 60, 80]
        # percentiles=[0 , 10 , 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # percentiles=[20, 30, 40, 50, 60, 70, 80, 90, 100]
        # percentiles= np.linspace(1,0.9,10)
        #-----------------------------------------------------------------------
        # from pytorch_grad_cam.metrics.perturbation_confidence import (PerturbationConfidenceMetric,
        #                                                               RemoveMostRelevantFirst,
        #                                                               RemoveLeastRelevantFirst)
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        #-----------------------------------------------------------------------
        try:
            assert not isinstance(grayscale_cams,torch.Tensor)
        except AssertionError as e:
            import pdb;pdb.set_trace()
        assert grayscale_cams.ndim == 4
        grayscale_cams = grayscale_cams[:,0,...]
        
        targets = [ClassifierOutputTarget(target_id)]
        mrf_per_threshold = {}
        # import ipdb;ipdb.set_trace()
        mrf_cam_metric = PerturbationMostRelevantFirstAverage(percentiles=100 - np.array(delete_percentiles))
        mrf_scores = mrf_cam_metric(input_tensor, grayscale_cams, targets, model,extras=mrf_per_threshold)
        print(f"The average confidence increase with most relevant Perturbation accross  thresholds: {mrf_scores[0]}")

        lrf_cam_metric = PerturbationLeastRelevantFirstAverage(percentiles=100 - np.array(delete_percentiles))
        lrf_per_threshold = {}
        lrf_scores = lrf_cam_metric(input_tensor, grayscale_cams, targets, model,extras=lrf_per_threshold)
        # import ipdb;ipdb.set_trace()
        # mrf_probability_increase_per_threshold =
        # mrf_increase_indicator_per_threshold =
        # mrf_per_threshold_out = {}
        # lrf_per_threshold_out = {}
        if os.environ.get('DBG_METRICS_VIZ',False) == '1':
            print(lrf_scores)
            print(mrf_scores)
            import ipdb;ipdb.set_trace()
            pass
        detailed_out = {}
        # import ipdb;ipdb.set_trace()
        for order_name,order_data in ({'mrf':mrf_per_threshold,'lrf':lrf_per_threshold}).items():
            detailed_out[order_name] = {}
            for k in ['probability_increase','increase_indicator','binary_mask', 'perturbed']:
                detailed_out[order_name][k] = convert_keep_thresholds_to_delete_({percentile:order_data[percentile][k] for percentile in mrf_per_threshold})
         
        print(colorful.magenta('REMEMBER:the scores etc are according to deletion'))
        # print('check if lrf and mrf dict are different');import pdb;pdb.set_trace()    
        print(f"The average confidence increase with least relevant Perturbation accross  thresholds: {lrf_scores[0]}")
        metric_data = {'mrf_scores':mrf_scores,
                       'lrf_scores':lrf_scores,
                       'metricname':'perturbation',
        }
        for order_name in detailed_out:
            for k in ['probability_increase','increase_indicator','binary_mask', 'perturbed']:
                assert k in detailed_out[order_name]
                metric_data[f'{order_name}_{k}'] = detailed_out[order_name][k]
        # import ipdb;ipdb.set_trace()
        return metric_data
        # return (mrf_cam_metric,lrf_cam_metric)

            

def run_gpnn_eval(input_tensor,grayscale_cams,model,target_id,
                delete_percentiles=[20, 40, 60, 80]):
    # if delete_percentiles is None:
    #     delete_percentiles=[20, 40, 60, 80]
    import pytorch_grad_cam.metrics.perturbation_confidence as perturbation_confidence
    with AddPath('benchmark/pytorch_grad_cam') as ctx:
        #-----------------------------------------------------------------------
        from pytorch_grad_cam.metrics.gpnn_eval import GPNNMostRelevantFirstAverage,GPNNLeastRelevantFirstAverage
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        #-----------------------------------------------------------------------
        assert not isinstance(grayscale_cams,torch.Tensor)
        assert grayscale_cams.ndim == 4
        grayscale_cams = grayscale_cams[:,0,...]
        
        targets = [ClassifierOutputTarget(target_id)]
        mrf_per_threshold = {}
        mrf_cam_metric = GPNNMostRelevantFirstAverage(percentiles=100 - np.array(delete_percentiles))

        mrf_scores = mrf_cam_metric(input_tensor, grayscale_cams, targets, model,extras=mrf_per_threshold)
        # import ipdb;ipdb.set_trace()
        # original_scores_mrf =  perturbation_confidence.extras['original_scores'] 
        # scores_after_imputation_mrf =  perturbation_confidence.extras['scores_after_imputation'] 
        print(f"The average confidence increase with most relevant GPNN accross 4 thresholds: {mrf_scores[0]}")

        lrf_cam_metric = GPNNLeastRelevantFirstAverage(percentiles=100 - np.array( delete_percentiles))
        lrf_per_threshold = {}
        lrf_scores = lrf_cam_metric(input_tensor, grayscale_cams, targets, model,extras=lrf_per_threshold)
        # import ipdb;ipdb.set_trace()
        mrf_per_threshold = convert_keep_thresholds_to_delete_(mrf_per_threshold)
        lrf_per_threshold = convert_keep_thresholds_to_delete_(lrf_per_threshold)
        # print('check if lrf and mrf dict are different');import pdb;pdb.set_trace()    
                
        # original_scores_lrf =  perturbation_confidence.extras['original_scores'] 
        # scores_after_imputation_lrf = perturbation_confidence.extras['scores_after_imputation'] 
        print(f"The average confidence increase with least relevant GPNN accross 4 thresholds: {lrf_scores[0]}")
        metric_data = {'mrf_scores':mrf_scores,'lrf_scores':lrf_scores,'metricname':'road',
        'mrf_per_threshold':mrf_per_threshold,
        'lrf_per_threshold':lrf_per_threshold,
        # 'scores_after_imputation_mrf':scores_after_imputation_mrf,
        # 'original_scores_mrf':original_scores_mrf,
        # 'scores_after_imputation_lrf':scores_after_imputation_lrf,
        # 'original_scores_lrf':original_scores_lrf
        }
        return metric_data
        # return (mrf_cam_metric,lrf_cam_metric)

            