#%%
# import ipdb;ipdb.set_trace()
# import os
# os.chdir('/root/evaluate-saliency-4/jigsaw')    
import register_ipdb
import dutils
import colorful
import torch
import torchvision
import numpy as np
from benchmark.benchmark_utils import ChangeDir,AddPath
import skimage.io
from PIL import Image
import sys
from cnn import get_target_id
import os
from pydoc import importfile
from benchmark import settings
from collections import defaultdict
import torch
import dutils
IMAGENET_ROOT = '/root/bigfiles/dataset/imagenet'
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
##############################################################################################

def denormalize_tensor(t,vgg_mean=[0.485, 0.456, 0.406],
                     vgg_std=[0.229, 0.224, 0.225]):
    device = t.device
    # out = (t - torch.tensor(vgg_mean).to(device)[None,:,None,None])/torch.tensor(vgg_std).to(device)[None,:,None,None]
    out = (t * torch.tensor(vgg_std).to(device)[None,:,None,None]) + torch.tensor(vgg_mean).to(device)[None,:,None,None]
    return out
def normalize_tensor(t,vgg_mean=[0.485, 0.456, 0.406],
                     vgg_std=[0.229, 0.224, 0.225]):
    device = t.device
    # out = (t - torch.tensor(vgg_mean).to(device)[None,:,None,None])/torch.tensor(vgg_std).to(device)[None,:,None,None]
    # out = (t * torch.tensor(vgg_std).to(device)[None,:,None,None]) + torch.tensor(vgg_mean).to(device)[None,:,None,None]
    out = t - torch.tensor(vgg_mean).to(device)[None,:,None,None]
    out = out / torch.tensor(vgg_std).to(device)[None,:,None,None]
    return out
def convert_keep_thresholds_to_delete_(d):
    new_d = {}
    for k,v in d.items():
        knew = 100 - k
        new_d[knew] = v
    # print('compare new_d and d');import pdb;pdb.set_trace()
    return new_d
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
def run_pointing_game(input_tensor,grayscale_cams,model,target_id,imroot,imagenet_root):
    #TODO = None
    class TODO():pass
    scale_size = TODO
    assert grayscale_cams.ndim == 4
    assert grayscale_cams.shape[:2] == (1,1)
    scale_size = grayscale_cams.shape[-2:]
    import imagenet_localization_parser
    bbox_info = imagenet_localization_parser.get_voc_label(
                root_dir = os.path.join(imagenet_root,'bboxes','val'),
                x = imroot)
    print(bbox_info)
    imsize_WH = [int(bbox_info['annotation']['size'][k]) for k in ['width','height']]
    bbox = bbox_info['annotation']['object'][0]['bndbox']
    print(imsize_WH)
    print(scale_size)
    print(bbox)
    bbox_x0x1y0y1_TODO = lambda bbox:[int(bbox[k]) for k in ['xmin','xmax','ymin','ymax']]
    bbox_x0x1y0y1 = bbox_x0x1y0y1_TODO(bbox)
    print(bbox_x0x1y0y1)
    W,H = imsize_WH
    new_W,new_H = scale_size

    scaled_bbox_x0x1y0y1 = [loc * 1. * nsiz/siz for loc,siz,nsiz in zip(bbox_x0x1y0y1,[W,W,H,H],[new_W,new_W,new_H,new_H])]
    # print(scaled_bbox_x0x1y0y1)
    # print(grayscale_cams.shape)
    saliency_HW = grayscale_cams[0,0]
    highest_point = saliency_HW.max()
    # print(highest_point)
    YX_highest = (saliency_HW == highest_point).nonzero()
    if len(YX_highest[0]) > 1:
        YX_highest = (int(YX_highest[0].mean()),int(YX_highest[1].mean()))
    # print(YX_highest)
    
    def is_in(YX_highest,scaled_bbox_x0x1y0y1):
        Ytest,Xtest = YX_highest
        left,right,top,bottom = scaled_bbox_x0x1y0y1
        assert left < right
        assert top < bottom
        return all([Ytest > top, Ytest < bottom, Xtest > left, Xtest < right])
    hit = is_in(YX_highest,scaled_bbox_x0x1y0y1)
    # print(hit)
    # import ipdb;ipdb.set_trace()
    # def overlay_bbox_on_image_tensor(input_tensor,bbox_x0x1y0y1):
    def overlay_bbox_on_image_tensor(input_tensor,scaled_bbox_x0x1y0y1):
        import cv2
        import numpy as np
        assert input_tensor.shape[0] == 1
        input_ = input_tensor[0].permute(1,2,0).cpu().numpy()        
        x0,x1,y0,y1 = scaled_bbox_x0x1y0y1
        x0,x1,y0,y1 = int(x0),int(x1),int(y0),int(y1)
        input_[y0:y1,x0] = 1
        input_[y0:y1,x1] = 1
        input_[y0,x0:x1] = 1
        input_[y1,x0:x1] = 1
        return input_
    overlayed = overlay_bbox_on_image_tensor(input_tensor,scaled_bbox_x0x1y0y1)
    metric_data = {
        'hit':hit,
        'overlayed_bbox':overlayed,
        'metricname':'pointing-game',
    }
    # import ipdb;ipdb.set_trace()
    return metric_data



def run_chattopadhyay(input_tensor,grayscale_cams,model,target_id):
    # import ipdb;ipdb.set_trace()

    
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
    masked = normalize_tensor(denormalize_tensor(input_tensor) * stretched_cams)
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
        'masked':tensor_to_numpy(masked),
        'metricname':'chattopadhyay',
    }
    for k,v in metric_data.items():
        if isinstance(v,torch.Tensor):
            metric_data[k] = tensor_to_numpy(v).squeeze()
    #print(metric_data)
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
##############################################################################################
#%% run
import pickle
import glob
# from utils import get_image_tensor
def get_image_tensor(impath,size=224):
    from cnn import get_vgg_transform
    vgg_transform = get_vgg_transform(size)
    im_ = skimage.io.imread(impath)
    if im_.ndim == 2:
        im_ = np.concatenate([im_[...,None],im_[...,None],im_[...,None]],axis=-1)
    im_pil = Image.fromarray(im_)
    ref = vgg_transform(im_pil).unsqueeze(0)
    return ref
#%%
available_metrics = [  
                    'perturbation',
                    'road',
                    ],
available_methodnames = [  
                'gradcam',
                'smoothgrad',
                'fullgrad',
                'integrated-gradients',
                'gradients',
                'inputXgradients',
                'jigsaw-saliency'
                ]
def get_hack_for_im_save_dirs(methodname,modelname):
    save_dir = os.path.join(settings.RESULTS_DIR_librecam,f'{methodname}-{modelname}')
    keep_imroots = glob.glob(os.path.join(save_dir,'*/'))
    keep_imroots = [ os.path.basename(n.rstrip(os.path.sep)) for n in keep_imroots]
    keep_imroots = [ ( n[:-len('.JPEG')] if '.' in n else n) for n in keep_imroots]
    def hack_for_im_save_dirs(im_save_dirs):
        out_im_save_dirs = []
        for im_save_dir in im_save_dirs:
            imroot = os.path.basename(im_save_dir.rstrip(os.path.sep))    
            # impath = os.path.join(imagenet_root,'images','val',imroot + '.JPEG')
            if imroot in keep_imroots:
                out_im_save_dirs.append(im_save_dir)
        # assert len(out_im_save_dirs) == len(keep_imroots)
        return out_im_save_dirs 
    
    return hack_for_im_save_dirs


def main(
    metrics = available_metrics,
    methodnames = available_methodnames,
    skip=False,
    start=0,
    end=None,
    modelname = 'vgg16',
    use_images_common_to=None,
    delete_percentiles=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    imagenet_root = IMAGENET_ROOT
    NBIG = 100
    assert isinstance(metrics,list) 
    assert isinstance(methodnames,list) 
    def get_model(modelname,device=device):
        print('TODO:move me to outside')
        import libre_cam_models.relevance.vgg
        import libre_cam_models.relevance.resnet

        if modelname == 'vgg16':
            model = libre_cam_models.relevance.vgg.vgg16_bn(pretrained = True).to(device)
        else:
            model = libre_cam_models.relevance.resnet.resnet50(pretrained = True).to(device)        
        return model
    model = get_model(modelname,device=device)

    model.eval()
    # impaths = ['/root/evaluate-saliency-4/cyclegan results/samoyed1.png'] 
    # class_name = 'samoyed'
    '''
    image_paths = sorted(glob.glob(os.path.join(imagenet_root,'images','val','*.JPEG')))
    image_paths = image_paths[start:]
    if end is not None:
        image_paths = image_paths[:end]
    size=(224)
    image_paths = list(image_paths)
    '''
    has_errors = []
    ##############################################################################################
    # assert False
    for methodname in methodnames:
        for metricname in metrics:
            methoddir =  os.path.join(settings.RESULTS_DIR_librecam,f'{methodname}-{modelname}')
            print(methoddir)
            #============================================================
            im_save_dirs = sorted(glob.glob(os.path.join(methoddir,'*/')))
            if end is not None:
                im_save_dirs = im_save_dirs[start:end]
            else:
                im_save_dirs = im_save_dirs[start:]
            im_save_dirs = list(im_save_dirs)
            # import ipdb;ipdb.set_trace()
            if len(im_save_dirs) == 0:
                # import colorful
                import ipdb;ipdb.set_trace()
                print(colorful.magenta('im_save_dirs is empty'))
                import time;time.sleep(3)
            #============================================================
            if use_images_common_to is not None:
                # assert False
                assert use_images_common_to in available_methodnames
                hack_for_im_save_dirs = get_hack_for_im_save_dirs(use_images_common_to,modelname)
                print('#'*50,'\n','hack_for_im_save_dirs','\n','#'*50)    
                im_save_dirs = hack_for_im_save_dirs(im_save_dirs)
            consolidated = defaultdict(lambda :defaultdict(int))

            # print(im_save_dirs)
            for i,im_save_dir in enumerate(im_save_dirs):
                if delete_percentiles == [20,40,60,80]:
                    print(f'TODO: you sure the delete percentiles are {delete_percentiles}')
                imroot = os.path.basename(im_save_dir.rstrip(os.path.sep))
                impath = os.path.join(imagenet_root,'images','val',imroot + '.JPEG')
                #----------------------------------------------------------------
                pklname = glob.glob(os.path.join(im_save_dir,'*.pkl'))
                assert len(pklname) == 1
                pklname = pklname[0]
                try:
                    with open(pklname,'rb') as f:
                        loaded = pickle.load(f)
                    saliency = loaded.get('saliency',np.nan)
                    if isinstance(saliency,torch.Tensor):
                        saliency = tensor_to_numpy(saliency)
                    assert not np.isnan(np.sum(saliency))
                except EOFError as e:
                    print(colorful.red(f'{im_save_dir} has errors,continuing'))
                    import time;time.sleep(2)
                    has_errors.append(im_save_dir)
                    continue
                except AssertionError as e:
                    has_errors.append(im_save_dir)
                    print(colorful.red(f'{im_save_dir} has nan,continuing'))
                    import time;time.sleep(2)
                    continue
                # import ipdb;ipdb.set_trace()
                #--------------------------------------------------------------------
                target_id = loaded['target_id']
                import imagenet_synsets
                classname = imagenet_synsets.synsets[target_id]['label']
                classname = classname.split(',')[0]
                classname = classname.replace(' ','_')
                #--------------------------------------------------------------------
                if skip:
                    save_dir = os.path.join(settings.RESULTS_DIR_librecam,f'{metricname}-{methodname}-{modelname}',imroot)
                    pklname = os.path.join(save_dir,f'{classname}{target_id}.pkl')
                    if os.path.exists(pklname):
                        try:
                            with open(pklname,'rb') as f:
                                pass
                            print(f'{pklname} exists, skipping')
                            continue
                        except EOFError as e:
                            print(f'{pklname} corrupt,overwriting')
                        
                        
                    elif methodname == 'gradcam':
                        '''
                        was there some issue with gradcam, that i kept this section?
                        '''

                        # import pdb;pdb.set_trace()
                        pass
                #--------------------------------------------------------------------
                # if methodname == 'jigsaw-saliency':
                #     print(im_save_dir)
                #     imroot = os.path.basename(im_save_dir.rstrip(os.path.sep))
                #     print(imroot)
                #     impath = os.path.join(imagenet_root,'images','val',imroot + '.JPEG')                    
                #     input_tensor = ref = get_image_tensor(impath,size=size).to(device)
                #     tensor_to_numpy = lambda t:t.detach().cpu().numpy()
                #     loaded['saliency'] = tensor_to_numpy(torch.zeros_like(input_tensor)[:,:1])
                #     loaded['target_id'] = 10
                #     loaded['method'] = 'jigsaw-saliency'
                if methodname == 'inputXgradients':
                    assert loaded['method'] ==  methodname[:-1]
                elif methodname == 'jigsaw-saliency':
                    if 'method' in loaded:
                        assert loaded['method'] ==  methodname
                else:
                    try:
                        assert loaded['method'] ==  methodname
                    except AssertionError:
                        if methodname == 'gradcam' and loaded['method'] == 'scorecam':
                            import colorful
                            print(colorful.orange('known problem, saliency_data for gradcam stored the method as "scorecam"'))
                try:
                    saliency = loaded['saliency']
                except KeyError as e:
                    assert methodname == 'jigsaw-saliency'
                    saliency = loaded['trends']['im_overlayed_importances']

                if saliency.ndim == 4:
                    assert saliency.shape[0] == 1
                    saliency = saliency[0]
                if saliency.shape[0] > 1:
                    # 3 channel
                    # for integrated gradients
                    saliency = saliency.sum(dim=0,keepdim=True)
                # import pdb;pdb.set_trace()
                # target_id = loaded['target_id']
                # input_tensor = ref = get_image_tensor(impath,size=size).to(device)
                input_tensor = get_image_tensor(impath,size=saliency.shape[-2:]).to(device)
                if isinstance(saliency,torch.Tensor):
                    saliency = tensor_to_numpy(saliency)
                if saliency.ndim == 3:
                    saliency = saliency[None,...]
                if metricname == 'road':
                    # import time
                    # t0 = time.time()                    
                    metric_data = run_road(input_tensor,saliency,model,target_id,delete_percentiles=delete_percentiles)
                    # t1 = time.time()
                    # print(f'elapsed:{t1-t0}')                    
                elif metricname == 'gpnn_eval':
                    # import ipdb;ipdb.set_trace()
                    # import time
                    # t0 = time.time()
                    metric_data = run_gpnn_eval(input_tensor,saliency,model,target_id,delete_percentiles=delete_percentiles)                    
                    # t1 = time.time()
                    # print(f'elapsed:{t1-t0}')
                elif metricname == 'perturbation':
                    # import ipdb;ipdb.set_trace()
                    metric_data =run_perturbation(input_tensor,saliency,model,target_id,delete_percentiles=delete_percentiles)
                    if i > NBIG:
                        # ['mrf_scores', 'lrf_scores', 'metricname', 'mrf_probability_increase', 'mrf_increase_indicator', 'mrf_binary_mask', 'mrf_perturbed', 'lrf_probability_increase', 'lrf_increase_indicator', 'lrf_binary_mask', 'lrf_perturbed', 'modelname', 'methodname', 'classname', 'delete_percentiles', 'target_id', 'saliency']
                        for k in ['lrf_binary_mask','lrf_perturbed','mrf_binary_mask','mrf_perturbed']:
                            if k in metric_data:
                                del metric_data[k]
                    # import ipdb;ipdb.set_trace()
                    
                    if i < NBIG:
                        for order_name in ['mrf','lrf']:
                            for extra_info_name in ['probability_increase','increase_indicator','binary_mask', 'perturbed']:                        
                                for t,val in metric_data[ f'{order_name}_{extra_info_name}'].items():
                                    consolidated[f'{order_name}_{extra_info_name}'][t] += metric_data[f'{order_name}_{extra_info_name}'][t]          
                                    
                    metric_data.update(
                        dict(
                            # modelname = modelname,
                            # methodname = methodname,
                            # classname = classname,
                            delete_percentiles=delete_percentiles,
                            # target_id=target_id,
                            # saliency = saliency,
                        )
                    )                                              
                elif metricname == 'chattopadhyay':
                    # import ipdb;ipdb.set_trace()
                    metric_data =run_chattopadhyay(input_tensor,saliency,model,target_id)
                    # '''
                    if i > NBIG:
                        # ['mrf_scores', 'lrf_scores', 'metricname', 'mrf_probability_increase', 'mrf_increase_indicator', 'mrf_binary_mask', 'mrf_perturbed', 'lrf_probability_increase', 'lrf_increase_indicator', 'lrf_binary_mask', 'lrf_perturbed', 'modelname', 'methodname', 'classname', 'delete_percentiles', 'target_id', 'saliency']
                        for k in ['masked']:
                            if k in metric_data:
                                del metric_data[k]
                    # import ipdb;ipdb.set_trace()
                    # '''                    
                elif metricname == 'pointing-game':
                    metric_data = run_pointing_game(input_tensor,saliency,model,target_id,imroot,IMAGENET_ROOT)

                    if i > NBIG:
                        for k in 'overlayed_bbox':
                            if k in metric_data:
                                del metric_data[k]
                        pass

                    # import ipdb;ipdb.set_trace()
                    # '''                                        
                else:
                    assert False,f'{metricname} not recognized'
                #----------------------------------------------------------
                # import imagenet_synsets
                # classname = imagenet_synsets.synsets[target_id]['label']
                # classname = classname.split(',')[0]
                if False:
                    classname = '_'.join(classname)
                else:
                    classname = classname.replace(' ','_')                    
                metric_data.update(
                    dict(
                        modelname = modelname,
                        methodname = methodname,
                        classname = classname,
                        # delete_percentiles=delete_percentiles,
                        target_id=target_id,
                        saliency = saliency,
                    )
                )

                '''
                for t,val in metric_data['mrf_per_threshold'].items():
                    consolidated['mrf_per_threshold'][t] += metric_data['mrf_per_threshold'][t]
                for t,val in metric_data['lrf_per_threshold'].items():
                    consolidated['lrf_per_threshold'][t] += metric_data['lrf_per_threshold'][t]
                '''                            
                # if 'scores_after_imputation_mrf' in metric_data:
                #     consolidated['mrf_per_threshold']
                # if 'scores_after_imputation_lrf' in metric_data:
                #----------------------------------------------------------
                from benchmark.benchmark_utils import create_dir
                # im_save_dir = create_im_save_dir(experiment_name=f'{metricname}-{methodname}-{modelname}',impath=impath)  
                save_dir = create_dir(os.path.join(f'{metricname}-{methodname}-{modelname}',imroot),root_dir=settings.METRICS_DIR_librecam)
                savename = os.path.join(save_dir,f'{classname}{target_id}.pkl')
                # import pdb;pdb.set_trace()
                if False and 'relaxing requirement of 20 thresholds':
                    assert len(metric_data['mrf_per_threshold'].keys()) == 20
                # import ipdb;ipdb.set_trace()
                with open(savename,'wb') as f:
                    pickle.dump(metric_data,f)
            '''
            consolidated['n_images'] = len(im_save_dirs)

            for t,val in metric_data['mrf_per_threshold'].items():
                consolidated['mrf_per_threshold'][t] /= len(im_save_dirs)
            for t,val in metric_data['lrf_per_threshold'].items():
                consolidated['lrf_per_threshold'][t] /= len(im_save_dirs)
            #----------------------------------------------------------
            consolidated_savename = os.path.join(settings.METRICS_DIR_librecam,f'{metricname}-{methodname}-{modelname}','consolidated.pkl')
            with open(consolidated_savename,'wb') as f:
                pickle.dump(dict(consolidated),f)        
            '''
            # break
        # break
##############################################################################################
def outer_main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--methodnames',nargs='*',default = [  
                    #'gradcam',
                    #'smoothgrad',
                    #'fullgrad',
                    #'integrated-gradients',
                    #'gradients',
                    #'inputXgradients',
                    #'jigsaw-saliency',
                    # 'elp',
                    ])
    parser.add_argument('--metrics',nargs='*',
                        default=[  
                        'perturbation',
                        # 'road',
                        ])
    
    parser.add_argument('--modelname',type=str,default='vgg16')
    parser.add_argument('--skip',type=lambda v: (v.lower()=='true'),default=False)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--end',type=int,default=None)
    parser.add_argument('--use-images-common-to',type=str,default=None)
    
    # from libracam paper: page 5 figure 1: the comparison of attribution quality at threshold levels t ∈ [0, 1] with the increment of 0.1 from left to right in all plots. The
    # quality measures are evaluated with images multiplied with attribution maps where the attribution values less than t are set to the zero value.
    parser.add_argument('--delete_percentiles',type=float,nargs='*',default=[0,10,20, 30,40, 50,60, 70,80,90,100])
    args = parser.parse_args()
    # import pdb;pdb.set_trace()
    main(metrics=args.metrics,methodnames=args.methodnames,skip=args.skip,start=args.start,end=args.end,use_images_common_to=args.use_images_common_to,delete_percentiles=args.delete_percentiles,modelname=args.modelname)
    
if __name__ == '__main__':    
    outer_main()
# %%
