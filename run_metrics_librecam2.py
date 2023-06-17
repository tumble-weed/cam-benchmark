#%%
# import ipdb;ipdb.set_trace()
# import os
# os.chdir('/root/evaluate-saliency-4/jigsaw')    
# python -m benchmark.run_metrics_librecam --modelname resnet50 --methodnames gpnn-gradcam --dataset imagenet --metrics chattopadhyay --skip false
import register_ipdb
import dutils
import colorful
import torch
import torchvision
import numpy as np
from benchmark.benchmark_utils import ChangeDir,AddPath,update_running_avg
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
import lzma
from median_filter import weighted_median_filter
import blosc2
# from benchmark.pascal_run_competing_saliency_librecam2 import get_gt
from benchmark.ground_truth_handler import get_gt,get_classname_and_target_id_from_obj
import skimage.transform
from benchmark.metrics_data_handler import MetricsDataHandler
IMAGENET_ROOT = "/root/bigfiles/dataset/imagenet"
PASCAL_ROOT = "/root/bigfiles/dataset/VOCdevkit"
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
USE_LZMA = True
from cnn_utils import (normalize_tensor,denormalize_tensor,pascal_mean,pascal_std,)
from benchmark.metrics import (run_perturbation,run_road,run_chattopadhyay,run_pointing_game,run_gpnn_eval,)

##############################################################################################
#%% run
import pickle
import glob
from cnn_utils import get_image_tensor,get_model
from benchmark.benchmark_utils import get_hack_for_im_save_dirs
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

##############################################################################################
def main(
    metrics = available_metrics,
    methodnames = available_methodnames,
    skip=False,
    start=0,
    end=None,
    modelname = 'vgg16',
    use_images_common_to=None,
    dataset = None,
    delete_percentiles=None,
    median_k = 41):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pascal_root = PASCAL_ROOT
    imagenet_root = IMAGENET_ROOT
    pascal_root = "/root/bigfiles/dataset/VOCdevkit"
    NBIG = 100
    assert isinstance(metrics,list) 
    assert isinstance(methodnames,list) 

    
    has_relevancecam = 'relevancecam' if [any('relevancecam' in methodname for methodname in methodnames)] else None
    # import ipdb;ipdb.set_trace()
    if len(metrics) == 1 and 'pointing_game' in metrics:
        model = get_model(dataset,'dummymodel',has_relevancecam,device=device)    
    else:    
        model = get_model(dataset,modelname,has_relevancecam,device=device)    
    model.eval()

    has_errors = []
    running_averages = {}
    #############################################################################################
    # assert False
    
    for methodname in methodnames:
        for metricname in metrics:
            if dataset in ['voc','pascal']:
                dataset_stub = f'pascal-'
            elif dataset in ['imagenet']:
                dataset_stub = f'imagenet-'
                # dataset_stub = ''
            else:
                assert False,f'dataset not understood:{dataset}'
            # methoddir =  os.path.join(settings.RESULTS_DIR_librecam,f'{dataset_stub}{methodname}-{modelname}')
            methodxz = os.path.join(settings.RESULTS_DIR_librecam,f'{dataset_stub}{methodname}-{modelname}.xz')
            print(os.path.exists(methodxz))
            with lzma.open(methodxz,'rb') as f:
                methoddata = pickle.load(f)
            
            saliencybl2 = os.path.join(settings.RESULTS_DIR_librecam,f'{dataset}-{methodname}-{modelname}-saliency.bl2')
            loaded_saliency = blosc2.load_array(saliencybl2)                            
            imroots = list(methoddata.keys())

            metrics_obj = MetricsDataHandler(dataset,metricname,methodname,modelname)
            #============================================================
            if len(imroots) == 0:
                # import colorful
                print(colorful.magenta(f'im_save_dirs is empty,methodxz:{methodxz}'))
                import ipdb;ipdb.set_trace()
                import time;time.sleep(3)
            #============================================================
            if use_images_common_to is not None:
                assert False,'not implemented'
                # assert use_images_common_to in available_methodnames
                # hack_for_im_save_dirs = get_hack_for_im_save_dirs(use_images_common_to,modelname)
                # print('#'*50,'\n','hack_for_im_save_dirs','\n','#'*50)    
                # im_save_dirs = hack_for_im_save_dirs(im_save_dirs)
            consolidated = defaultdict(lambda :defaultdict(int))


            for i,imroot in enumerate(imroots):
                metrics_obj.clear_create_dir(imroot)
                if delete_percentiles == [20,40,60,80]:
                    print(f'TODO: you sure the delete percentiles are {delete_percentiles}')
                # imroot = os.path.basename(im_save_dir.rstrip(os.path.sep))
                if dataset in [None,'','imagenet']:
                    impath = os.path.join(imagenet_root,'images','val',imroot + '.JPEG')
                else:
                    impath = os.path.join(pascal_root,'VOC2007','JPEGImages',imroot + '.jpg')
                
                if os.environ.get('DBG_METRICS_VIZ',False)=='1':
                    assert False,'not tested with all file'
                    # imroot = 'ILSVRC2012_val_00002330'
                    # imroot = 'ILSVRC2012_val_00000005'
                    # imroot = 'ILSVRC2012_val_00001251' #test-tube
                
                    # imroot = 'ILSVRC2012_val_00002330' # boxer dog
                    # imroot = 'ILSVRC2012_val_00000138'
                    # imroot = 'ILSVRC2012_val_00002412' # strawberry
                    # imroot = 'ILSVRC2012_val_00004538' # dog in snow
                    imroot = 'ILSVRC2012_val_00002483' # zebra
                    
                    # skip_incorrect = False
                    impath = os.path.join(os.path.dirname(impath),f'{imroot}.JPEG')
                    # print(colorful.red_on_blue(f'setting imroot to {imroot},
                    #                            skip_incorrect to {skip_incorrect}'))                
                    print(colorful.red_on_blue(f'setting imroot to {imroot}'))                
                    im_save_dir = os.path.join(os.path.dirname(im_save_dir.rstrip(os.path.sep)),imroot)
                    
                #----------------------------------------------------------------

                image_metrics = {}
                imagedata = methoddata[imroot]
                # import ipdb;ipdb.set_trace()
                for obj in imagedata:
                    assert len(obj) == 1
                    classname = list(obj.keys())[0]
                    loaded = obj[classname]
                    target_id = loaded['target_id']
                    bbox_info,target_ids,classnames = get_gt(imroot,dataset,pascal_root=PASCAL_ROOT,imagenet_root=IMAGENET_ROOT)

                    #--------------------------------------------------------------------
                    if skip:
                        loaded_metrics = metrics_obj.load_one_object(imroot,classname,target_id)
                        
                        if loaded_metrics is not None:
                            
                            print('CONTINUING')
                            import ipdb;ipdb.set_trace()
                            continue
                        """
                        # assert False, ' not tested'
                        save_dir = os.path.join(settings.METRICS_DIR_librecam,f'{dataset}-{metricname}-{methodname}-{modelname}',imroot)
                        if USE_LZMA:
                            pklname = os.path.join(save_dir,f'{classname}{target_id}.xz')
                        else:
                            pklname = os.path.join(save_dir,f'{classname}{target_id}.pkl')
                        if os.path.exists(pklname):
                            try:
                                # assert False,'modify this to lzma'
                                # import ipdb; ipdb.set_trace()
                                if USE_LZMA:
                                    with lzma.open(pklname,'rb') as f:
                                        pass
                                else:
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
                        """
                    #--------------------------------------------------------------------
                    arrayix = loaded['arrayix']
                    saliency = loaded_saliency[arrayix]
                    if 'gpnn' in methodname:
                        assert False, 'this is not correct for gpnn-mycam'
                        if not os.environ.get('NO_MEDIAN_FILTERING',False):
                            print(colorful.yellow_on_cyan('median filtering gpnn-saliency '))
                            if not isinstance(saliency,torch.Tensor):
                                saliency = torch.tensor(saliency,device=device)
                                assert saliency.ndim == 4
                            saliency_pre = saliency
                            if median_k != 0:
                                import ipdb; ipdb.set_trace()
                                print(colorful.red("how is this working with a wrong padding"))
                                saliency = weighted_median_filter(saliency, 41, None, padding=median_k//2)
                            
                    if "numpy" in str(saliency.__class__):
                        saliency = torch.tensor(saliency,device=device)
                    if saliency.ndim == 4:
                        assert saliency.shape[0] == 1
                        saliency = saliency[0]
                    if saliency.ndim == 3:
                        if saliency.shape[0] > 1:
                            # 3 channel
                            # for integrated gradients
                            saliency = saliency.sum(dim=0,keepdim=True)
                    if saliency.ndim == 2:
                        saliency = saliency.unsqueeze(0)
                    input_tensor = get_image_tensor(impath,size=saliency.shape[-2:],dataset=dataset).to(device)
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
                        metric_data =run_chattopadhyay(input_tensor,saliency,model,target_id,dataset)
                        
                        # '''
                        if i > NBIG:
                            # ['mrf_scores', 'lrf_scores', 'metricname', 'mrf_probability_increase', 'mrf_increase_indicator', 'mrf_binary_mask', 'mrf_perturbed', 'lrf_probability_increase', 'lrf_increase_indicator', 'lrf_binary_mask', 'lrf_perturbed', 'modelname', 'methodname', 'classname', 'delete_percentiles', 'target_id', 'saliency']
                            for k in ['masked']:
                                if k in metric_data:
                                    del metric_data[k]
                        # import ipdb;ipdb.set_trace()
                        # '''                    
                    elif metricname == 'pointing_game':
                        # import ipdb;ipdb.set_trace()
                        metric_data = run_pointing_game(input_tensor,saliency,model,target_id,imroot,
                        bbox_info,dataset,
                        # IMAGENET_ROOT
                        )
                        if target_id in image_metrics:
                            import ipdb;ipdb.set_trace()
                            if image_metrics[target_id]['hit'] != True:
                                import ipdb;ipdb.set_trace()
                                if metric_data['hit'] == True:
                                    image_metrics[target_id]['hit'] = True
                            import ipdb;ipdb.set_trace()
                        else:
                            image_metrics[target_id] = metric_data
                            
                        metric_data = image_metrics[target_id]

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

                    if metricname == 'chattopadhyay':
                        metricnames_to_avg = [
                                    'normalized_drop_in_prob',
                                    'normalized_rise_in_prob',
                                    'increase_indicator',
                                    'decrease_indicator',
                                    'normalized_rise_in_prob2',

                        ]
                    elif metricname == 'perturbation':
                        import ipdb;ipdb.set_trace()
                    elif metricname == 'pointing_game':
                        metricnames_to_avg = ['hit']
                        # import ipdb;ipdb.set_trace()
                    update_running_avg(metric_data,running_averages,metricnames_to_avg)
                    print(colorful.red_on_yellow(running_averages))
                    # import ipdb;ipdb.set_trace()
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
                            # saliency = saliency,
                        )
                    )
                    
                    # continue
                    if True:
                        # print(target_id)
                        # import ipdb;ipdb.set_trace()
                        metrics_obj.dump(imroot,classname,target_id,metric_data)
                        # import ipdb;ipdb.set_trace()
                    if False:
                        from benchmark.benchmark_utils import create_dir
                        """
                        if median_k != 41 and 'gpnn' in  methodname:
                            save_dir = create_dir(os.path.join(f'{dataset}-{metricname}-{methodname}-median{median_k}-{modelname}',imroot),root_dir=settings.METRICS_DIR_librecam)
                        else:
                            save_dir = create_dir(os.path.join(f'{dataset}-{metricname}-{methodname}-{modelname}',imroot),root_dir=settings.METRICS_DIR_librecam)
                        if USE_LZMA:
                            savename = os.path.join(save_dir,f'{classname}{target_id}.xz')    
                        else:
                            savename = os.path.join(save_dir,f'{classname}{target_id}.pkl')

                        if False and 'relaxing requirement of 20 thresholds':
                            assert len(metric_data['mrf_per_threshold'].keys()) == 20
                        if os.environ.get('DBG_METRICS_VIZ',False) == '1':
                            import ipdb;ipdb.set_trace()
                        """
                        import ipdb; ipdb.set_trace()
                        dutils.cipdb('DBG_POINTING_GAME')
                        if os.environ.get('DBG_POINTING_GAME',False) != '1':
                            if USE_LZMA:
                                with lzma.open(savename,'wb') as f:
                                    pickle.dump(metric_data,f)
                            else:
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
    parser.add_argument('--dataset',type=str,default='voc')    
    parser.add_argument('--median_k',type=int,default=0)    
    # from libracam paper: page 5 figure 1: the comparison of attribution quality at threshold levels t ∈ [0, 1] with the increment of 0.1 from left to right in all plots. The
    # quality measures are evaluated with images multiplied with attribution maps where the attribution values less than t are set to the zero value.
    parser.add_argument('--delete_percentiles',type=float,nargs='*',default=[0,10,20, 30,40, 50,60, 70,80,90,100])
    args = parser.parse_args()
    # import pdb;pdb.set_trace()
    main(metrics=args.metrics,methodnames=args.methodnames,skip=args.skip,start=args.start,end=args.end,use_images_common_to=args.use_images_common_to,delete_percentiles=args.delete_percentiles,modelname=args.modelname,dataset = args.dataset,median_k = args.median_k)
    
if __name__ == '__main__':    
    outer_main()
# %%
