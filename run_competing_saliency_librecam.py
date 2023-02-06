#%%
# import os
# os.chdir('/root/evaluate-saliency-4/jigsaw')    
import register_ipdb
import dutils
import torch
import torchvision
import numpy as np
from benchmark.benchmark_utils import ChangeDir,AddPath
import skimage.io
from PIL import Image
import pickle
import glob
import os
import colorful
# '/root/evaluate-saliency-4/jigsaw/imagenet'


#from pydoc import importfile
#%% gradcam
from benchmark.benchmark_utils import create_im_save_dir
import os
import pickle
from benchmark import settings
import dutils
import builtins
builtins.dutils = dutils
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
DISABLED = False
def get_image_tensor(impath,size=(224,)):
    from cnn import get_vgg_transform
    vgg_transform = get_vgg_transform(size)
    im_ = skimage.io.imread(impath)
    if im_.ndim == 2:
        im_ = np.concatenate([im_[...,None],im_[...,None],im_[...,None]],axis=-1)
    im_pil = Image.fromarray(im_)
    ref = vgg_transform(im_pil).unsqueeze(0)
    return ref
##############################################################################################
##############################################################################################
# GPNN-GRADCAM
from gpnn_gradcam import main as gpnn_gradcam_main
from trivial_gradcam import main as trivial_gradcam_main
from hacky_scorecam import main as hacky_scorecam_main
from hacky_relevancecam import main as hacky_relevancecam_main
import torch
IMAGENET_ROOT = "/root/bigfiles/dataset/imagenet"
def denormalize_tensor(t,vgg_mean=[0.485, 0.456, 0.406],
                     vgg_std=[0.229, 0.224, 0.225]):
    device = t.device
    # out = (t - torch.tensor(vgg_mean).to(device)[None,:,None,None])/torch.tensor(vgg_std).to(device)[None,:,None,None]
    out = (t * torch.tensor(vgg_std).to(device)[None,:,None,None]) + torch.tensor(vgg_mean).to(device)[None,:,None,None]
    return out
def run_trivial_cam(model,ref,target_id,
                    HFLIP=False,NOISE=False,
                    base_method=None,
                    device=None):
    # assert False
    method = 'trivialcam'
    method += f'-{base_method}'
    if HFLIP:
        method += '-hflip'
    if NOISE:
        method += '-noise'
    ref = denormalize_tensor(ref)
    print(ref.__class__)
    from saliency_for_scorecam import get_cams
    avg_saliency,scores,probs = get_cams(ref,target_id,method=method,gradcam_scale_cams=None,cnn=model)    
    return {
                        'saliency':avg_saliency,
                        'target_id':target_id,
                        'method':method,
                        #'cam0':cam0,
                        'ref':tensor_to_numpy(ref.permute(0,2,3,1))[0],
                        }

def run_relevance_cam(model,ref,target_id,base_method='gradcam',device=None):
    # return
    ref = denormalize_tensor(ref)
    print(ref.__class__)
    from saliency_for_scorecam import get_cams
    avg_saliency,scores,probs = get_cams(ref,target_id,method='relevancecam',gradcam_scale_cams=None,cnn=model)    
    print('TODO:save original image, cam saliency and avg saliency')
    return {
                        'saliency':avg_saliency,
                        'target_id':target_id,
                        'method':'relevancecam',
                        #'cam0':cam0,
                        'ref':tensor_to_numpy(ref.permute(0,2,3,1))[0],
                        }
    return saliency_data

def run_libra_cam(model,ref,target_id,base_method='gradcam',device=None):
    # return
    ref = denormalize_tensor(ref)
    print(ref.__class__)
    from saliency_for_scorecam import get_cams
    avg_saliency,scores,probs = get_cams(ref,target_id,method='libracam',gradcam_scale_cams=None,cnn=model)    
    print('TODO:save original image, cam saliency and avg saliency')
    return {
                        'saliency':avg_saliency,
                        'target_id':target_id,
                        'method':'relevancecam',
                        #'cam0':cam0,
                        'ref':tensor_to_numpy(ref.permute(0,2,3,1))[0],
                        }
    return saliency_data
def run_score_cam(model,ref,target_id,base_method='gradcam',device=None):
    from saliency_for_scorecam import get_cams
    avg_saliency,scores,probs = get_cams(ref,target_id,method='scorecam',gradcam_scale_cams=None,cnn=model)
    print('TODO:save original image, cam saliency and avg saliency')
    return {
            'saliency':tensor_to_numpy(avg_saliency),
            'target_id':target_id,
            'method':'scorecam',
            #'cam0':cam0,
            'ref':tensor_to_numpy(ref.permute(0,2,3,1))[0],
            }

def run_gradcam(model,ref,target_id,base_method='gradcam',device=None):
    from saliency_for_scorecam import get_cams
    avg_saliency,scores,probs = get_cams(ref,target_id,method='gradcam',gradcam_scale_cams=None,cnn=model)
    print('TODO:save original image, cam saliency and avg saliency')
    return {
            'saliency':avg_saliency,
            'target_id':target_id,
            'method':'gradcam',
            #'cam0':cam0,
            'ref':tensor_to_numpy(ref.permute(0,2,3,1))[0],
            }
def run_gradcampp(model,ref,target_id,base_method='gradcam',device=None):
    from saliency_for_scorecam import get_cams
    avg_saliency,scores,probs = get_cams(ref,target_id,method='gradcampp',gradcam_scale_cams=None,cnn=model)
    print('TODO:save original image, cam saliency and avg saliency')
    return {
            'saliency':avg_saliency,
            'target_id':target_id,
            'method':'gradcampp',
            #'cam0':cam0,
            'ref':tensor_to_numpy(ref.permute(0,2,3,1))[0],
            }
# def run_trivial_cam(model,ref,target_id,base_method='gradcam',device=None):
#     # return
#     assert False, 'this will actually run score_cam, correct this'
#     ref = denormalize_tensor(ref)
#     print(ref.__class__)
#     import time;time.sleep(5)
#     config = {
#         'out_dir':'gpnn-gradcam/output',
#         'iters':10,
#         # 'iters':1,#10
#         'coarse_dim':14,#
#         # 'coarse_dim':28,
#         # 'coarse_dim':100,#
#         'out_size':0,
#         'patch_size':7,
#         # 'patch_size':15,
#         'stride':1,
#         'pyramid_ratio':4/3,
#         # 'pyramid_ratio':2,
#         'faiss':True,
#         # 'faiss':False,
#         'no_cuda':False,
#         #---------------------------------------------
#         'in':None,
#         'sigma':4*0.75,
#         # 'sigma':0.3*0.75,
#         'alpha':0.005,
#         'task':'random_sample',
#         #---------------------------------------------
#         # 'input_img':original_imname,
#         'input_img':ref,
#         'batch_size':10,
#         #---------------------------------------------
#         'implementation':'gpnn',#'efficient-gpnn','gpnn'
#         'init_from':'zeros',#'zeros','target'
#         'keys_type':'single-resolution',#'multi-resolution','single-resolution'
#         #---------------------------------------------
#         'use_pca':True,
#         'n_pca_components':30,
#         #---------------------------------------------
#         'patch_aggregation':'distance-weighted',#'uniform','distance-weighted','median'
#         'imagenet_target':target_id,
#         'n_super_iters':10,
#         #---------------------------------------------
#         'device':device,        
#         'index_type':'ivf',
#         'base_method':base_method,
#         }    
#     more_returns = {}
#     # avg_saliency = trivial_gradcam_main(config,cnn=model,save_results=False,more_returns=more_returns)
#     print('hacking trivial cam as hacky_score_cam')
#     time.sleep(5)
#     avg_saliency = hacky_scorecam_main(config,cnn=model,save_results=False,more_returns=more_returns)
#     print('TODO:save original image, cam saliency and avg saliency')
#     return {
#                         'saliency':avg_saliency,
#                         'target_id':target_id,
#                         'method':'trivial-gradcam',
#                         #'cam0':cam0,
#                         'ref':tensor_to_numpy(ref.permute(0,2,3,1))[0],
#                         }
#     return saliency_data
def run_gpnn_gradcam(model,ref,target_id,base_method='gradcam',aggregation=None,device=None):
    # return {}
    ref = denormalize_tensor(ref)
    print(ref.__class__)
    import time;time.sleep(5)
    config = {
        'out_dir':'gpnn-gradcam/output',
        'iters':10,
        # 'iters':1,#10
        'coarse_dim':14,#
        # 'coarse_dim':28,
        # 'coarse_dim':100,#
        'out_size':0,
        'patch_size':7,
        # 'patch_size':15,
        'stride':1,
        'pyramid_ratio':4/3,
        # 'pyramid_ratio':2,
        'faiss':True,
        # 'faiss':False,
        'no_cuda':False,
        #---------------------------------------------
        'in':None,
        'sigma':4*0.75,
        # 'sigma':0.3*0.75,
        'alpha':0.005,
        'task':'random_sample',
        #---------------------------------------------
        # 'input_img':original_imname,
        'input_img':ref,
        'batch_size':10,
        #---------------------------------------------
        'implementation':'gpnn',#'efficient-gpnn','gpnn'
        'init_from':'zeros',#'zeros','target'
        'keys_type':'single-resolution',#'multi-resolution','single-resolution'
        #---------------------------------------------
        'use_pca':True,
        'n_pca_components':30,
        #---------------------------------------------
        'patch_aggregation':aggregation,#'uniform','distance-weighted','median'
        'imagenet_target':target_id,
        'n_super_iters':10,
        #---------------------------------------------
        'device':device,        
        'index_type':'ivf',
        'base_method':base_method,
        }    
    
    print(colorful.red(f'setting super_iters to {config["n_super_iters"]}'))
    print(colorful.red(f'setting iters to {config["iters"]}'))
    print(colorful.red(f'setting batch_size to {config["batch_size"]}'))
    import time;time.sleep(2)
    avg_saliency = gpnn_gradcam_main(config,cnn=model,save_results=False)
    return {
                        'saliency':avg_saliency,
                        'target_id':target_id,
                        'method':'gpnn-gradcam',
                        }
    return saliency_data
##############################################################################################
##############################################################################################
def main(    
    methodnames = None,
    skip=False,
    start = 0,
    end=None,
    images_common_to=None,
    modelname = 'vgg16',
    device = 'cuda',
    other_info={}):
    print('check modelname')
    # import ipdb;ipdb.set_trace()
    
    imagenet_root = IMAGENET_ROOT 
    # model = torchvision.models.__dict__[modelname](pretrained=True).to(device)
    def get_model(modelname,device=None):
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
    image_paths = sorted(glob.glob(os.path.join(imagenet_root,'images','val','*.JPEG')))
    size=(224,224)
    image_paths = list((image_paths))
    # import pdb;pdb.set_trace()
    if end is not None:
        image_paths = image_paths[:end]
    image_paths = image_paths[start:]
    print(start,end)
    # assert False
    ##############################################################################################
    # print('before methodnames');import pdb;pdb.set_trace()
    for methodname in methodnames:
        if images_common_to is not None:
            hack_for_im_save_dirs = get_hack_for_im_save_dirs(images_common_to,modelname)
            print('#'*50,'\n','hack_for_im_save_dirs','\n','#'*50) 
            # import pdb;pdb.set_trace()   
            image_paths = hack_for_im_save_dirs(image_paths)                
        for i,impath in enumerate(image_paths):
            if os.environ.get('BREAK_COMPETING',None):
                BREAK_COMPETING = int(os.environ['BREAK_COMPETING'])
                if i > BREAK_COMPETING:
                    print(colorful.red(f'breaking out of run_competing_librecam at {i}'))
                    import time;time.sleep(2)
                    break            
            # if 'skip2' and (i < 1750):
            #     continue
        # for impath in image_paths[1750:]:
            # if impath.endswith('ILSVRC2012_val_00000005.JPEG'):
            #     import pdb;pdb.set_trace()

            imroot = os.path.basename(impath).split('.')[0]
        #     bboxpath = os.path.join(imagenet_root,'bboxes','val',imroot + '.xml')    
            import imagenet_localization_parser
            from benchmark.synset_utils import get_synset_id,synset_id_to_imagenet_class_ix
            bbox_info = imagenet_localization_parser.get_voc_label(
                root_dir = os.path.join(imagenet_root,'bboxes','val'),
                x = imroot)
        #     print(bbox_info)
            synset_id = bbox_info['annotation']['object'][0]['name']
        #     print(synset_id)
            target_id = synset_id_to_imagenet_class_ix(synset_id)
        #     print(target_id)
            import imagenet_synsets
            classname = imagenet_synsets.synsets[target_id]['label']
            classname = classname.split(',')[0]
            if False:
                classname = '_'.join(classname)
            else:
                classname = classname.replace(' ','_')
            #-------------------------------------------------------------------
            if skip:
                # assert False,'only skips if integrated-gradients'
                # from benchmark import settings
                imroot = os.path.split(impath)[-1].split('.')[0]
                im_save_dir = os.path.join(settings.RESULTS_DIR_librecam,f'{methodname}-{modelname}',imroot)
                pklname = os.path.join(im_save_dir,f'{classname}{target_id}.pkl')
                # import pdb;pdb.set_trace()
                if os.path.exists(pklname):
                    # import pdb;pdb.set_trace()
                    try:
                        with open(pklname,'rb') as f:
                            loaded = pickle.load(f)
                            saliency = loaded['saliency']
                            assert not np.isnan(saliency).any()
                        print(f'{pklname} exists, skipping')
                        continue
                    except EOFError as e:
                        print(f'{pklname} corrupt,overwriting')
                    except AssertionError as e:
                        print(f'nan found{pklname}, overwriting')
            #-------------------------------------------------------------------

            ref = get_image_tensor(impath,size=size).to(device)


            '''
            from pytorch_grad_cam.metrics.road import ROADMostRelevantFirstAverage,ROADLeastRelevantFirstAverage
            '''
            #######################################################################################
            #.......................................................
#            if methodname in ['gpnn-gradcam','gpnn-gradcampp']:
#                # print('before jigsaw');
#                # import ipdb;ipdb.set_trace()
#                if methodname == 'gpnn-gradcam':
#                    base_method = 'gradcam'
#                elif methodname == 'gpnn-gradcampp':
#                    base_method = 'gradcampp'
#                else:
#                    assert False,f'{methodname} not recognized'
#                saliency_data =run_gpnn_gradcam(model,ref,target_id,base_method=base_method,device=device)
            if methodname in ['gpnn-gradcam','gpnn-gradcampp','gpnn-gradcam-uniform','gpnn-gradcampp-uniform']:
                # print('before jigsaw');
                parts = methodname.split('-')
                aggregation = 'distance-weighted'
                base_method = parts[1]
                if len(parts) > 2:
                    aggregation = parts[2]                    
                # if False:
                #     if TODO:
                #     # if methodname == 'gpnn-gradcam':
                #         base_method = 'gradcam'
                #     # elif methodname == 'gpnn-gradcampp':
                #         base_method = 'gradcampp'
                #     else:
                #         assert False,f'{methodname} not recognized'
                # import ipdb;ipdb.set_trace()
                saliency_data =run_gpnn_gradcam(model,ref,target_id,base_method=base_method,aggregation=aggregation,device=device)
            #.......................................................          
            elif methodname == 'gradcam':
                saliency_data = run_gradcam(model,ref,target_id,base_method='gradcam',
                device=device)
            elif methodname == 'gradcampp':
                saliency_data = run_gradcampp(model,ref,target_id,base_method='gradcam',device=device)                
            elif methodname == 'scorecam':
                saliency_data = run_score_cam(model,ref,target_id,base_method='scorecam',device=device)
            elif methodname == 'relevancecam':
                saliency_data = run_relevance_cam(model,ref,target_id,base_method='relevancecam',device=device)                
            elif methodname == 'libracam':
                saliency_data = run_libra_cam(model,ref,target_id,base_method='libracam',device=device)                   
            elif 'trivial' in methodname:
                if 'gradcampp' in methodname:
                    base_method = 'gradcampp'
                else:
                    base_method = 'gradcam'
                HFLIP=False
                NOISE=False
                if 'hflip' in methodname:
                    HFLIP=True
                if 'noise' in methodname:
                    NOISE=True
                    
                saliency_data = run_trivial_cam(model,ref,target_id,HFLIP=HFLIP,NOISE=NOISE,
                                                base_method=base_method,device=device)                 

            else:
                assert False,f'{methodname} not recognized'
            #.......................................................          
            assert not np.isnan(saliency_data['saliency']).any()
            saliency_data.update(
                dict(
                    modelname = modelname,
                    impath = impath,
                    classname = classname
                )
            )            
            im_save_dir = create_im_save_dir(experiment_name=f'{methodname}-{modelname}',root_dir=settings.RESULTS_DIR_librecam,impath=impath)  
            # import ipdb;ipdb.set_trace()
            savename = os.path.join(im_save_dir,f'{classname}{target_id}.pkl')
            if False:
                with open(savename,'wb') as f:
                    pickle.dump(saliency_data,f)        

            else:
                import threading
                def save_data(saliency_data, savename):
                    with open(savename, 'wb') as f:
                        pickle.dump(saliency_data, f)

                t = threading.Thread(target=save_data, args=(saliency_data, savename))
                t.start()
            

    ##############################################################################################         
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--methodnames',nargs='*',default = [  
                    # 'gradcam',
                    # 'smoothgrad',
                    # 'fullgrad',
                    # 'integrated-gradients',
                    # 'gradients',
                    #'inputXgradients',
                    # 'jigsaw-saliency',
                    # 'elp',
                    # 'trivial-gradcam-hflip',
                    # 'trivial-gradcam-noise',
                    # 'trivial-gradcam-hlip-noise',                    
                    ])
    parser.add_argument('--skip',type=lambda v: (v.lower()=='true'),default=False)
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--end',type=int,default=None)
    parser.add_argument('--device',type=str,default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--images-common-to',type=str,default=None)
    parser.add_argument('--modelname',type=str,default='vgg16')
    args = parser.parse_args()
    # import pdb;pdb.set_trace()
    # main(methodnames=args.methodnames,skip=args.skip)    
    DEBUG = False
    
    main(skip=args.skip,start=args.start,end=args.end,device=args.device,
    images_common_to=args.images_common_to,methodnames=args.methodnames,modelname=args.modelname)
       
