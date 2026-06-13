import dutils
from dutils import p46,p47,pause,pause2,tensor_to_numpy
import torch
import os
import numpy as np
import lzma
import pickle
import colorful
import tqdm
import argparse
# dutils.init()
import glob
import cam_benchmark.elp_masking as elp_masking
import cam_benchmark.road
# import torchvision
import wandb
from torchray.benchmark.models import get_model, get_transform
from torchray.benchmark.datasets import get_dataset
METRICS_ROOT_DIR= os.getenv('TORCHRAYMETRICS',"/data/bigfiles/other/metrics-torchray/")
# RESULTS_ROOT_DIR = dutils.hardcode(RESULTS_ROOT_DIR="/data/bigfiles/other/results-torchray")
RESULTS_ROOT_DIR = os.getenv('TORCHRAYRESULTS',"/data/bigfiles/other/metrics-torchray/")
#RESULTS_ROOT_DIR = dutils.hardcode(RESULTS_ROOT_DIR="/data/bigfiles/other/results-torchray/old_multi_results_mar4")
#RESULTS_ROOT_DIR2 = dutils.hardcode(RESULTS_ROOT_DIR="/data/bigfiles/other/results-torchray2")
def _get_binary_mask(mask, ratio_retained=None):
    if ratio_retained is None:
        if not( all([
            len( mask.unique()) in [1,2],
            mask.max() in [0.,1.],
            mask.min() in [0.,1.],
            ])):
            print(colorful.red('mask is not binary'))
            p46()
        mask_01 = mask
    else:
        #masked = dutils.hardcode(masked = torch.zeros_like(ref))
        sorted_mask_descending,argsort_descending = torch.sort(mask.flatten(),descending=False)
        dutils.note('this should be labeled _ascending right? confirm that same masks are made as in run_deletion_game')
        dutils.pause()
        cutoff_ix = int(len(sorted_mask_descending)*ratio_retained)
        cutoff_value = sorted_mask_descending[cutoff_ix]
        mask_01 = (mask >= cutoff_ix ).float()
        dutils.pause()
    return mask_01
def impute_where_0(ref,mask,ratio_retained=None,
perturbation = elp_masking.BLUR_PERTURBATION,
max_blur=20,
imputation='blur',
generator=None,
):
    mask_01 = _get_binary_mask(mask,ratio_retained)
    if imputation == 'blur':
        masked,perturbation = elp_masking.get_masked_input(
                                ref,
                                mask_01,
                                perturbation=perturbation,
                                num_levels=8,
                                # num_levels=12,
                                variant=elp_masking.PRESERVE_VARIANT,
                                max_blur=max_blur,
                                smooth=0)
    elif imputation == 'road':
        imputer = cam_benchmark.road.NoisyLinearImputer()
        assert ref.shape[0] == 1
        assert mask_01.shape[0] == 1
        masked = imputer(ref[0].cpu(),mask_01[0,0].cpu(), generator=generator)
        masked = masked[None,...]
        pass
    elif imputation == 'zero':
        # same as deletion2.py impute_where_0 'zero' branch
        masked = ref * mask_01
    else:
        p47()
    pause2('DBG_METRICS_MAR6')
    return masked,perturbation

def run_deletion_game(model,ref,target_id,
mask,ratios_retained,batch_size=dutils.TODO,
    perturbation = elp_masking.BLUR_PERTURBATION,
    max_blur=20,
    imputation ='blur',
    feat_layer = None,
    experiment = "class",
    feat_layer_name = None,
    return_deleted_images = False,
):
    device = ref.device
    ratios_retained = torch.tensor(ratios_retained,device=device)
    deleted_images = torch.zeros((len(ratios_retained),) + ref.shape[1:],device=device)
    
    ref_scores = model(ref)
    ref_probs = torch.softmax(ref_scores,dim=1)
    if ref_scores.ndim == 4:
        ref_scores = ref_scores.mean(dim=(-1,-2))
        ref_probs = ref_probs.mean(dim=(-1,-2))
    ref_probs = ref_probs[:,target_id]
    ref_scores = ref_scores[:,target_id]
    if feat_layer is not None:
        ref_feats = feat_layer.feats
        assert ref_feats.ndim == 2

    #=================================================================
    assert mask.ndim == 4
    assert mask.shape[:2] == (1,1)
    flat_mask = mask.flatten()
    sorted_mask_ascending,argsort_ascending = torch.sort(flat_mask,descending=False)
    _,unsort_ascending = torch.sort(argsort_ascending) 
    cutoff_ixs = (len(sorted_mask_ascending)*ratios_retained).long()

    if False and 'old style with cutoff value':
        cutoff_ixs = torch.clamp(cutoff_ixs,0,len(sorted_mask_ascending) - 1).long()
        cutoff_values = sorted_mask_ascending[cutoff_ixs]
        cutoff_values[ratios_retained==0] = cutoff_values[ratios_retained==0] - 1e-8
        mask_01 = (mask <= cutoff_values[:,None,None,None] ).float()
    if True and 'new style with cutoff ix':
        # p47()
        cutoff_ixs = torch.clamp(cutoff_ixs,0,len(sorted_mask_ascending)).long()
        dummy_range = torch.arange(flat_mask.shape[0],device=flat_mask.device)
        dummy_mask_01 = (dummy_range[None,:] < cutoff_ixs[:,None])
        pause2('DBG_METRICS_MAR6')
        flat_mask_01 = dummy_mask_01[:,unsort_ascending]
        mask_01 = flat_mask_01.view(cutoff_ixs.shape[0],*mask.shape[1:])
        

    if True or (ratios_retained == 0).any():
        assert mask_01[ratios_retained == 0].sum() == 0
    if True or (ratios_retained == 1).any():
        assert mask_01[ratios_retained == 1].sum() == np.prod(mask_01[0].shape)

    #=================================================================
    # if imputation == 'road':
    #     from concurrent.futures import ProcessPoolExecutor
    #     imputer = cam_benchmark.road.NoisyLinearImputer()
    #     #imputer.to(ref.device)
    #     assert ref.shape[0] == 1
    #     # assert mask_01.shape[0] == 1
    #     ref_ = ref.cpu()
    #     mask_01_ = mask_01.cpu()
    #     with ProcessPoolExecutor(max_workers=10) as e:
    #         curried_impute_where_0 = lambda mask_01_:imputer(ref_[0],mask_01_[0])
    #         masked = list(e.map(curried_impute_where_0,mask_01.unsqueeze(1)))
    #         print(os.getpid(),'masked done')
    #         # deleted_images = [pair[0] for pair in deleted_images_and_perturbation]
    #         deleted_images = torch.tensor(masked,device=ref.device,dtype=ref.dtype)
    # else:
    
    #................................................................
    if imputation == 'road':
        with dutils.Timer('concurrent-imputation') as timer:
            import concurrent.futures

            # Create per-index generators so threads don't race on shared RNG
            generators = [torch.Generator().manual_seed(i) for i in range(len(ratios_retained))]

            def process_mask(i,mask_01_i, ratio_retained, gen):
                pause2('DBG_METRICS_MAR6')
                deleted_ref, perturbation_result = impute_where_0(
                    ref,
                    mask_01_i,
                    ratio_retained=None,
                    perturbation=perturbation,
                    max_blur=max_blur,
                    imputation=imputation,
                    generator=gen,
                )
                return i, deleted_ref, perturbation_result

            with dutils.MaybeThreadPoolExecutor() as executor:
                futures = [executor.submit(process_mask, i, mask_01[i:i+1], ratio_retained, generators[i]) for i, ratio_retained in enumerate(ratios_retained)]

                for future in concurrent.futures.as_completed(futures):
                    i, deleted_ref_result, perturbation_result = future.result()
                    deleted_images[i:i+1] = deleted_ref_result
                    perturbation = perturbation_result  # If `perturbation` must be shared, this line may need rethinking.
    
    else:
        with dutils.Timer('looped-imputation') as timer0:
            for i,ratio_retained in enumerate(ratios_retained):
                #dutils.img_save(mask_01[i],f'mask_01_{mask_01[i].sum()}.png')
                pause2('DBG_METRICS_MAR6')
                gen = torch.Generator().manual_seed(i) if imputation == 'road' else None
                deleted_ref, perturbation= impute_where_0(ref,mask_01[i:i+1],ratio_retained=None,perturbation=perturbation,max_blur=max_blur,imputation=imputation,generator=gen)
                deleted_images[i:i+1] = deleted_ref
    #................................................................
    # for yy in [0,-1]:dutils.img_save(mask_01[yy],f'mask01_{yy}.png',vmin=0,vmax=1,cmap='gray',use_matplotlib=False)
    # p47()
    #dutils.img_save(deleted_images[i:i+1],'deleted.png')
    #dutils.pause()
    assert deleted_images.shape[0] <= batch_size, 'implement batched forward'
    
    with torch.inference_mode():
        scores = model(deleted_images)
    if True:
        probs = torch.softmax(scores,dim=1)
        if scores.ndim == 4:
            scores = scores.mean(dim=(-1,-2))
            probs = probs.mean(dim=(-1,-2))
        probs = probs[:,target_id]
        scores = scores[:,target_id]
        #dutils.note('check broadcasting of probs')
        #dutils.pause();
        assert probs.ndim == 1
        diff_in_probs = probs - ref_probs
        '''
        # (1,20,1,1)
        # (1,3,300,500) --> (1,20,2,5)
        # (1,1000) 
        '''
        # model(deleted_ref)
        # ref = dutils.hardcode(masked = torch.zeros_like(ref))
        probs = tensor_to_numpy(probs)
        diff_in_probs = tensor_to_numpy(diff_in_probs)
        ref_probs = tensor_to_numpy(ref_probs)
        #p47()
        results = dict(
            probs = probs,
            ref_probs = ref_probs,
            diff_in_probs = diff_in_probs,
            ratios = ratios_retained,
            imputation = imputation,
        )
        if return_deleted_images:
            # opt-in (analysis/visualization callers only); not stored in metrics xz
            results['deleted_images'] = tensor_to_numpy(deleted_images)
    if feat_layer is not None:
        feats = feat_layer.feats
        assert feats.ndim == 2
        feat_distance = ((feats - ref_feats)**2).sum(dim=-1)
        feat_distance = tensor_to_numpy(feat_distance)
        results['feat_distance'] = feat_distance
    return results

def add_to_results_xz(method=dutils.TODO,
            arch=dutils.TODO,
            dataset=dutils.TODO,
            results_root_dir=dutils.TODO,
            imputation = 'blur',
            **kwargs,
):  
    '''
    add the insertion and deletion metrics to the results xz files
    '''
    #p45()
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    methoddir = os.path.join(results_root_dir,f'{dataset}-{method}-{arch}')
    resultpattern = os.path.join(methoddir,'*','*.xz') 
    resultsxzfiles = glob.glob(resultpattern)

    if imputation == 'blur':
        metrics_dir = os.path.join(METRICS_ROOT_DIR,"deletion",f"{dataset}-{method}-{arch}")
    else:
        metrics_dir = os.path.join(METRICS_ROOT_DIR,"deletion",f"{dataset}-{method}-{arch}-{imputation}")
    metricpattern = os.path.join(metrics_dir,'*','*.xz') 
    metricsxzfiles = glob.glob(metricpattern)
    # xzfiles = list(sorted(glob.glob(os.path.join(methoddir,'*','*.xz'))))
    small_xzpath = os.path.join(RESULTS_ROOT_DIR,"mnist-grad_cam-resnet8/0/77.xz")
#.............................................................
    if False:
        methoddir_new_results = os.path.join(results_root_dir,f'{dataset}-{method}-{arch}_new_results')
        with lzma.open(small_xzpath,'rb') as f:
            small_loaded = pickle.load(f)

#.............................................................
    for resultxzfile,metricxzfile in tqdm.tqdm(dutils.trunciter(zip(resultsxzfiles,metricsxzfiles),enabled=False,max_iter=10)):
        print(resultxzfile)
        print(metricxzfile)
        with lzma.open(resultxzfile,'rb') as f:
            result = pickle.load(f) 
        with lzma.open(metricxzfile,'rb') as f:
            metric = pickle.load(f) 
        result['insertion'] = metric['insertion']
        result['deletion'] = metric['deletion']
        stub = os.path.basename(resultxzfile)
        imroot = os.path.basename(os.path.dirname(resultxzfile))
        new_resultsfile = os.path.join(methoddir,imroot,stub)
        #p46()
        with lzma.open(new_resultsfile,'wb') as f:
            pickle.dump(result,f)
        with lzma.open(new_resultsfile,'rb') as f:
            reloaded = pickle.load(f)
#.............................................................
        if False:
            assert set(reloaded.keys()).intersection(set(small_loaded.keys())) == set(small_loaded.keys())
            assert set(reloaded['insertion'].keys()) == set(small_loaded['insertion'].keys())
            assert set(reloaded['deletion'].keys()) == set(small_loaded['deletion'].keys())
#.............................................................
        #dutils.pause()
def get_data(method,dataset):
    if dataset in ['voc_2007','coco']:
        if method == "rise":
            input_size = (224, 224)
        else:
            input_size = 224        
    elif dataset in ['imagenet-5000']:
        input_size = 224
    elif dataset in ['cifar-10','cifar-100','mnist']:
        input_size = (32,32)
    else:
        dutils.pause()
    #subset = 'test'
    if dataset == 'voc_2007':
        subset = 'test'
    elif dataset == 'coco':
        subset = 'val2014'
    elif dataset == 'imagenet-5000':
        subset = 'val'
    elif dataset in ['cifar-10','cifar-100']:
        subset = 'val'
    elif dataset in ['mnist']:
        subset = 'val'
    else:
        assert False
    
    transform = get_transform(size=input_size,
                                dataset=dataset)
    
    data = get_dataset(name=dataset,
                        subset=subset,
                        transform=transform,
                        download=False,
                        limiter=None)   
    return data

def _find_image_ix(data,xzfile):    
    found = False
    imroot = os.path.basename(os.path.dirname(xzfile))
    #dutils.pause()
    for imix,impath in enumerate(data.images):
        if imroot in impath:
            found = True
            break
    assert found
    return imix,imroot
def run_on_xzfile(xzfile,model,ratios_retained,imputation,max_blur,batch_size,feat_layer,data,device):        
    print(xzfile)
    xzfile = os.path.abspath(xzfile)
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    imix,imroot = _find_image_ix(data,xzfile)
    #imix = 0
    ref,y = data[imix]
    assert ref.ndim == 3,'expecting ref to be 3d'
    ref = ref[None]
    ref = ref.to(device)
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    try:
        with lzma.open(xzfile,'rb') as f:
            loaded = pickle.load(f)
    except Exception as e:
        print(f'{xzfile} is corrupt, error: {e}')
        return None
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    class_id =loaded['class_id']
    saliency = loaded['saliency']
    if saliency.max() > 0:
        saliency = saliency/saliency.max()
    assert saliency.max() <= 1.
    assert saliency.min() >= 0
    class_name = loaded['class_name']
    assert saliency.ndim == 4
    saliency = torch.tensor(saliency,device=ref.device)
    saliency = torch.nn.functional.interpolate(saliency,ref.shape[-2:],mode="bilinear")
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    results_deletion = run_deletion_game(model,ref,class_id,
        saliency,ratios_retained,batch_size=batch_size,max_blur=max_blur,imputation=imputation, feat_layer = feat_layer)
    results_insertion = run_deletion_game(model,ref,class_id,
        1-saliency,ratios_retained,batch_size=batch_size,max_blur=max_blur,imputation=imputation, feat_layer = feat_layer)
    return dict(results_insertion = results_insertion, results_deletion = results_deletion,imroot=imroot,class_name=class_name,class_id=class_id,method=loaded['methodname'])

def _get_save_dir(dataset,method,arch,imputation):
    if imputation == 'blur':
        save_dir = os.path.join(METRICS_ROOT_DIR,"deletion",f"{dataset}-{method}-{arch}")
    else:
        save_dir = os.path.join(METRICS_ROOT_DIR,"deletion",f"{dataset}-{method}-{arch}-{imputation}")
    return save_dir
def _get_results_xzfiles(results_root_dir,method,arch,dataset,start=0,ntodo=-1):
    methoddir = os.path.join(results_root_dir,f'{dataset}-{method}-{arch}')  
    xzfiles = glob.glob(os.path.join(methoddir,'*','*.xz'))
    assert len(xzfiles), f'xzfiles is empty, {methoddir}'
    return xzfiles
def _get_feat_layer(model,feat_layer):
    if feat_layer is None:
        return None
    if isinstance(feat_layer,str):
        feat_layer = getattr(model,feat_layer)
    if hasattr(feat_layer,'layer'):
        feat_layer = feat_layer.layer
    return feat_layer

def run(method=dutils.TODO,dataset=dutils.TODO,arch=dutils.TODO,
results_root_dir=dutils.TODO,
save_root_dir=dutils.TODO,
batch_size = dutils.TODO,
max_blur = dutils.TODO,
imputation='blur',
ratios = dutils.TODO,
start = 0,
feat_layer = None,
ntodo=-1,
device = dutils.hardcode(device="cuda"),
overwrite=False,
**ignore
):
    # renaming here as input argument is named ratios
    ratios_retained = ratios
    ratios_retained = np.array(ratios_retained)
    if not np.allclose((np.sort(ratios_retained ) - np.sort(1-ratios_retained)),np.zeros(ratios_retained.shape) ):
        print(colorful.red('ratios_retained and 1-ratios_retained are not equal'))
        dutils.pause()
    if device == 'cuda':
        if not torch.cuda.is_available():
            device = 'cpu'
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    model = get_model(
            arch=arch,
            dataset=dataset,
            convert_to_fully_convolutional=True,
        )

    model.to(device)
    model.eval()
    data = get_data(method,dataset)
    feat_layer = _get_feat_layer(model,feat_layer)
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    save_dir = _get_save_dir(dataset,method,arch,imputation)
    xzfiles = _get_results_xzfiles(results_root_dir,method,arch,dataset,start=0,ntodo=-1)
    xzfiles = xzfiles[start:]
    if ntodo > 0:
        xzfiles = xzfiles[:ntodo]
    running_scores = {'insertion':[],'deletion':[]}
    for xzfile in tqdm.tqdm(dutils.trunciter(xzfiles,enabled=False,max_iter=10)):
        # check the metrics file exists
        metrics_savepath=get_metrics_savepath_from_results_xzpath(xzfile,save_dir)
        if not overwrite:
            if os.path.exists(metrics_savepath):
                print(colorful.red(f'{metrics_savepath} already exists'))
                continue
        # p46()

        deletion_results_ = run_on_xzfile(xzfile,model,ratios_retained,imputation,max_blur,batch_size,feat_layer,data,device)
        
        
        metrics = dict( 
                insertion = deletion_results_['results_insertion'],
                deletion= deletion_results_['results_deletion'],
                arch = arch,
                dataset = dataset,
                method = deletion_results_['method'],
                imroot = deletion_results_['imroot'],
                class_name = deletion_results_['class_name'],
                class_id = deletion_results_['class_id'],
            )
        save_metrics(metrics,xzfile,save_dir,running_scores)
    '''
    <parent-directory>/000001/dog11.xz
    <parent-directory>/000001/person14.xz
    <parent-directory>/000002/car6.xz
    '''
    pass

def get_metrics_savepath_from_results_xzpath(xzfile,save_dir):
    classname_classid_xz  = os.path.basename(xzfile)
    imroot = os.path.basename(os.path.dirname(xzfile))
    os.makedirs(os.path.join(save_dir,imroot),exist_ok=True)
    
    savepath = os.path.join(save_dir,imroot,classname_classid_xz)

    return savepath

def save_metrics(results,xzfile,save_dir,running_scores):
    running_scores['insertion'].append(results['insertion']['probs'])
    running_scores['deletion'].append(results['deletion']['probs'])

    savepath=get_metrics_savepath_from_results_xzpath(xzfile,save_dir)
    print(savepath)
    with lzma.open(savepath,'wb') as f:
        pickle.dump(results,f)
    #========= WANDB LOGGING ==============
    wandb.log(dict(running_insertion_score = np.array(running_scores['insertion']).mean()),commit=False)
    wandb.log(dict(running_deletion_score = np.array(running_scores['deletion']).mean()),commit=False)
    wandb.log(dict(xzfile=xzfile),commit=False)
    wandb.log({})    

def _init_wandb(args):
    if wandb.run is None:
       wandb.init(project=f"deletion-{args.dataset}-{args.method}-{args.arch}-{args.imputation}",config=dict(dataset=args.dataset,method=args.method,arch=args.arch,imputation=args.imputation,ratios=args.ratios))
def get_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--method",type=str)
    parser.add_argument("--arch",type=str)
    parser.add_argument("--dataset",type=str)
    parser.add_argument("--ratios",type=float,nargs="*")
    parser.add_argument("--results_root_dir",type=str,default=RESULTS_ROOT_DIR)
    parser.add_argument("--save_root_dir",type=str,default=METRICS_ROOT_DIR)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--max_blur",type=float,default=20)
    parser.add_argument("--imputation",type=str,default='blur',choices=['blur','road'])
    parser.add_argument("--add-to-results-xz",type=lambda t:t.lower() == 'true',default=False,dest="add_to_results_xz")
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--ntodo",type=int,default=-1)
    parser.add_argument("--overwrite",type=lambda t:t.lower() == 'true',default=False)
    args = parser.parse_args()
    return args
def main():
    args = get_args()
    _init_wandb(args)
   
    if not args.add_to_results_xz:
        run(**vars(args))
    else:
        #p46()
        add_to_results_xz(**vars(args))

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method("spawn")  # Ensures safe multiprocessing
    import torch  # Import after setting spawn mode
    main()
    pass
