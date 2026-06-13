# import os
# os.environ('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse

# dutils.init()
import glob
import lzma
import os
import pickle

import colorful
import dutils
import numpy as np
import torch
import torchvision
import tqdm
from dutils import p46, p47, pause, pause2, tensor_to_numpy
from torchray.benchmark.datasets import get_dataset
from torchray.benchmark.models import get_model, get_transform
import traceback
# from multithresh_saliency.run_self_saliency import get_layernames
import cam_benchmark.deletion
import cam_benchmark.elp_masking as elp_masking
import cam_benchmark.road
import wandb
from cam_benchmark.cnn_utils import register_feat_hook

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

METRICS_ROOT_DIR="/data/bigfiles/other/metrics-torchray/"
RESULTS_ROOT_DIR = dutils.hardcode(RESULTS_ROOT_DIR="/data/bigfiles/other/results-torchray")
#RESULTS_ROOT_DIR = dutils.hardcode(RESULTS_ROOT_DIR="/data/bigfiles/other/results-torchray/old_multi_results_mar4")
#RESULTS_ROOT_DIR2 = dutils.hardcode(RESULTS_ROOT_DIR="/data/bigfiles/other/results-torchray2")
def impute_where_0(ref,mask,ratio_retained=None,
perturbation = elp_masking.BLUR_PERTURBATION,
max_blur=20,
imputation='blur',
generator=None,
):
    if ratio_retained is None:
        if not( all([
            len( mask.unique()) in [1,2],
            mask.max() in [0.,1.],
            mask.min() in [0.,1.],
            ])):
            dutils.pause()
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
        #imputer.to(ref.device)
        assert ref.shape[0] == 1
        assert mask_01.shape[0] == 1
        masked = imputer(ref[0].cpu(),mask_01[0,0].cpu(), generator=generator)
        masked = masked[None,...]
        #p47()
        pass
    elif imputation == 'zero':
        masked = ref * mask_01

    else:
        print(f'unknown imputation method {imputation}')
        p47()
    #dutils.img_save(masked,f'masked_{mask.sum()}.png')
    pause2('DBG_METRICS_MAR6')
    return masked,perturbation

def run_deletion_game(model,ref,target_id,
mask,ratios_retained,batch_size=dutils.TODO,
    perturbation = elp_masking.BLUR_PERTURBATION,
    max_blur=20,
    imputation ='blur',
    feat_layer = None,
    experiment = ['class','feat','channel'][0],
    feat_layer_name = None,
):
    if experiment == 'channel':
        return run_channel_deletion_game(model,ref,target_id,
            mask,ratios_retained,batch_size=batch_size,
                perturbation = perturbation,
                max_blur=max_blur,
                imputation =imputation,
                feat_layer = feat_layer,
                experiment = experiment,
                feat_layer_name = feat_layer_name,
            )
    return cam_benchmark.deletion.run_deletion_game(
        model,ref,target_id,
            mask,ratios_retained,batch_size=batch_size,
                perturbation = perturbation,
                max_blur=max_blur,
                imputation =imputation,
                feat_layer = feat_layer,
                experiment = experiment,
                feat_layer_name = feat_layer_name,
    )

def run_channel_deletion_game(model,ref,target_id,
mask,ratios_retained,batch_size=dutils.TODO,
    perturbation = elp_masking.BLUR_PERTURBATION,
    max_blur=20,
    imputation ='blur',
    feat_layer = None,
    experiment = ['class','feat','channel'],
    feat_layer_name = None,
):
    device = ref.device
    ref_input = ref
    assert feat_layer is not None, "need to provide feat_layer for channel experiment"
    assert feat_layer_name is not None, "need to provide feat_layer_name for channel experiment"    
    assert model is not None, "need to provide model for channel experiment"
    with torch.inference_mode():
        with register_feat_hook(feat_layer) as ref_feats:
            _ = model(ref)
            ref_feats = ref_feats[0].detach().clone()
    ref_input = ref_feats

    assert model is not None
    if mask.ndim != ref_input.ndim:
        if mask.ndim == 4 and ref_input.ndim == 2:
            assert mask.shape[-2:] == (1,1)
            mask = mask[...,0,0]

    ratios_retained = torch.tensor(ratios_retained,device=device)
    deleted_input = torch.zeros((len(ratios_retained),) + ref_input.shape[1:],device=device)
    with register_feat_hook(feat_layer) as ref_feats_:
        ref_scores = model(ref)
        ref_feats = ref_feats_[0].detach().clone()
    ref_probs = torch.softmax(ref_scores,dim=1)
    if ref_scores.ndim == 4:
        ref_scores = ref_scores.mean(dim=(-1,-2))
        ref_probs = ref_probs.mean(dim=(-1,-2))
    ref_probs = ref_probs[:,target_id]
    ref_scores = ref_scores[:,target_id]

    #=================================================================
    # assert mask.ndim == 2, f'mask dim {mask.ndim}'
    assert mask.shape[0] == 1, f'mask shape {mask.shape}'
    flat_mask = mask.flatten()
    sorted_mask_ascending,argsort_ascending = torch.sort(flat_mask,descending=False)
    _,unsort_ascending = torch.sort(argsort_ascending) 
    cutoff_ixs = (len(sorted_mask_ascending)*ratios_retained).long()
    
    cutoff_ixs = torch.clamp(cutoff_ixs,0,len(sorted_mask_ascending)).long()
    dummy_range = torch.arange(flat_mask.shape[0],device=flat_mask.device)
    dummy_mask_01 = (dummy_range[None,:] < cutoff_ixs[:,None]).to(ref_feats.dtype)

    pause2('DBG_METRICS_MAR6')
    flat_mask_01 = dummy_mask_01[:,unsort_ascending]
    mask_01 = flat_mask_01.view(cutoff_ixs.shape[0],*mask.shape[1:])
        

    assert mask_01[ratios_retained == 0].sum() == 0, f'mask_01[ratios_retained == 0].sum() {mask_01[ratios_retained == 0].sum()}'

    assert mask_01[ratios_retained == 1].sum() == np.prod(mask_01[0].shape), f'mask_01[ratios_retained == 1].sum() {mask_01[ratios_retained == 1].sum()}'

    if ref_input.ndim > mask_01.ndim:
        for _ in range(ref_input.ndim - mask_01.ndim):
            mask_01 = mask_01[...,None]
    #=================================================================
    def masking_hook(m,i,o):
        assert imputation == 'zero', 'only zero imputation is supported for channel experiment'
        assert mask_01.shape[:-2] == o.shape[:-2]
        new_o = mask_01 * o
        return new_o

    hook = feat_layer.register_forward_hook(masking_hook)
    with torch.inference_mode():
        # repeat or expand dimension 0
        with register_feat_hook(feat_layer) as feats_of_masked_:
            scores = model(ref.repeat(len(ratios_retained),1,1,1))
            feats_of_masked = feats_of_masked_[0]
    hook.remove()
    probs = torch.softmax(scores,dim=1)
    if scores.ndim == 4:
        scores = scores.mean(dim=(-1,-2))
        probs = probs.mean(dim=(-1,-2))
    probs = probs[:,target_id]
    scores = scores[:,target_id]

    assert probs.ndim == 1, f'probs.ndim {probs.ndim}'
    diff_in_probs = probs - ref_probs
    '''
    # (1,20,1,1)
    # (1,3,300,500) --> (1,20,2,5)
    # (1,1000) 
    '''

    probs = tensor_to_numpy(probs)
    diff_in_probs = tensor_to_numpy(diff_in_probs)
    ref_probs = tensor_to_numpy(ref_probs)

    results = dict(
        probs = probs,
        ref_probs = ref_probs,
        diff_in_probs = diff_in_probs,
        ratios = ratios_retained,
        imputation = imputation,
    )
    return results

def add_to_results_xz(method=dutils.TODO,
            arch=dutils.TODO,
            dataset=dutils.TODO,
            results_root_dir=dutils.TODO,
            imputation = 'blur',
            **kwargs,
):
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

def run(method=dutils.TODO,dataset=dutils.TODO,arch=dutils.TODO,
results_root_dir=dutils.TODO,
save_root_dir=METRICS_ROOT_DIR,
batch_size = dutils.TODO,
max_blur = dutils.TODO,
imputation='blur',
ratios = dutils.TODO,
start = 0,
feat_layer = None,
feat_layer_name = None,
device = dutils.hardcode(device="cuda"),
experiment = 'class',
input_size=None,
ntodo=-1,
**ignore
):

    if len(ignore):
        print(colorful.red(f'need toadd {ignore.keys()} to run arguments'))
    #ratios_retained = dutils.hardcode(ratios_retained=np.linspace(0,1,10))
    ratios_retained = ratios
    ratios_retained = np.array(ratios_retained)
    if not np.allclose((np.sort(ratios_retained ) - np.sort(1-ratios_retained)),np.zeros(ratios_retained.shape) ):
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
    if input_size is None:
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
    if feat_layer is None and isinstance(feat_layer_name,str):
        names_and_mods = list(model.named_modules())
        layer_names = [el[0] for el in names_and_mods]
        feat_layers = [el[1] for el in names_and_mods]
        feat_layer = feat_layers[layer_names.index(feat_layer_name)]

    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    if imputation == 'blur':
        save_dir = os.path.join(save_root_dir,"deletion",f"{dataset}-{method}-{arch}")
    else:
        save_dir = os.path.join(save_root_dir,"deletion",f"{dataset}-{method}-{arch}-{imputation}")
    methoddir = os.path.join(results_root_dir,f'{dataset}-{method}-{arch}')
    pattern = os.path.join(methoddir,'*','*.xz')
    xzfiles = glob.glob(pattern)
    assert len(xzfiles), f'xzfiles is empty, {methoddir}'

    # Order by the dataset's own image order so the start/ntodo slice picks the same
    # samples the benchmark processes (glob order is arbitrary; a [0:30] slice of it
    # was an arbitrary 30 files, not samples 0..29). Loud KeyError if an xz's imroot
    # is not in the dataset.
    imroot_order = {os.path.splitext(os.path.basename(p))[0]: i
                    for i, p in enumerate(data.images)}
    xzfiles = sorted(xzfiles, key=lambda x: imroot_order[os.path.basename(os.path.dirname(x))])

    xzfiles = xzfiles[start:( start+ntodo if ntodo not in (None,-1) else len(xzfiles))]

    running_scores = {'insertion':[],'deletion':[]}
    for xzfile in tqdm.tqdm(dutils.trunciter(xzfiles,enabled=False,max_iter=10)):
        print(xzfile)
        xzfile = os.path.abspath(xzfile)
        #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        found = False
        imroot = os.path.basename(os.path.dirname(xzfile))

        for imix,impath in enumerate(data.images):
            if imroot in impath:
                found = True
                break
        assert found
        ref,y = data[imix]
        ref = ref[None]
        ref = ref.to(device)
        #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        try:
            with lzma.open(xzfile,'rb') as f:
                loaded = pickle.load(f)
        except Exception as e:
            print(traceback.format_exc())
            p46()
            continue
        #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        class_id =loaded['class_id']
        saliency = loaded['saliency']

        '''
        if saliency.max() > 1:
            saliency = saliency/saliency.max()
        '''
        if saliency.max() > 0:
            saliency = saliency/saliency.max()
        assert saliency.max() <= 1.
        assert saliency.min() >= 0
        class_name = loaded['class_name']
        assert saliency.ndim == 4
        saliency = torch.tensor(saliency,device=ref.device)
        if experiment != 'channel':
            saliency = torch.nn.functional.interpolate(saliency,ref.shape[-2:],mode="bilinear")
        #dutils.img_save(saliency,"saliency.png")
        #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        results_deletion = run_deletion_game(model,ref,class_id,
           saliency,ratios_retained,batch_size=batch_size,max_blur=max_blur,imputation=imputation, feat_layer = feat_layer,feat_layer_name=feat_layer_name,experiment=experiment)
        results_insertion = run_deletion_game(model,ref,class_id,
            1-saliency,ratios_retained,batch_size=batch_size,max_blur=max_blur,imputation=imputation, feat_layer = feat_layer,feat_layer_name=feat_layer_name,experiment=experiment)
        results = dict(
            insertion = results_insertion,
            deletion= results_deletion,
            arch = arch,
            dataset = dataset,
            method = method,
            imroot = imroot,
            class_name = class_name,
            class_id = class_id,
        )
        running_scores['insertion'].append(results_insertion['probs'])
        running_scores['deletion'].append(results_deletion['probs'])
        wandb.log(dict(running_insertion_score = np.array(running_scores['insertion']).mean()),commit=False)
        wandb.log(dict(running_deletion_score = np.array(running_scores['deletion']).mean()),commit=False)
        """
        deletion/voc_2007-grad_cam-resnet50
        """
        classname_classid_xz  = os.path.basename(xzfile)
        imroot = os.path.basename(os.path.dirname(xzfile))
        os.makedirs(os.path.join(save_dir,imroot),exist_ok=True)
        savepath = os.path.join(save_dir,imroot,classname_classid_xz)

        print(savepath)

        with lzma.open(savepath,'wb') as f:
            pickle.dump(results,f)
        wandb.log(dict(xzfile=xzfile),commit=False)
        wandb.log({})

    '''
    <parent-directory>/000001/dog11.xz
    <parent-directory>/000001/person14.xz
    <parent-directory>/000002/car6.xz
    '''
    pass

def main():
    #"""
    parser = argparse.ArgumentParser() 
    parser.add_argument("--method",type=str)
    parser.add_argument("--arch",type=str)
    parser.add_argument("--dataset",type=str)
    parser.add_argument("--ratios",type=float,nargs="*")
    parser.add_argument("--results_root_dir",type=str,default=RESULTS_ROOT_DIR)
    parser.add_argument("--save_root_dir",type=str,default=METRICS_ROOT_DIR)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--max_blur",type=float,default=20)
    parser.add_argument("--imputation",type=str,default='blur',choices=['blur','road','zero'])
    parser.add_argument("--add-to-results-xz",type=lambda t:t.lower() == 'true',default=False,dest="add_to_results_xz")
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--experiment",type=str,default='class',choices=['class','feat','channel'])
    args = parser.parse_args()
    
    #"""
    #args = argparse.Namespace()
    #args.batch_size = 32
    #args.method = dutils.hardcode(method = "extremal_perturbation")
    #args.arch = dutils.hardcode(arch= "resnet50")
    #args.dataset = dutils.hardcode(dataset= "voc_2007")
    #args.results_root_dir = dutils.hardcode(results_root_dir=RESULTS_ROOT_DIR)
    #args.save_root_dir = dutils.hardcode(save_root_dir=METRICS_ROOT_DIR)
    # python cam_benchmark.deletion --method grad_cam --arch vgg16 --dataset imagenet-5000 --ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 
    # p46()
    if wandb.run is None:
       wandb.init(project=f"deletion-{args.dataset}-{args.method}-{args.arch}-{args.imputation}",config=dict(dataset=args.dataset,method=args.method,arch=args.arch,imputation=args.imputation,ratios=args.ratios, experiment=args.experiment))
   
    if not args.add_to_results_xz:
        run(**vars(args))
    else:
        #dutils.pause()
        add_to_results_xz(**vars(args))

if __name__ == '__main__':
    main()
    pass
