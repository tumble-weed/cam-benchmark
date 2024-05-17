import dutils
dutils.init()
import glob
import cam_benchmark.elp_masking as elp_masking
import cam_benchmark.road
import torchvision
METRICS_ROOT_DIR="/root/bigfiles/other/metrics-torchray/"
RESULTS_ROOT_DIR = dutils.hardcode(RESULTS_ROOT_DIR="/root/bigfiles/other/results-torchray")
#RESULTS_ROOT_DIR = dutils.hardcode(RESULTS_ROOT_DIR="/root/bigfiles/other/results-torchray/old_multi_results_mar4")
#RESULTS_ROOT_DIR2 = dutils.hardcode(RESULTS_ROOT_DIR="/root/bigfiles/other/results-torchray2")
def impute_where_0(ref,mask,ratio_retained=None,
perturbation = elp_masking.BLUR_PERTURBATION,
max_blur=20,
imputation='blur',
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
        masked = imputer(ref[0].cpu(),mask_01[0,0].cpu())
        masked = masked[None,...]
        #p47()
        pass
    else:
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
):
    device = ref.device
    ratios_retained = torch.tensor(ratios_retained,device=device)
    deleted_images = torch.zeros((len(ratios_retained),) + ref.shape[1:],device=device)
    # p46()
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
        #p47()
        pause2('DBG_METRICS_MAR6')
        flat_mask_01 = dummy_mask_01[:,unsort_ascending]
        mask_01 = flat_mask_01.view(cutoff_ixs.shape[0],*mask.shape[1:])
        

    #p47()
    if True or (ratios_retained == 0).any():
        assert mask_01[ratios_retained == 0].sum() == 0
    if True or (ratios_retained == 1).any():
        assert mask_01[ratios_retained == 1].sum() == np.prod(mask_01[0].shape)
    #=================================================================
    for i,ratio_retained in enumerate(ratios_retained):
        #dutils.img_save(mask_01[i],f'mask_01_{mask_01[i].sum()}.png')
        pause2('DBG_METRICS_MAR6')
        deleted_ref, perturbation= impute_where_0(ref,mask_01[i:i+1],ratio_retained=None,perturbation=perturbation,max_blur=max_blur,imputation=imputation)
        deleted_images[i:i+1] = deleted_ref

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
    if feat_layer is not None:
        feats = feat_layer.feats
        assert feats.ndim == 2
        feat_distance = ((feats - ref_feats)**2).sum(dim=-1)
        feat_distance = tensor_to_numpy(feat_distance)
        results['feat_distance'] = feat_distance

    #dutils.pause()
    return results
    #pass
def add_to_results_xz(method=dutils.TODO,
            arch=dutils.TODO,
            dataset=dutils.TODO,
            results_root_dir=dutils.TODO,
            imputation = 'blur',
            **kwargs,
):
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
def run(method=dutils.TODO,dataset=dutils.TODO,arch=dutils.TODO,
results_root_dir=dutils.TODO,
save_root_dir=dutils.TODO,
batch_size = dutils.TODO,
max_blur = dutils.TODO,
imputation='blur',
ratios = dutils.TODO,
start = 0,
feat_layer = None,
**ignore
):
    if len(ignore):
        print(colorful.red(f'need toadd {ignore.keys()} to run arguments'))
    #ratios_retained = dutils.hardcode(ratios_retained=np.linspace(0,1,10))
    ratios_retained = ratios
    ratios_retained = np.array(ratios_retained)
    if not np.allclose((np.sort(ratios_retained ) - np.sort(1-ratios_retained)),np.zeros(ratios_retained.shape) ):
        dutils.pause()
    device = dutils.hardcode(device="cuda")
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    #if dataset == 'voc_2007':
    if True:
# model = dutils.hardcode(model = torchvision.models.vgg16(pretrained=True))
        from torchray.benchmark.models import get_model, get_transform
        model = get_model(
                arch=arch,
                dataset=dataset,
                convert_to_fully_convolutional=True,
            )
# dutils.pause()
        model.to(device)
        model.eval()
    #elif 'imagenet' in dataset:
    #    dutils.pause()
    #    pass
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    #if dataset == 'voc_2007':
        # ref = dutils.hardcode(ref = torch.zeros(1,3,224,224,device=device))
        from torchray.benchmark.models import get_transform
        from torchray.benchmark.datasets import get_dataset
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
    if feat_layer is not dutils.TODO:
        from multithresh_saliency.run_self_saliency import get_layernames
        feat_layers,layer_names = get_layernames(network=arch,model=model)
        feat_layer = feat_layers[layer_names.index(feat_layer)]
        p46()
    #elif 'imagenet' in dataset:
    #    dutils.pause()
    #    pass
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    if imputation == 'blur':
        save_dir = os.path.join(METRICS_ROOT_DIR,"deletion",f"{dataset}-{method}-{arch}")
    else:
        save_dir = os.path.join(METRICS_ROOT_DIR,"deletion",f"{dataset}-{method}-{arch}-{imputation}")

    methoddir = os.path.join(results_root_dir,f'{dataset}-{method}-{arch}')
    pattern = os.path.join(methoddir,'*','*.xz') 
    xzfiles = glob.glob(pattern)
    assert len(xzfiles), f'xzfiles is empty, {methoddir}'
    #p46()
    xzfiles = xzfiles[start:]
    # xzfiles = list(sorted(glob.glob(os.path.join(methoddir,'*','*.xz'))))
    # p47()
    for xzfile in tqdm.tqdm(dutils.trunciter(xzfiles,enabled=False,max_iter=10)):
        print(xzfile)
        xzfile = os.path.abspath(xzfile)
        #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        found = False
        imroot = os.path.basename(os.path.dirname(xzfile))
        #dutils.pause()
        for imix,impath in enumerate(data.images):
            if imroot in impath:
                found = True
                break
        assert found
        ref,y = data[imix]
        ref = ref[None]
        ref = ref.to(device)
        # dutils.pause()
        pass
        #ref = dutils.hardcode(ref = torch.randn(1,3,224,224))
        #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        try:
            with lzma.open(xzfile,'rb') as f:
                loaded = pickle.load(f)
        except Exception:
            print(f'{xzfile} is corrupt')
            # dutils.pause()
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
        saliency = torch.nn.functional.interpolate(saliency,ref.shape[-2:],mode="bilinear")
        #dutils.img_save(saliency,"saliency.png")
        #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        results_deletion = run_deletion_game(model,ref,class_id,
           saliency,ratios_retained,batch_size=batch_size,max_blur=max_blur,imputation=imputation, feat_layer = feat_layer)
        results_insertion = run_deletion_game(model,ref,class_id,
            1-saliency,ratios_retained,batch_size=batch_size,max_blur=max_blur,imputation=imputation, feat_layer = feat_layer)
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
        # break
        """
        deletion/voc_2007-grad_cam-resnet50
        """
        classname_classid_xz  = os.path.basename(xzfile)
        imroot = os.path.basename(os.path.dirname(xzfile))
        os.makedirs(os.path.join(save_dir,imroot),exist_ok=True)
        savepath = os.path.join(save_dir,imroot,classname_classid_xz)

        print(savepath)
        #dutils.pause()
        # p46()
        with lzma.open(savepath,'wb') as f:
            pickle.dump(results,f)

    #dutils.pause()
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
    parser.add_argument("--imputation",type=str,default='blur',choices=['blur','road'])
    parser.add_argument("--add-to-results-xz",type=lambda t:t.lower() == 'true',default=False,dest="add_to_results_xz")
    parser.add_argument("--start",type=int,default=0)
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
    if not args.add_to_results_xz:
        run(**vars(args))
    else:
        #dutils.pause()
        add_to_results_xz(**vars(args))

if __name__ == '__main__':
    main()
    pass
