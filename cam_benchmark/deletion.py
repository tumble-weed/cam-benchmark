import dutils
dutils.init()
import glob
METRICS_ROOT_DIR="/root/bigfiles/other/metrics-torchray/"
RESULTS_ROOT_DIR = dutils.hardcode(RESULTS_ROOT_DIR="/root/bigfiles/other/results-torchray")
def delete(ref,mask,ratio_retained):
    masked = dutils.hardcode(masked = torch.zeros_like(ref))
    return masked

def run_deletion_game(model,ref,target_id,
mask,ratios_retained,batch_size=dutils.TODO):
    device = ref.device
    deleted_images = torch.zeros((len(ratios_retained),) + ref.shape[1:],device=device)
    ref_scores = model(ref)
    ref_probs = torch.softmax(ref_scores,dim=1)
    if ref_scores.ndim == 4:
        ref_scores = ref_scores.mean(dim=(-1,-2))
        ref_probs = ref_probs.mean(dim=(-1,-2))
    ref_probs = ref_probs[:,target_id]
    ref_scores = ref_scores[:,target_id]
    for i,ratio_retained in enumerate(ratios_retained):
        deleted_ref = delete(ref,mask,ratio_retained)
        deleted_images[i:i+1] = deleted_ref
    assert deleted_images.shape[0] <= batch_size, 'implement batched forward'
    with torch.inference_mode():
        scores = model(deleted_images)
    probs = torch.softmax(scores,dim=1)
    if scores.ndim == 4:
        scores = scores.mean(dim=(-1,-2))
        probs = probs.mean(dim=(-1,-2))
    probs = probs[:,target_id]
    scores = scores[:,target_id]
    dutils.note('check broadcasting of probs')
    dutils.pause();
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
    results = dict(
        probs = probs,
        ref_probs = ref_probs,
        diff_in_probs = diff_in_probs,
        ratios = ratios_retained,

    )
    #dutils.pause()
    return results
    #pass
def main():
    """
    parser = argparse.ArgumentParser() 
    parser.add_argument("--method",type=str)
    parser.add_argument("--arch",type=str)
    parser.add_argument("--dataset",type=str)
    parser.add_argument("--ratios",type=float,nargs="*")
    parser.add_argument("--results_root_dir",type=str,default=RESULTS_ROOT_DIR)
    parser.add_argument("--save_root_dir",type=str,default=METRICS_ROOT_DIR)
    parser.add_argument("--batch_size",type=int,default=32)
    args = parser.parse_args()
    """
    args = argparse.Namespace()
    args.batch_size = 32
    args.method = dutils.hardcode(method = "dummy")
    args.arch = dutils.hardcode(arch= "resnet50")
    args.dataset = dutils.hardcode(dataset= "voc_2007")
    args.results_root_dir = dutils.hardcode(results_root_dir=RESULTS_ROOT_DIR)
    args.save_root_dir = dutils.hardcode(save_root_dir=METRICS_ROOT_DIR)
    run(**vars(args))
def run(method=dutils.TODO,dataset=dutils.TODO,arch=dutils.TODO,
results_root_dir=dutils.TODO,
save_root_dir=dutils.TODO,
batch_size = dutils.TODO,
**ignore
):
    if len(ignore):
        print(colorful.red(f'need toadd {ignore.keys()} to run arguments'))
    ratios_retained = dutils.hardcode(ratios_retained=np.linspace(0,1,10))
    device = dutils.hardcode(device="cuda")
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    if dataset == 'voc_2007':
# model = dutils.hardcode(model = torchvision.models.vgg16(pretrained=True))
        from torchray.benchmark.models import get_model, get_transform
        model = get_model(
                arch=arch,
                dataset=dataset,
                convert_to_fully_convolutional=True,
            )
# dutils.pause()
        model.to(device)
    elif 'imagenet' in dataset:
        dutils.pause()
        pass
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    if dataset == 'voc_2007':
        # ref = dutils.hardcode(ref = torch.zeros(1,3,224,224,device=device))
        from torchray.benchmark.models import get_transform
        from torchray.benchmark.datasets import get_dataset
        if method == "rise":
            input_size = (224, 224)
        else:
            input_size = 224        
        subset = 'test'
        transform = get_transform(size=input_size,
                                  dataset=dataset)
        
        data = get_dataset(name=dataset,
                            subset=subset,
                            transform=transform,
                            download=False,
                            limiter=None)
    elif 'imagenet' in dataset:
        dutils.pause()
        pass
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    save_dir = os.path.join(METRICS_ROOT_DIR,"deletion",f"{dataset}-{method}-{arch}")

    methoddir = os.path.join(results_root_dir,f'{dataset}-{method}-{arch}')
    pattern = os.path.join(methoddir,'*','*.xz') 
    xzfiles = glob.glob(pattern)
    # xzfiles = list(sorted(glob.glob(os.path.join(methoddir,'*','*.xz'))))
    for xzfile in tqdm.tqdm(dutils.trunciter(xzfiles,enabled=True,max_iter=10)):
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
            print(f'{xzpath} is corrupt')
            # dutils.pause()
            continue
        #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        class_id =loaded['class_id']
        mask = loaded['saliency']
        class_name = loaded['class_name']
        assert mask.ndim == 4
        mask = torch.tensor(mask,device=ref.device)
        mask = torch.nn.functional.interpolate(mask,ref.shape[-2:],mode="bilinear")
        dutils.img_save(mask,"mask.png")
        dutils.pause()
        #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        results_insetion = run_deletion_game(model,ref,class_id,
            mask,ratios_retained,batch_size=batch_size)
        results_deletion = run_deletion_game(model,ref,class_id,
            mask,ratios_retained,batch_size=batch_size)
        results = dict(
        insertion = results_insertion,
        deletion= results_deletion
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
        os.makedirs(os.path.join(save_dir,imroot))
        savepath = os.path.join(save_dir,imroot,classname_classid_xz)
        with lzma.open(savepath,'wb') as f:
            pickle.dump(results_deletion,f)

    dutils.pause()
    '''
    <parent-directory>/000001/dog11.xz
    <parent-directory>/000001/person14.xz
    <parent-directory>/000002/car6.xz
    '''
    pass
import torchvision
if __name__ == '__main__':
    main()
    pass
