import dutils
dutils.init()
import torchvision
#..................................................
def randomize_linear_layer(l):
    str_class = str(l.__class__).lower()
    assert 'linear' in str_class
    l.weight.data.copy_(torch.randn_like(l.weight))
    # l.weight.data.copy_(torch.zeros_like(l.weight))
    pass
def randomize_conv2d_layer(l):
    str_class = str(l.__class__).lower()
    assert 'conv2d' in str_class
    l.weight.data.copy_(torch.randn_like(l.weight))
    # l.weight.data.copy_(torch.zeros_like(l.weight))
    pass
#..................................................
def randomize_last_n_layers(model,n):
    if 'vgg' in str(model.__class__):
        classifier_layers = list(model.classifier.children())
        feature_layers = list(model.features.children())
        all_layers = feature_layers + classifier_layers
        # n = 3
        n_done = 0
        for l in reversed(all_layers):
            str_class = str(l.__class__).lower()
            print(str_class)
            if 'linear' in str_class:
                randomize_linear_layer(l)
                n_done += 1
                # pass
            if 'conv2d' in str_class:
                randomize_conv2d_layer(l)
                n_done += 1
            if n_done == n:
                break
        return (n_done < n)

            # break
        # dutils.pause()
    elif 'resnet' in str(model.__class__):
        dutils.pause()
    # pass
# def create_randomized_model(model,n_layers):
#     return False
def run_cascade_sanity(ref,target,run_method,method,dataset,arch,device='cuda'):
    n_layers = 1
    cascade_sanity_results = []
    while True:
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
        is_too_many_layers = randomize_last_n_layers(model,n_layers)
        if is_too_many_layers:
            break
        if True:
            saliency = run_method(model,ref,target)
        if isinstance(saliency,torch.Tensor):
            saliency = tensor_to_numpy(saliency)
        dutils.pause()
        results_dict = {
            'n_layers_randomized':n_layers,
            'saliency': saliency,
            'arch':arch,
            'dataset':dataset,
            'method':method,
        }
        cascade_sanity_results.append(results_dict)
        n_layers += 1
        '''
        if dutils.hack('early break',default=True):
            if n_layers == 4:
                break
        '''
        del model
    # dutils.pause()
    
    return cascade_sanity_results
    
    
def run_and_save_sanity_check(ref,target,run_method,method,dataset,arch,imroot,device='cuda',
cascade_parent_dir = dutils.TODO,
):
    cascade_sanity_results = run_cascade_sanity(ref,target,run_method,method,dataset,arch,device=device)
    # cascade_sanity_results = dutils.hardcode(

    #     cascade_sanity_results = [{'n_layers_randomized':1,'saliency':torch.zeros(1,1,224,224,device=device),'arch':'vgg16','dataset':'voc_2007','method':method},
    #     {'n_layers_randomized':1,'saliency':torch.zeros(1,1,224,224,device=device),'arch':'vgg16','dataset':'voc_2007','method':method}]

    # )
    # dutils.pause()
    savepath = os.path.join(cascade_parent_dir,f'cascade_sanity-{dataset}-{method}-{arch}-{imroot}.pkl')
    dutils.pause()
    with open(savepath,'wb') as f:
        pickle.dump(cascade_sanity_results,f)
    pass
# run_and_save_sanity_check()
def dummy_attribution(model,ref,target):
    return torch.zeros(1,1,224,224,device=ref.device)

import torchray.attribution.extremal_perturbation_variants as  extremal_perturbation_variants
def get_wrapper_for_extremal_perturbation(method,dataset,method_kwargs):
    def wrapper_for_extremal_perturbation(model,ref,target):
        if True:
            # assert dutils.UNTESTED
            # from torchray.attribution.extremal_perturbation_variants import run_method
            # import torchray.attribution.extremal_perturbation_variants
            # assert dutils.UNTESTED
            info = extremal_perturbation_variants.run_method(
                    ref,target,model,
                    ref.shape[-2:],
                    #----------------------
                    method,
                    dataset,
                    #----------------------
                    extra_info = None,
                    debug = 0,
                    boom = False,
                    #----------------------
                    **method_kwargs
                    #----------------------
                    
            )
            # if self.experiment.dataset == 'coco':
            #     assert dutils.UNTESTED
            # info['filename'] = filename
            # results['info'][class_id] = info
            # point = _saliency_to_point(torch.tensor(info['saliency']))
            # assert (tensor_to_numpy(point) == tensor_to_numpy(info['point'])).all()
            return info['saliency']
    return wrapper_for_extremal_perturbation

def main(method,dataset,arch,imroot,target,device='cuda'):
    dutils.note('pass device')
    cascade_parent_dir = '/root/bigfiles/other/results-torchray'
    imroot = os.path.splitext(os.path.basename(imroot))[0]
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
        found = False
        for imix,impath in enumerate(data.images):
            if imroot in impath:
                found = True
                break
        assert found
        ref,y = data[imix]
        ref = ref[None]
        ref =ref.to(device)
        # dutils.pause()
        pass
    elif 'imagenet' in dataset:
        dutils.pause()
        pass
    if method.startswith('extremal_perturbation'):
        # run_method = dutils.hardcode(run_method = lambda *args,**kwargs:torch.zeros(1,1,224,224,device=device))
        # wrapper_for_extremal_perturbation
        method_kwargs = {}
        run_method = get_wrapper_for_extremal_perturbation(method,dataset,method_kwargs)
        # pass
    elif method == 'dummy1':
        run_method = dummy_attribution
    dutils.note('extremal_perturbation...gp, multithresh_saliency')
    run_and_save_sanity_check(ref,target,run_method,method,dataset,arch,imroot,device=device,cascade_parent_dir=cascade_parent_dir)
    # dutils.pause()

    pass
if __name__ == '__main__':
    args = dutils.hardcode(args=argparse.Namespace())
    args.method = "extremal_perturbation"
    # args.method = "extremal_perturbation_with_simple_scale_and_crop_normalized"
    args.dataset = 'voc_2007'
    # args.dataset = 'imagenet'
    args.arch = 'vgg16'
    # args.impath = '/root/bigfiles/dataset/voc/VOCdevkit/VOCdevkit_2007/VOC2007/JPEGImages/000001.jpg'
    args.imroot = '000001.jpg'
    args.target = 14
    dutils.note('check if target is valid for image')
    main(args.method,args.dataset,args.arch,args.imroot,args.target)
    # pass