import dutils
import torch
import torchvision
import skimage.io
from PIL import Image
##############################################################################################

pascal_bgr_mean = [103.939, 116.779, 123.68]
pascal_mean = [m / 255. for m in reversed(pascal_bgr_mean)]
pascal_std = [1 / 255.] * 3

def denormalize_tensor(t,vgg_mean=[0.485, 0.456, 0.406],
                     vgg_std=[0.229, 0.224, 0.225]):
    device = t.device
    # out = (t - torch.tensor(vgg_mean).to(device)[None,:,None,None])/torch.tensor(vgg_std).to(device)[None,:,None,None]
    out = (t * torch.tensor(vgg_std).to(device)[None,:,None,None]) + torch.tensor(vgg_mean).to(device)[None,:,None,None]
    return out
def denormalize_tensor_for_dataset(t,dataset= None):
    device = t.device
    # out = (t - torch.tensor(vgg_mean).to(device)[None,:,None,None])/torch.tensor(vgg_std).to(device)[None,:,None,None]
    if dataset == 'imagenet':
        vgg_mean=[0.485, 0.456, 0.406]
        vgg_std=[0.229, 0.224, 0.225]
        out = (t * torch.tensor(vgg_std).to(device)[None,:,None,None]) + torch.tensor(vgg_mean).to(device)[None,:,None,None]
        return out
    elif dataset in ['pascal','voc']:
        # pass
        bgr_mean = [103.939, 116.779, 123.68]
        mean = [m / 255. for m in reversed(bgr_mean)]
        std = [1 / 255.] * 3
        out = (t * torch.tensor(std).to(device)[None,:,None,None]) + torch.tensor(mean).to(device)[None,:,None,None]
        return out
    else:
        assert False
vgg_mean=[0.485, 0.456, 0.406]
vgg_std=[0.229, 0.224, 0.225]
def get_vgg_transform(size):
    vgg_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(size),
            torchvision.transforms.Normalize(mean=vgg_mean,std=vgg_std),
            ]
        )
    return vgg_transform

def normalize_tensor(t,vgg_mean=vgg_mean,
                     vgg_std=vgg_std):
    device = t.device
    # out = (t - torch.tensor(vgg_mean).to(device)[None,:,None,None])/torch.tensor(vgg_std).to(device)[None,:,None,None]
    # out = (t * torch.tensor(vgg_std).to(device)[None,:,None,None]) + torch.tensor(vgg_mean).to(device)[None,:,None,None]
    out = t - torch.tensor(vgg_mean).to(device)[None,:,None,None]
    out = out / torch.tensor(vgg_std).to(device)[None,:,None,None]
    return out

def get_image_tensor(impath,size=(224,),dataset=None):
    im_ = skimage.io.imread(impath)
    if im_.ndim == 2:
        im_ = np.concatenate([im_[...,None],im_[...,None],im_[...,None]],axis=-1)
    im_pil = Image.fromarray(im_)
    # import ipdb;ipdb.set_trace()
    if dataset == 'imagenet':
        # from cam_benchmark.cnn import get_vgg_transform
        vgg_transform = get_vgg_transform(size)
        ref = vgg_transform(im_pil).unsqueeze(0)
        return ref
    elif dataset in ['pascal','voc']:
        # vgg_mean = (0.485, 0.456, 0.406)
        # vgg_std = (0.229, 0.224, 0.225)
        bgr_mean = [103.939, 116.779, 123.68]
        mean = [m / 255. for m in reversed(bgr_mean)]
        std = [1 / 255.] * 3
        
        vgg_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(size),
                torchvision.transforms.Normalize(mean=mean,std=std),
                ]
            )
        ref = vgg_transform(im_pil).unsqueeze(0)
        # import ipdb;ipdb.set_trace()
        return ref
    else:
        assert False
        

def get_model(dataset,modelname,is_relevancecam,device=None):
    
    assert modelname in ['vgg16','resnet50','dummymodel'],f'{modelname} not recognized'
    rprop = False
    convert_to_fully_convolutional=True
    if is_relevancecam:
        # convert_to_fully_convolutional=False
        rprop = True
    # print('TODO:fix hardcode of voc')
    if modelname == 'dummymodel':
        class MockObject():
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # self.return_value = self                
            def __getattr__(self, name):
                return MockObject()      
            def __call__(self,*args,**kwargs):
                pass
        return MockObject()    
    if dataset in ['imagenet','']:
        dataset = 'imagenet'
        # if dutils.hack2('ignore model',default = False):
        #    import torchvision
        #    model = torchvision.models.vgg16(pretrained=True).to(device)
        # else:
        # dutils.pause()
        if True:
            import cam_benchmark.libre_cam_models.relevance.vgg
            import cam_benchmark.libre_cam_models.relevance.resnet
            # import ipdb;ipdb.set_trace()
            if modelname == 'vgg16':
                model = cam_benchmark.libre_cam_models.relevance.vgg.vgg16_bn(pretrained = True).to(device)
            else:
                model = cam_benchmark.libre_cam_models.relevance.resnet.resnet50(pretrained = True).to(device)                    
    elif dataset in ['voc','pascal']:
        dataset = 'voc'
        from benchmark.architectures import get_model as get_model_
        model = get_model_(
                    arch=modelname,
                    dataset=dataset,
                    rprop = rprop,
                    convert_to_fully_convolutional=convert_to_fully_convolutional,
                    
                )
    else:
        assert False
    model.eval()
    model.to(device)
    model.dataset = dataset
    return model

VOC_ROOT_DIR = '/root/bigfiles/dataset/voc/VOCdevkit/VOC2007'
IMAGENET_ROOT_DIR = '/root/bigfiles/dataset/imagenet/'

def get_image_tensor_and_gt(impath,size=dutils.TODO,dataset=dutils.TODO):
    # im = skimage.io.imread(impath)
    # im_pil = Image.fromarray(im)
    imroot = os.path.splitext(os.path.basename(impath))[0]
    ref = get_image_tensor(impath,size=(224,),dataset=dataset)
    import cam_benchmark.imagenet_localization_parser
    # cam_benchmark.get_gt_bbox(impath,class_name,size)
    if dataset == 'imagenet':
        imagenet_annotation_dir = os.path.join(IMAGENET_ROOT_DIR,"bboxes/val/")
        label = cam_benchmark.imagenet_localization_parser.get_voc_label(
    root_dir = imagenet_annotation_dir,
    x = imroot,
    full_filename = None)
        bbox_info = label
        synsets = [obj['name'] for obj in label['annotation']['object']]
        import cam_benchmark.synset_utils
        target_ids = [cam_benchmark.synset_utils.synset_id_to_imagenet_class_ix(synset) for synset in synsets]
        classnames = [cam_benchmark.synset_utils.synset_id_to_imagenet_class_name(synset) for synset in synsets]
        # pass
    
    # dutils.pause()
    return ref,bbox_info,target_ids,classnames
    
