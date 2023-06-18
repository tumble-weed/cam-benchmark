import copy
from benchmark.metrics_data_handler import MetricsDataHandler
import register_ipdb
import skimage.io
from matplotlib import cm
import skimage.transform
import os
import dutils
def get_metrics_for_all_methods(
    metricname,
    methodnames,
    modelname,
    dataset =None,
    ):
    metrics_all_methods = {}
    for methodname in methodnames:
        metrics_obj = MetricsDataHandler(dataset,metricname,methodname,modelname)
        metrics_all_images = metrics_obj.load_all_images_and_objects()
        metrics_all_methods[methodname] = metrics_all_images
        # select only correct
        # select difficult
        # average selected metrics
    return metrics_all_methods

def average_measurements(
        metrics_all_images,
        measurements = []
                        ):
    gross = {m:0 for m in measurements}
    # gross['n'] = 0
    n = 0
    for imroot,objects in metrics_all_images.items():
        for obj in objects:
            for m in measurements:
                gross[m] = gross[m] + obj[m]
            n += 1
    if n ==0:
        assert all([gross[m] == 0  for m in measurements])
        n = 1
    averages = {m:gross[m]/n for m in measurements}
    averages['n'] = n
    return averages

def get_easy_pointing_game(methodnames,modelname,dataset =None,for_methodname=None):
    assert for_methodname is not None
    
    metrics_all_methods = get_metrics_for_all_methods(
                        'pointing_game',
                        methodnames,
                        modelname,
                        dataset =dataset,
                        )
    all_possible_imroots = set(list(metrics_all_methods.values())[0].keys())    
    from collections import defaultdict
    metrics_per_imroot = defaultdict(dict)
    for imroot in all_possible_imroots:
        for methodname in methodnames:
            metrics_one_method = metrics_all_methods[methodname][imroot]
            metrics_one_method = [metrics_one_method_one_object['hit'] for metrics_one_method_one_object in metrics_one_method]
            """
            if len(metrics_one_method) > 1:
                assert False
            """
            # assert imroot in metrics_per_imroot
            metrics_per_imroot[imroot][methodname] = metrics_one_method#[imroot]    

    
    """
    find imroots that are easy for for_methodname but not for other methods
    """
    other_methodnames = [m for m in methodnames if m != for_methodname]
    avg_for_other_methods = {}
    for imroot in all_possible_imroots:
        metrics_for_other_methods = [metrics_per_imroot[imroot][m] for m in other_methodnames]
        as_float = [ float(obj) for method in metrics_for_other_methods for obj in method]
        avg_for_other_methods[imroot] = sum(as_float)/len(as_float)
    imroots_hard_to_easy = sorted(avg_for_other_methods,key=lambda x:avg_for_other_methods[x])
    avg_hard_to_easy = [avg_for_other_methods[r] for r in imroots_hard_to_easy]
    # keep only ixs where for_methodname has a hit
    easy_imroots = []
    for imroot in imroots_hard_to_easy:
        if all(metrics_per_imroot[imroot][for_methodname]):
            easy_imroots.append(imroot)
    import ipdb;ipdb.set_trace()
def compare_pointing_game(
    methodnames,
    modelname,
    dataset =None,
):
    metrics_all_methods = get_metrics_for_all_methods(
                        'pointing_game',
                        methodnames,
                        modelname,
                        dataset =dataset,
                        )
    if ('voc' in dataset) or ('pascal' in dataset):
        from ebp_difficult import VOCDifficult
        voc_difficult = VOCDifficult()
    all_possible_imroots = set(list(metrics_all_methods.values())[0].keys())
    #=====================================================
    for method in metrics_all_methods:
        method_imroots = set(metrics_all_methods[method].keys())
        assert len(method_imroots.difference(all_possible_imroots)) == 0
        assert len(all_possible_imroots.difference(method_imroots)) == 0
    #=====================================================
    from collections import defaultdict
    metrics_difficult_all_methods = {methodname:defaultdict(list) for methodname in methodnames}
    all_possible_imroots = list(all_possible_imroots)
    
    for imroot in all_possible_imroots:
        for methodname in methodnames:
            metrics_one_method = metrics_all_methods[methodname]
            # import ipdb;ipdb.set_trace()
            for obj in metrics_one_method[imroot]:
                
                target_id = obj['target_id']
                if dataset == 'voc' or dataset == 'pascal':
                    if voc_difficult.is_difficult(imroot,target_id):
                        # import ipdb;ipdb.set_trace()
                        metrics_difficult_all_methods[methodname][imroot].append(obj) 
    #=====================================================
    # import ipdb;ipdb.set_trace()
    average_pointing_game = {}
    average_pointing_game_difficult = {}
    for methodname in methodnames:
        metrics_all_images = metrics_all_methods[methodname]
        metrics_difficult_all_images = metrics_difficult_all_methods[methodname]
        method_average = average_measurements(
            metrics_all_images,
            measurements = ['hit']
                            )
        method_average_difficult = average_measurements(
            metrics_difficult_all_images,
            measurements = ['hit']
                            )
        average_pointing_game[methodname] = method_average
        average_pointing_game_difficult[methodname] = method_average_difficult
    #=====================================================
    return average_pointing_game,average_pointing_game_difficult


def compare_chattopadhyay(
    methodnames,modelname,
    dataset=None):
    
    metrics_all_methods = get_metrics_for_all_methods(
                        'chattopadhyay',
                        methodnames,
                        modelname,
                        dataset = dataset,
                        )   
    average_chattopadhyay = {}
    for methodname in methodnames:
        metrics_all_images = metrics_all_methods[methodname]
        method_average = average_measurements(metrics_all_images,measurements=[
            'normalized_drop_in_prob', 
            #'normalized_rise_in_prob', 
            'increase_indicator', 
            #'decrease_indicator'
        ])
        average_chattopadhyay[methodname] = method_average
    return average_chattopadhyay

import os
from benchmark import settings
import pickle
import lzma
import blosc2
def get_methoddata(dataset,methodname,modelname):
    methodxz = os.path.join(settings.RESULTS_DIR_librecam,f'{dataset}-{methodname}-{modelname}.xz')
    print(os.path.exists(methodxz))
    with lzma.open(methodxz,'rb') as f:
        methoddata = pickle.load(f)
    
    saliencybl2 = os.path.join(settings.RESULTS_DIR_librecam,f'{dataset}-{methodname}-{modelname}-saliency.bl2')
    loaded_saliency = blosc2.load_array(saliencybl2)                            
    # loaded_saliency = blosc2.load_array(saliencybl2)                            
    available_imroots = list(methoddata.keys())    
    return methoddata,loaded_saliency,available_imroots
    # arrayix = loaded['arrayix']
    # saliency = loaded_saliency[arrayix]  
    
def visualize_cherry_picked(methodnames,modelnames,datasets):
    import easy_images_for_mycam
    import dutils
    if modelnames is None:
        modelnames = ['vgg16','resnet50']
    if datasets is None:
        datasets = ['pascal','imagenet']
    # dataset_models = [f'{dataset}_{modelname}' for dataset in datasets for modelname in modelnames]
    for dataset in datasets:
        for modelname in modelnames:
            # for dataset_model in dataset_models:
            dataset_model = f'{dataset}_{modelname}'
            image_list = easy_images_for_mycam.__dict__[dataset_model]
            # assert False
            visualize_image_list(
                    methodnames,
                    modelname,
                    image_list,
                    dataset=dataset,
                    
                )
def visualize_image_method_model():
    pass
def visualize_image_list(
    methodnames,
    modelname,
    image_list,
    dataset=None,
    # root_save_dir=None,
    
):
    root_save_dir = os.path.join(settings.RESULTS_DIR_librecam,'method-saliency-comparison')
    for imroot in image_list:
        for methodnamej in methodnames:

            methoddata,loaded_saliency,available_imroots = get_methoddata(dataset,methodnamej,modelname)
            imagedata = methoddata[imroot]
            for obj in imagedata:
                classname = list(obj.keys())[0]
                loaded = obj[classname]
                target_id= loaded['target_id']
                arrayix = loaded['arrayix']
                saliency = loaded_saliency[arrayix]
                if saliency.max() > 1:
                    saliency = (saliency - saliency.min())/(saliency.max() - saliency.min())                
                saliency = (saliency - saliency.min())/(saliency.max() - saliency.min())                
                if saliency.ndim == 4:
                    saliency = saliency[0,0]
                elif saliency.ndim == 3:
                    saliency = saliency[0]                    
                #=============================
                
                if root_save_dir is not None:
                    d2 = os.path.join(root_save_dir,dataset,imroot)

                    try:
                        os.makedirs(d2)
                    except FileExistsError as e:
                        pass
        #             saliency_ = cm.hot(saliency[0])
                    
                    assert saliency.ndim == 2
                    saliency_ = cm.jet(saliency)
                    savename = os.path.join(d2,f'{modelname}{methodnamej}{target_id}.png')
                    skimage.io.imsave(savename,saliency_)
                    print(savename)
                    if dataset == 'pascal':
                        im = skimage.io.imread(os.path.join(dutils.PASCAL_IMAGE_ROOT,imroot+'.jpg'))
                    elif dataset == 'imagenet':
                        im = skimage.io.imread(os.path.join(dutils.IMAGENET_IMAGE_ROOT,imroot+'.JPEG'))
                    
                    savename2 = os.path.join(d2,f'im-{modelname}{methodnamej}{target_id}.png')
                    # import ipdb;ipdb.set_trace()
                    skimage.io.imsave(savename2,skimage.transform.resize(im,saliency_.shape[:2]))
                    print(savename)
                    # assert False
                    
                    #=============================                
                # assert False

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--operation',default=None,type=str)
    parser.add_argument('--modelname',default=None,type=str)
    parser.add_argument('--dataset',default=None,type=str)
    parser.add_argument('--modelnames',default=None,type=str,nargs='*')
    parser.add_argument('--datasets',default=None,type=str,nargs='*')    
    args = parser.parse_args()
    #~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
    if args.operation != 'visualize_cherry_picked':
        assert args.dataset is not None
        assert args.modelname is not None
    #~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
    methodnames= [
        'loadgen-gpnn-mycam',
        'gradcam',
    'gradcampp',
    'relevancecam',
    'scorecam',
    'layercam',
    'cameras',
    
    ]
    if args.operation == 'chattopadhyay':
        average_chattopadhyay = compare_chattopadhyay(
            methodnames, args.modelname,args.dataset
        )
        # import ipdb;ipdb.set_trace()
    elif args.operation == 'pointing_game':
        if 'loadgen-gpnn-mycam' in methodnames:
            methodnames[methodnames.index('loadgen-gpnn-mycam')]  = 'gpnn-mycam'
            
        average_pointing_game,average_pointing_game_difficult = compare_pointing_game(
            methodnames, args.modelname,args.dataset
        )
        """
        dump dictionary to json file
        """
        import json
        json_fname = f'benchmark/{args.operation}-{args.dataset}-{args.modelname}.json'
        with open(json_fname,'w') as f:
            json.dump(average_pointing_game,f)
            
        json_fname = f'benchmark/{args.operation}-difficult-{args.dataset}-{args.modelname}.json'
        with open(json_fname,'w') as f:
            json.dump(average_pointing_game_difficult,f)            
            
        # import ipdb;ipdb.set_trace()

    elif args.operation == 'easy_pointing_game':
        get_easy_pointing_game(methodnames,args.modelname,dataset =args.dataset,for_methodname='gpnn-mycam')
        pass
    elif args.operation == 'visualize_cherry_picked':
        # get_easy_pointing_game(methodnames,args.modelname,dataset =args.dataset,for_methodname='gpnn-mycam')
        visualize_cherry_picked(methodnames,args.modelnames,args.datasets)

    else:
        assert False,'operation not recognized'


    