# from visualize_perturbation_metrics import get_metrics_for_1_image as p_get_metrics_for_1_image
# from visualize_chattopadhyay_metrics import   get_metrics_for_1_image as c_get_metrics_for_1_image
import colorful
import dutils
import register_ipdb
from benchmark import settings
from benchmark.analysis_utils import get_metrics_for_1_image,get_available_im_dirs, load_metrics,load_metrics_from_imdirs,get_avg_metrics
import numpy as np
import json
import jdata
import os
import pprint
import dutils   
def compare_on_single_image(
  metricnames,
  methodnames,
  modelname,
  dataset=None,
  results_dir = settings.METRICS_DIR_librecam,
  im_id = 0,
):
    print(colorful.green('visualizing many methods and metrics on a single image'))
    per_metric = {}
    for metricname in metricnames:
        per_metric[metricname] = {}
        for methodname in methodnames:
            loaded,imroot = get_metrics_for_1_image(im_id,dataset,methodname,modelname,metricname,results_dir)
            if metricname == 'perturbation':    
                '''
                dict_keys(['mrf_scores', 'lrf_scores', 'metricname', 'mrf_probability_increase', 'mrf_increase_indicator', 'mrf_binary_mask', 'mrf_perturbed', 'lrf_probability_increase', 'lrf_increase_indicator', 'lrf_binary_mask', 'lrf_perturbed', 'modelname', 'methodname', 'classname', 'delete_percentiles', 'target_id', 'saliency'])
                '''
                # import ipdb;ipdb.set_trace()
                keys = [
                    'mrf_scores','lrf_scores',
                    'mrf_probability_increase','lrf_probability_increase'
                        ]
                assert np.isclose(loaded['mrf_probability_increase'][0],0)
            elif metricname == 'chattopadhyay':
                # keys = loaded.keys()
                keys = ['normalized_drop_in_prob', 'normalized_rise_in_prob', 'increase_indicator', 'decrease_indicator']
                # pass    
            assert loaded['metricname'] == metricname
            loaded = {k:v for k,v in loaded.items() if k in keys}
            per_metric[metricname][methodname] = loaded
        # import ipdb;ipdb.set_trace()    
        print(colorful.green(f'{metricname}'))
        
        pp = pprint.PrettyPrinter(depth=4)
        print(colorful.salmon(pp.pformat(per_metric[metricname])))

        # print(colorful.salmon( jdata.dumps(per_metric[metricname],indent=4)))
    
def find_common_imroots_for_metric(dataset,metricname,methodnames,results_dir):
    common_directories = []
    # common_directories[metricname] = {}
    for methodname in methodnames:
        # loaded = get_metrics_for_1_image(im_id,dataset,methodname,modelname,metricname,results_dir)
        imdirs = get_available_im_dirs(dataset,methodname,modelname,metricname,results_dir)
        imroots = [imdir.rstrip('/').split('/')[-1] for imdir in imdirs]
        if not len(common_directories):
            common_directories = imroots
        common_directories = list(set(common_directories) & set(imroots))

        return common_directories

def compare_on_common_images(
  metricnames,
  methodnames,
  modelname,
  dataset=None,
  results_dir = settings.METRICS_DIR_librecam,
  im_id = 0,
):
    print(colorful.green('visualizing many methods and metrics on common images'))
    # per_metric = {}
    common_directories = {}
    common_imroots = {}
    dirs = {}
    for metricname in metricnames:
        # per_metric[metricname] = {}
        common_directories[metricname] = find_common_imroots_for_metric(dataset,metricname,methodnames,results_dir)
        common_imroots[metricname] = [os.path.split(d.rstrip(os.path.sep)) for d in common_directories[metricname]]
    assert all([len(common_directories[m]) > 0  for m in metricnames])
    print(f'common directories: {[len(common_directories[m]) for m in metricnames]}')
    # import ipdb;ipdb.set_trace()
    avg_metrics = {}
    metric_method_imdirs = {}
    metric_method_loaded = {}
    for metricname in metricnames:
        avg_metrics[metricname] = {}
        metric_method_imdirs[metricname] = {}
        metric_method_loaded[metricname] = {}
        for methodname in methodnames:
            more_imdirs = get_available_im_dirs(dataset,methodname,modelname,metricname,results_dir)
            imdirs = []
            for imdir in more_imdirs:
                this_imroot = imdir.rstrip(os.path.sep).split(os.path.sep)[-1]
                if this_imroot in common_directories[metricname]:
                    imdirs.append(imdir)
            
            metric_method_imdirs[metricname][methodname] = imdirs
            all_loaded = load_metrics_from_imdirs(imdirs)
            import ipdb;ipdb.set_trace()
            metric_method_loaded[metricname][methodname] = all_loaded
            if metricname == 'chattopadhyay':
                submetricnames=['normalized_drop_in_prob',
                'normalized_rise_in_prob',
                'increase_indicator',
                'decrease_indicator']
            
            avg_metrics[metricname][methodname] = get_avg_metrics(all_loaded,metricnames=submetricnames)
        
        if False and 'see':
            for metricname in metricnames:
                # common_imroots
                nimages = len(common_imroots[metricname])
                submetric = 'normalized_drop_in_prob'
                
                for i in range(nimages):
                    of_image = {}
                    for methodname in methodnames:
                        m = metric_method_loaded[metricname][methodname][i]
                        of_image[methodname] = m
                    best_method = min(of_image, key=lambda k: of_image[k][submetric])
                    if best_method == 'gradcam':
                        # import ipdb;ipdb.set_trace()
                        for methodname in methodnames:
                            s = of_image[methodname][submetric]
                            dutils.img_save(np.transpose(of_image[methodname]['masked'],(0,2,3,1))[0],f'{methodname}{str(s)}.png')
                        import ipdb;ipdb.set_trace()
    pp = pprint.PrettyPrinter(depth=4)
    print(colorful.salmon(pp.pformat(avg_metrics)))
    
    # import ipdb;ipdb.set_trace()            

metricnames = [
                # 'perturbation',
               'chattopadhyay'
               ]
methodnames = ['scorecam',
               'gradcam',
               'gradcampp',
               'relevancecam',
               
            #    'gpnn-gradcam',
            #    'gpnn-gradcampp',
            #    'gpnn-gradcam-uniform',
               
            #    'trivial-gradcam-hflip',
            #     'trivial-gradcam-noise',
            #     'trivial-gradcam-noise-hflip',
               ]

modelname = 'resnet50'

# compare_on_single_image(
#   metricnames,
#   methodnames,
#   modelname,
#   dataset=None,
#   results_dir = settings.METRICS_DIR_librecam,
#   im_id = 1,
# )


compare_on_common_images(
  metricnames,
  methodnames,
  modelname,
  dataset=None,
  results_dir = settings.METRICS_DIR_librecam,
  im_id = 0,
)