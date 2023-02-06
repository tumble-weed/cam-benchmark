##############################################################################################
#%%
# import ipdb;ipdb.set_trace()
# import os
# os.chdir('/root/evaluate-saliency-4/jigsaw')    
import register_ipdb
import colorful
import torch
import torchvision
import numpy as np
from benchmark.benchmark_utils import ChangeDir,AddPath
import skimage.io
from PIL import Image
import sys
from cnn import get_target_id
import os
from pydoc import importfile
from benchmark import settings
from collections import defaultdict
import pickle
import glob
import dutils
from matplotlib import pyplot as plt
IMAGENET_ROOT = '/root/bigfiles/dataset/imagenet'
tensor_to_numpy = lambda t:t.detach().cpu().numpy()
##############################################################################################
def load_metrics(load_dir):
  # print(im_save_dir )
  # im_load_dir = os.path.join(settings.RESULTS_DIR_librecam,f'{metricname}-{methodname}-{modelname}',)
  # imroot = os.path.basename(im_save_dir.rstrip(os.path.sep))
  # load_dir = create_dir(os.path.join('metrics',f'{metricname}-{methodname}-{modelname}',imroot),root_dir=settings.RESULTS_DIR_librecam)
  if False:
      loadname = os.path.join(load_dir,f'{classname}{target_id}.pkl')
      with open(savename,'wb') as f:
          pickle.dump(metric_data,f)
  # print(load_dir)
  pklnames = list(glob.glob(os.path.join(load_dir,'*.pkl')))
  # print(pklnames)
  assert len(pklnames) == 1
  pklname = pklnames[0]
  with open(pklname,'rb') as f:
    loaded = pickle.load(f)  
  # print(loaded.keys())

  return loaded

def get_available_im_dirs(dataset,methodname,modelname,metricname,results_dir):
  dataset_stub = dataset
  if dataset in ['imagenet',None]:
    dataset_stub = ''
  parts = [dataset_stub,metricname,methodname,modelname]
  parts = [el for el in parts if len(el) > 0]
  experiment_dir = os.path.join(results_dir,'-'.join(parts))
  # print(experiment_dir)
  im_load_dirs = list(sorted(glob.glob(os.path.join(experiment_dir,"*/"))))
  return im_load_dirs

def get_metrics_for_1_image(im_id,dataset,methodname,modelname,metricname,results_dir):
  im_load_dirs = get_available_im_dirs(dataset,methodname,modelname,metricname,results_dir)
  im_load_dir = im_load_dirs[im_id]
  imroot = os.path.basename(im_load_dir.rstrip(os.path.sep))
  loaded = load_metrics(im_load_dir)
  return loaded,imroot

def load_metrics_from_imdirs(available_im_dirs):
    all_loaded = []
    for im_load_dir in available_im_dirs:
        loaded = load_metrics(im_load_dir)
        all_loaded.append(loaded)            
    return all_loaded
  
def get_avg_metrics(all_loaded,metricnames):
    per_metric = {}
    for k in metricnames:
        # assert isinstance(loaded[k],dict)
        # thresholds = sorted(loaded[k].keys())
        measurements = np.array([loaded[k] for loaded in all_loaded])
        assert not np.isnan(measurements).any()
        assert not np.isinf(measurements).any()
        per_metric[k] = measurements.mean()
    return per_metric
