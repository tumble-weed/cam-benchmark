import time
import pickle
import benchmark.settings as settings
import os
class Timer():
    timers = {}
    def __init__(self,name):
        self.name = name
        self.n_calls = 0
        self.total_time = 0
        self.avg_time = 0
        self.tic = None

        pass
    @classmethod
    def get(cls,name):
        if name not in cls.timers:
            timer = cls(name)
            cls.timers[name] = timer
        timer = cls.timers[name]
        return timer
    def __enter__(self):
        self.n_calls += 1
        self.tic = time.time()
        return self
    def __exit__(self,exec_type,exec_value,exec_tb):
        toc = time.time()
        elapsed = toc - self.tic
        self.total_time += elapsed
        self.avg_time = self.total_time/self.n_calls
        self.tic = None
        pass

def save_im_data(savename,P,ref,trends,patch_theta0,importances):
    import inspect
    if False:
        P_dict = {}
        for name in dir(P):
            if any([
                name.startswith("__"),
    #             isinstance(item,ModuleType),
                inspect.ismodule(P.__dict__[name]),
            ]):
                continue
            P_dict[name] = P.__dict__[name]
        print(P_dict)
        with open(savename,'wb') as f:
            pickle.dump({
                        'parameters':P_dict,
                        'ref':tensor_to_numpy(ref),
                        'trends':trends,
                        'patch_theta0':tensor_to_numpy(patch_theta0),
                        'importances':(importances).__getstate__()
                        },f)
    # import sys;sys.exit()

def create_dir(sub_dir,root_dir=settings.RESULTS_DIR):
    full_dir = os.path.join(root_dir,sub_dir)
    os.system(f'rm -rf {full_dir}')
    os.makedirs(full_dir)
    return full_dir


def create_im_save_dir(experiment_name='imagenet-saliency',root_dir=settings.RESULTS_DIR,impath=None):
    save_dir = os.path.join(root_dir,experiment_name)
    imroot = os.path.split(impath)[-1].split('.')[0]
    print(imroot)
    im_save_dir = os.path.join(save_dir,imroot)
    os.system(f'rm -rf {im_save_dir}')
    os.makedirs(im_save_dir)
    return im_save_dir

class ChangeDir():
    def __init__(self,d):
        self.dir0 = os.getcwd()
        self.d = d
        pass
    def __enter__(self):
        os.chdir(self.d)
        pass
    def __exit__(self,*args):
        os.chdir(self.dir0)
        pass

import sys
class AddPath():
    def __init__(self,d):
        self.d = d

        pass
    def __enter__(self):
        self.at = len(sys.path)
        sys.path.append(self.d)        
        pass
    def __exit__(self,*args):
        del sys.path[self.at]
        pass
