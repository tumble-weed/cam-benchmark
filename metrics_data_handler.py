import os
import glob
from benchmark import settings
import lzma
import pickle
from benchmark.benchmark_utils import create_dir
class MetricsDataHandler():
    def __init__(self,
    dataset,
    metricname,
    methodname,
    modelname,
    metrics_root_dir = settings.METRICS_DIR_librecam):
        self.metrics_root_dir = metrics_root_dir
        self.dataset = dataset
        self.metricname = metricname
        self.methodname = methodname
        self.modelname = modelname
        self.dir = os.path.join(metrics_root_dir,
        f'{dataset}-{metricname}-{methodname}-{modelname}'
        )
    def clear_create_dir(self,imroot):
        # import ipdb;ipdb.set_trace()
        save_dir = create_dir(os.path.join(self.dir,imroot),root_dir='/',purge=True)        
    def dump(self,imroot,classname,target_id,data,purge=False):
        # imdir= os.path.join(self.dir,imroot)
        # import ipdb;ipdb.set_trace()
        # save_dir = create_dir(os.path.join(self.dir,imroot),root_dir=self.metrics_root_dir)        
        save_dir = create_dir(os.path.join(self.dir,imroot),root_dir='/',purge=purge)        
        # import ipdb;ipdb.set_trace()
        savename =  os.path.join(save_dir,f'{classname}{target_id}.xz')    
        print('!!!!!!!!!!!!')
        print(savename)
        print('!!!!!!!!!!!!')
        # import ipdb;ipdb.set_trace()
        with lzma.open(savename,'wb') as f:
            # import ipdb;ipdb.set_trace()
            pickle.dump(data,f)        
        
    def load_all_images_and_objects(self):
        available_imdirs = glob.glob(os.path.join(self.dir,'*/'))
        metrics_all_images = {}
        for imdir in available_imdirs:
            imroot = os.path.basename(imdir.rstrip(os.path.sep))
            metrics_one_image = self.load_all_objects(imroot)
            metrics_all_images[imroot] = metrics_one_image
        return metrics_all_images

    def load_all_objects(self,imroot):
        imdir= os.path.join(self.dir,imroot)
        pklnames = glob.glob(os.path.join(imdir,'*.xz'))
        """
        if imroot == '000001':
            import ipdb;ipdb.set_trace()
        """
        all_loaded = []
        # print(pklnames)
        print(imdir)
        for pklname in pklnames:
            with lzma.open(pklname) as f:
                
                loaded = pickle.load(f)
                all_loaded.append(loaded)
        return all_loaded
    def load_one_object(self,imroot,classname,target_id):
        imdir= os.path.join(self.dir,imroot)
        pklname = os.path.join(imdir,f'{classname}{target_id}.xz')
        # import ipdb; ipdb.set_trace()   
        if not os.path.exists(pklname):
            return None
        try:
            with lzma.open(pklname,'rb') as f:
                loaded = pickle.load(f)
        except (EOFError,lzma.LZMAError) as e:
            print(f'{pklname} corrpupt')
            return None
        return loaded
