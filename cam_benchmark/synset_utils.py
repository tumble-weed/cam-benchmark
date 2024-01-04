import os
import cam_benchmark.imagenet_synsets as imagenet_synsets
def get_synset_id(synset_name):
    import cam_benchmark.imagenet_synsets as imagenet_synsets
    label_vs_id = {more['label']: ('n'+more['id'][:-2]) for ix,more in imagenet_synsets.synsets.items()}
    synset_id = None
    imagenet_ix = None
    for ix,l in enumerate(label_vs_id):
        if synset_name in l:
            synset_id = label_vs_id[l]
            imagenet_ix = ix
            break
    print(l,synset_id)        
    # return synset_id,imagenet_ix
    return synset_id
def synset_id_to_imagenet_class_ix(synset_id):
    id_vs_class_ix = { ('n'+more['id'][:-2]):ix  for ix,more in imagenet_synsets.synsets.items()}
    return id_vs_class_ix[synset_id]
def synset_id_to_imagenet_class_name(synset_id):
    name_vs_class_ix = { ('n'+more['id'][:-2]):more['label']  for ix,more in imagenet_synsets.synsets.items()}
    return name_vs_class_ix[synset_id]
def class_id_to_synset_id(class_id):
    synset_id = imagenet_synsets.synsets[class_id]['id']    
    synset_id = 'n' + synset_id[:-len('-n')]
    return synset_id

def get_synset(synset_id):    
    if not os.path.isdir(f'benchmark/imagenet-synsets/{synset_id}'):
        if not os.path.exists(f'benchmark/imagenet-synsets/{synset_id}.tar'):
            assert synset_id == 'n02492035','only works with this particular synset :('
            assert False
            print(f'downloading {synset_id}.tar')
            os.system('gdown https://drive.google.com/uc?id=12jFYB400syhQ7qcDgovfc0vDPn2ac699')
        print(f'extracting {synset_id}.tar')
        os.system(f'tar -xvf benchmark/imagenet-synsets/{synset_id}.tar --one-top-level=benchmark/imagenet-synsets/{synset_id}')

