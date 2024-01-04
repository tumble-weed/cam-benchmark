import dutils
dutils.init()
import glob
def delete(ref,mask,ratio_retained):
    masked = dutils.hardcode(masked = torch.zeros_like(ref))
    return masked

def run_deletion_game(model,ref,target_id,
mask,ratios_retained,batch_size=dutils.TODO):
    device = ref.device
    deleted_images = torch.zeros((len(ratios_retained),) + ref.shape[1:],device=device)
    ref_scores = model(ref)
    ref_probs = torch.softmax(ref_scores,dim=1)
    ref_probs = ref_probs[:,target_id]
    for i,ratio_retained in enumerate(ratios_retained):
        deleted_ref = delete(ref,mask,ratio_retained)
        deleted_images[i:i+1] = deleted_ref
    assert deleted_images.shape[0] <= batch_size, 'implement batched forward'
    scores = model(deleted_images)
    probs = torch.softmax(scores,dim=1)
    probs = probs[:,target_id]
    '''
    # (1,20,1,1)
    # (1,3,300,500) --> (1,20,2,5)
    # (1,1000) 
    '''
    # model(deleted_ref)
    # ref = dutils.hardcode(masked = torch.zeros_like(ref))
    dutils.pause()
    pass
def main():
    model = dutils.hardcode(model = torchvision.models.vgg16(pretrained=True))
    batch_size = 32
    method = dutils.hardcode(method = "dummy")
    arch = dutils.hardcode(arch= "resnet50")
    dataset = dutils.hardcode(dataset= "voc_2007")

    root_results_dir = dutils.hardcode(root_results_dir="/root/bigfiles/other/results-torchray")
    methoddir = os.path.join(root_results_dir,f'{dataset}-{method}-{arch}')
    pattern = os.path.join(methoddir,'*','*.xz') 
    xzfiles = glob.glob(pattern)
    # xzfiles = list(sorted(glob.glob(os.path.join(methoddir,'*','*.xz'))))
    for xzfile in dutils.trunciter(xzfiles,enabled=True,max_iter=10):
        print(xzfile)
        ref = dutils.hardcode(ref = torch.randn(1,3,224,224))
        target_id = dutils.hardcode(target_id = 29)
        mask = dutils.hardcode(mask=torch.rand(1,1,224,224))
        ratios_retained = dutils.hardcode(ratios_retained=np.linspace(0,1,10))

        run_deletion_game(model,ref,target_id,
            mask,ratios_retained,batch_size=batch_size)
        # break
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