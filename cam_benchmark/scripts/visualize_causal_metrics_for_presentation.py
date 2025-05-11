# https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/CAM%20Metrics%20And%20Tuning%20Tutorial.ipynb
import dutils
dutils.init()
#sys.path.append('/root/evaluate-saliency-4/pytorch-grad-cam/pytorch_grad_cam')
sys.path.append('/root/evaluate-saliency-4/pytorch-grad-cam')
import warnings
warnings.filterwarnings('ignore')
from torchvision import models
import numpy as np
import cv2
import requests
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from PIL import Image
from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst, ROADLeastRelevantFirst
from pytorch_grad_cam.metrics.road import ROADLeastRelevantFirstAverage, ROADMostRelevantFirstAverage
from pytorch_grad_cam.metrics.road import ROADCombined
from pytorch_grad_cam.metrics.cam_mult_image import DropInConfidence, IncreaseInConfidence
from pytorch_grad_cam.sobel_cam import sobel_cam
mydir = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(mydir,'visualize_deletion_game_imputation')
os.makedirs(SAVE_DIR, exist_ok=True)
#=============================================================================
model = models.resnet50(pretrained=True)
model.eval()
image_url = "https://th.bing.com/th/id/R.94b33a074b9ceeb27b1c7fba0f66db74?rik=wN27mvigyFlXGg&riu=http%3a%2f%2fimages5.fanpop.com%2fimage%2fphotos%2f31400000%2fBear-Wallpaper-bears-31446777-1600-1200.jpg&ehk=oD0JPpRVTZZ6yizZtGQtnsBGK2pAap2xv3sU3A4bIMc%3d&risl=&pid=ImgRaw&r=0"
img = np.array(Image.open(requests.get(image_url, stream=True).raw))
img = cv2.resize(img, (224, 224))
img = np.float32(img) / 255
input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#=============================================================================
# The target for the CAM is the Bear category.
# As usual for classication, the target is the logit output
# before softmax, for that category.
targets = [ClassifierOutputTarget(295)]
target_layers = [model.layer4]
with GradCAM(model=model, target_layers=target_layers) as cam:
    grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
    cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
    cam_rgb = show_cam_on_image(np.zeros(img.shape), grayscale_cams[0, :], use_rgb=True)
cam = np.uint8(255*grayscale_cams[0, :])
cam = cv2.merge([cam, cam, cam])
images = np.hstack((np.uint8(255*img), cam , cam_image))
#Image.fromarray(images)
dutils.img_save(cam_rgb[None],os.path.join(SAVE_DIR,'bear_gradcam.png'),use_matplotlib=False)
dutils.img_save(cam_image[None],os.path.join(SAVE_DIR,'bear_gradcam_overlay.png'),use_matplotlib=False)
#=====================================================================================
# Now lets see how to evaluate this explanation:
if  'chattopadhyay' and False:

    # For the metrics we want to measure the change in the confidence, after softmax, that's why
    # we use ClassifierOutputSoftmaxTarget.
    targets = [ClassifierOutputSoftmaxTarget(295)]
    cam_metric = CamMultImageConfidenceChange()
    scores, visualizations = cam_metric(input_tensor, grayscale_cams, targets, model, return_visualization=True)
    score = scores[0]
    visualization = visualizations[0].cpu().numpy().transpose((1, 2, 0))
    visualization = deprocess_image(visualization)
    print(f"The confidence increase percent: {100*score}")
    print("The visualization of the pertubated image for the metric:")
    #Image.fromarray(visualization)

    dutils.img_save(visualization[None],os.path.join(SAVE_DIR,'bear_gradcam_evaluation.png'),use_matplotlib=False)
    #dutils.img_save(cam_image[None],os.path.join(SAVE_DIR,'bear_gradcam_overlay.png'),use_matplotlib=False)
    p46()
if 'chattopadhyay' and False:
#=====================================================================================
    print("Drop in confidence", DropInConfidence()(input_tensor, grayscale_cams, targets, model))
    print("Increase in confidence", IncreaseInConfidence()(input_tensor, grayscale_cams, targets, model))
#=====================================================================================
if 'chattopadhyay' and False:
    inverse_cams = 1 - grayscale_cams
    scores, visualizations = CamMultImageConfidenceChange()(input_tensor, inverse_cams, targets, model, return_visualization=True)
    score = scores[0]
    visualization = visualizations[0].cpu().numpy().transpose((1, 2, 0))
    visualization = deprocess_image(visualization)
    print(f"The confidence increase percent: {score}")
    print("The visualization of the pertubated image for the metric:")
    Image.fromarray(visualization)
#=====================================================================================
imputations = ['zero','road']
for imputation in imputations:
# Is this the deletion game?
    imputation_stub = '' if imputation == 'zero' else f'_{imputation}'
    percs = [0,25,50,75,100]
    causal_scores = {'deletion':[],'insertion':[]}
    for perc in percs:
        thresholded_cam = grayscale_cams < np.percentile(grayscale_cams, perc)
        if imputation == 'zero':
            scores, visualizations = CamMultImageConfidenceChange()(input_tensor, thresholded_cam, targets, model, return_visualization=True)
        elif imputation == 'road':
            #p46()
            cam_metric = ROADLeastRelevantFirst(percentile=perc)
            scores, visualizations = cam_metric(input_tensor, grayscale_cams, targets, model, return_visualization=True)
        score = scores[0]
        visualization = visualizations[0].cpu().numpy().transpose((1, 2, 0))
        visualization = deprocess_image(visualization)
        print(f"The confidence increase: {score}")
        print("The visualization of the pertubated image for the metric:")
        Image.fromarray(visualization)
        dutils.img_save(visualization[None],os.path.join(SAVE_DIR,f'bear_gradcam_deletion{imputation_stub}_{perc}.png'),use_matplotlib=False)
        dutils.img_save(thresholded_cam.astype(np.float32)[None],os.path.join(SAVE_DIR,f'bear_gradcam_deletion{imputation_stub}_raw_{perc}.png'),use_matplotlib=False)
        causal_scores['deletion'].append(score)
#=====================================================================================
    for perc in percs:
        thresholded_cam = grayscale_cams > np.percentile(grayscale_cams, perc)
        if imputation == 'zero':
            scores, visualizations = CamMultImageConfidenceChange()(input_tensor, thresholded_cam, targets, model, return_visualization=True)
        elif imputation == 'road':
            #p46()
            cam_metric = ROADMostRelevantFirst(percentile=perc)
            scores, visualizations = cam_metric(input_tensor, grayscale_cams, targets, model, return_visualization=True)

        score = scores[0]
        visualization = visualizations[0].cpu().numpy().transpose((1, 2, 0))
        visualization = deprocess_image(visualization)
        print(f"The confidence increase: {score}")
        print("The visualization of the pertubated image for the metric:")
        Image.fromarray(visualization)
        dutils.img_save(visualization[None],os.path.join(SAVE_DIR,f'bear_gradcam_insertion{imputation_stub}_{100-perc}.png'),use_matplotlib=False)
        dutils.img_save(thresholded_cam.astype(np.float32)[None],os.path.join(SAVE_DIR,f'bear_gradcam_insertion_raw{imputation_stub}_{100-perc}.png'),use_matplotlib=False)
        causal_scores['insertion'].append(score)

#dutils.save_plot(causal_scores['deletion'],'',,x=percs)
#dutils.save_plot(causal_scores['insetion'],'',os.path.join(SAVE_DIR,'insertion_graph.png'),x=percs)
#===========================================================
    savename = os.path.abspath(os.path.join(SAVE_DIR,f'deletion_graph{imputation_stub}.png'))
    title = ''
    print(colorful.tan(f"saving {os.path.realpath(savename)}"))
# /root/debug/myplot.png
# myplot.png
# http://localhost:10000/debug/myplot.png
# p46()
    assert savename.startswith(os.environ['httproot'])
    saveurl = os.environ['httpurl']+'/'+savename[len(os.environ['httproot']):]
    print(colorful.tan(f'saving {saveurl}'))
#=====================================================
    f = plt.figure()
    #plt.plot(100  - np.array(percs),causal_scores['deletion'],color='black')
    #plt.plot(np.array(percs),causal_scores['deletion'],color='black')
    #plt.plot(100 - np.array(percs),-np.array(causal_scores['deletion']),color='black')
    plt.plot(np.array(percs),-np.array(causal_scores['deletion']),color='black')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.xlim(0,100)
    plt.gca().set_facecolor('magenta')
    plt.gca().patch.set_alpha(0.2)
    plt.show()
    plt.savefig(savename)
    plt.close()
    p46()
#===========================================================
    savename = os.path.abspath(os.path.join(SAVE_DIR,f'insertion_graph{imputation_stub}.png'))
    title = ''
    print(colorful.tan(f"saving {os.path.realpath(savename)}"))
# /root/debug/myplot.png
# myplot.png
# http://localhost:10000/debug/myplot.png
# p46()
    assert savename.startswith(os.environ['httproot'])
    saveurl = os.environ['httpurl']+'/'+savename[len(os.environ['httproot']):]
    print(colorful.tan(f'saving {saveurl}'))
#=====================================================
    f = plt.figure()
    #plt.plot(100 - np.array(percs),causal_scores['insertion'],color='black')
    plt.plot(np.array(percs),causal_scores['insertion'],color='black')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.xlim(0,100)
    plt.gca().set_facecolor('orange')
    plt.gca().patch.set_alpha(0.2)
    plt.show()
    plt.savefig(savename)
    plt.close()
p46()
import sys;sys.exit()
#=====================================================================================
#=====================================================================================
if 'sobel' and False:

    sobel_cam_grayscale = sobel_cam(np.uint8(img * 255))
    thresholded_cam = sobel_cam_grayscale < np.percentile(sobel_cam_grayscale, 75)

    cam_metric = CamMultImageConfidenceChange()
    scores, visualizations = cam_metric(input_tensor, [thresholded_cam], targets, model, return_visualization=True)
    score = scores[0]
    visualization = visualizations[0].cpu().numpy().transpose((1, 2, 0))
    visualization = deprocess_image(visualization)
    print(f"The confidence increase: {score}")
    print("The visualization of the pertubated image for the metric:")
    sobel_cam_rgb = cv2.merge([sobel_cam_grayscale, sobel_cam_grayscale, sobel_cam_grayscale])
    Image.fromarray(np.hstack((sobel_cam_rgb, visualization)))

#=====================================================================================
if 'road' and False:

    cam_metric = ROADMostRelevantFirst(percentile=75)
    scores, visualizations = cam_metric(input_tensor, grayscale_cams, targets, model, return_visualization=True)
    score = scores[0]
    visualization = visualizations[0].cpu().numpy().transpose((1, 2, 0))
    visualization = deprocess_image(visualization)
    print(f"The confidence increase when removing 25% of the image: {score}")

    cam_metric = ROADMostRelevantFirst(percentile=90)
    scores, visualizations = cam_metric(input_tensor, grayscale_cams, targets, model, return_visualization=True)
    score = scores[0]
    visualization_10 = visualizations[0].cpu().numpy().transpose((1, 2, 0))
    visualization_10 = deprocess_image(visualization_10)
    print(f"The confidence increase when removing 10% of the image: {score}")
    print("The visualizations:")
    Image.fromarray(np.hstack((visualization, visualization_10)))


#=====================================================================================


    cam_metric = ROADMostRelevantFirstAverage(percentiles=[20, 40, 60, 80])
    scores = cam_metric(input_tensor, grayscale_cams, targets, model)
    print(f"The average confidence increase with ROAD accross 4 thresholds: {scores[0]}")
    scores = cam_metric(input_tensor, [sobel_cam_grayscale], targets, model)
    print(f"The average confidence increase for Sobel edge detection with ROAD accross 4 thresholds: {scores[0]}")


#=====================================================================================


    cam_metric = ROADMostRelevantFirstAverage(percentiles=[20, 40, 60, 80])
    scores = cam_metric(input_tensor, grayscale_cams * 0, targets, model)
    print(f"Empty CAM, Most relevant first avg confidence increase with ROAD accross 4 thresholds: {scores[0]}")


#=====================================================================================
    cam_metric = ROADLeastRelevantFirstAverage(percentiles=[20, 40, 60, 80])
    scores = cam_metric(input_tensor, grayscale_cams * 0, targets, model)
    print(f"Empty CAM, Least relevant first avg confidence increase with ROAD accross 4 thresholds: {scores[0]}")

#=====================================================================================
    cam_metric = ROADCombined(percentiles=[20, 40, 60, 80])
    scores = cam_metric(input_tensor, grayscale_cams * 0, targets, model)
    print(f"Empty CAM, Combined metric avg confidence increase with ROAD accross 4 thresholds (positive is better): {scores[0]}")

#=====================================================================================
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenGradCAM, AblationCAM, RandomCAM

# Showing the metrics on top of the CAM :
def visualize_score(visualization, score, name, percentiles):
    visualization = cv2.putText(visualization, name, (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, "(Least first - Most first)/2", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, f"Percentiles: {percentiles}", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, "Remove and Debias", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    visualization = cv2.putText(visualization, f"{score:.5f}", (10, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
    return visualization

def benchmark(input_tensor, target_layers, eigen_smooth=False, aug_smooth=False, category=281):
    #p46()
    #methods = [("GradCAM", GradCAM(model=model, target_layers=target_layers, use_cuda=True)),
    #           ("GradCAM++", GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)),
    #           ("EigenGradCAM", EigenGradCAM(model=model, target_layers=target_layers, use_cuda=True)),
    #           ("AblationCAM", AblationCAM(model=model, target_layers=target_layers, use_cuda=True)),
    #           ("RandomCAM", RandomCAM(model=model, target_layers=target_layers, use_cuda=True))]
    methods = [("GradCAM", GradCAM(model=model, target_layers=target_layers, )),
               ("GradCAM++", GradCAMPlusPlus(model=model, target_layers=target_layers, )),
               ("EigenGradCAM", EigenGradCAM(model=model, target_layers=target_layers, )),
               ("AblationCAM", AblationCAM(model=model, target_layers=target_layers, )),
               ("RandomCAM", RandomCAM(model=model, target_layers=target_layers, ))]

    cam_metric = ROADCombined(percentiles=[20, 40, 60, 80])
    targets = [ClassifierOutputTarget(category)]
    metric_targets = [ClassifierOutputSoftmaxTarget(category)]

    visualizations = []
    percentiles = [10, 50, 90]
    for name, cam_method in methods:
        with cam_method:
            attributions = cam_method(input_tensor=input_tensor,
                                      targets=targets, eigen_smooth=eigen_smooth, aug_smooth=aug_smooth)
        attribution = attributions[0, :]
        scores = cam_metric(input_tensor, attributions, metric_targets, model)
        score = scores[0]
        visualization = show_cam_on_image(cat_and_dog, attribution, use_rgb=True)
        visualization = visualize_score(visualization, score, name, percentiles)
        visualizations.append(visualization)
    return Image.fromarray(np.hstack(visualizations))

cat_and_dog_image_url = "https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/both.png"
cat_and_dog = np.array(Image.open(requests.get(cat_and_dog_image_url, stream=True).raw))
cat_and_dog = np.float32(cat_and_dog) / 255
input_tensor = preprocess_image(cat_and_dog, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
target_layers = [model.layer4]

model.cuda()
input_tensor = input_tensor.cuda()
np.random.seed(42)
benchmark(input_tensor, target_layers, eigen_smooth=False, aug_smooth=False)

#=====================================================================================
target_layers = [model.layer4[-2]]
benchmark(input_tensor, target_layers)
#=====================================================================================
np.random.seed(0)
benchmark(input_tensor, target_layers)
#=====================================================================================
# Let's look how it looks for one of the dog categories (that the model is much less confident about)
np.random.seed(0)
benchmark(input_tensor, target_layers, category=246)
#=====================================================================================
