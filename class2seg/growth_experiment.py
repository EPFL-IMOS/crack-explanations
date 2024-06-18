import random
import data
from transforms import get_val_transform
import torch
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import skeletonize
import pandas as pd
import numpy as np
from fire import Fire

from eval_seg import cracks_per_patch_max_width
from model.activation_map.map_utils import map_factory
from model.classifier import Classifier
from model.explainer_classifier import ExplainerClassifier
from model.segmenter import Segmenter
from model.eval_utils import post_factory, simple_post, GMM_post, closing_post, area_opening_post, threshold_post
from transforms import AddInverse


SEED = 42
random.seed(SEED)


def growth_experiment(model, model_bcos, model_explainer, model_unet, dataset_neg, dataset_pos, i=0):

    n_steps = 5

    results = []

    # 1. Pick random damage-free background image from negative images
    idx_neg = random.randint(0, len(dataset_neg))
    img_neg, _, id_neg = dataset_neg[idx_neg]

    # 2. Pick random crack skeleton (with minimum length?) from positive masks
    idx_pos = random.randint(0, len(dataset_pos))
    img_pos, mask, id_pos = dataset_pos[idx_pos]
    # get mean crack color
    # print(img_pos.shape, mask.shape)
    # crack_color = (img_pos * mask).sum(dim=(1, 2)) / mask.sum()
    crack_color = img_pos.amin(dim=(1,2))
    # print(crack_color)
    mask = mask.float().numpy()

    # To keep only the largest crack
    # cam = (mask*255).astype(np.uint8)
    # # find all of the connected components (white blobs in your image).
    # # im_with_separated_blobs is an image where each detected blob has a different pixel value ranging from 1 to nb_blobs - 1.
    # _, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(cam)
    # # stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information.
    # # here, we're interested only in the size of the blobs, contained in the last column of stats.
    # sizes = stats[:, -1]
    # # the following lines result in taking out the background which is also considered a component, which I find for most applications to not be the expected output.
    # # you may also keep the results as they are by commenting out the following lines. You'll have to update the ranges in the for loop below.
    # sizes_with_indices = sorted(zip(sizes[1:], range(1, len(sizes))), reverse=True)
    # print(im_with_separated_blobs.shape)

    # mask = np.where(im_with_separated_blobs == sizes_with_indices[0][1], 1, 0)
    # mask = np.dstack((mask, mask, mask))
    # print(mask.shape)
    mask = skeletonize(mask).astype(float)

    # 3. Random rotation (optional)

    # 4. Generate sequence of images
    C = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    masks = [mask]
    image = img_neg.numpy().transpose(1, 2, 0)
    images = [np.where(np.dstack(tuple(mask for _ in range(C))) == 1.0, crack_color, image)]
    for step in range(1, n_steps+1):
        dilated_mask = cv2.dilate(masks[step-1], kernel)
        masks.append(dilated_mask)
        images.append(np.where(np.dstack(tuple(dilated_mask for _ in range(C))) == 1.0, crack_color, image))

    # 5. Apply all XAI methods, Evaluate and return true/estimated severity metrics

    def get_explanation(img, map, run, cam_method):
        if cam_method == "bcos_grad":
            img, _ = AddInverse()(img, target=None)
            scores = model_bcos(img.unsqueeze(0)).softmax(1)
            attribution = run(map=map, image=img.unsqueeze(0), model=model_bcos)
        else:
            scores = model(img.unsqueeze(0)).softmax(1)
            attribution = run(map=map, image=img.unsqueeze(0), model=model)
        cams = []
        cam_m = attribution[0]
        for post_method in ("simple", "simple+final"):
            for method in post_method.split("+"):
                if method == "final":
                    cam_m = closing_post(cam_m, radius=5)
                    cam_m = area_opening_post(cam_m, min_area=50)
                    cam_m = closing_post(cam_m, radius=25)
                else:
                    cam_m = post_factory(method=method)(cam=cam_m, **kwargs)
            cams.append(cam_m)
        return cams[0], cams[1], scores[0,1].item()

    def get_explanation_explainer(img):
        scores = model(img.unsqueeze(0)).softmax(1)
        targets = torch.zeros((img.shape[0], 2))
        targets[:, 1] = 1.0
        with torch.no_grad():
            _, _, attribution, _, _ = model_explainer(img.unsqueeze(0), targets)
        cams = []
        cam_m = attribution.numpy()[0]
        for post_method in ("simple", "simple+final"):
            for method in post_method.split("+"):
                if method == "final":
                    # cam_m = closing_post(cam_m, radius=5)
                    # cam_m = area_opening_post(cam_m, min_area=50)
                    cam_m = closing_post(cam_m, radius=25)
                else:
                    cam_m = post_factory(method=method)(cam=cam_m, **kwargs)
            cams.append(cam_m)
        return cams[0], cams[1], scores[0,1].item()

    def get_segmentation(img):
        scores = model(img.unsqueeze(0)).softmax(1)
        with torch.no_grad():
            cam_m = model_unet(img.unsqueeze(0)).sigmoid().squeeze(1).detach().numpy()[0]
        cam_m_post = threshold_post(cam_m)
        return cam_m, cam_m_post, scores[0,1].item()

    for cam_method in ("inputxgrad", "intgrad", "deeplift", "deepliftshap", "gradientshap"): # "lrpVGG", "bcos_grad", "explainer", "unet"

        if cam_method == "lrpVGG":
            kwargs = {
                "rules": "segmentation"
            }
        elif cam_method in ("deeplift", "intgrad"):
            kwargs = {
                "baseline": "mean_baseline.pth"
            }
        elif cam_method in ("deepliftshap", "gradientshap"):
            kwargs = {
                "baseline": "baselines.pth"
            }
        else:
            kwargs = {}

        if cam_method != "explainer" and cam_method != "unet":
            map, run = map_factory(method=cam_method, model=model, **kwargs)

        fig, ax = plt.subplots(n_steps, 4, figsize=(8,11))
        cpp_true = []
        width_true = []
        area_true = []
        cpp_pred = []
        width_pred = []
        area_pred = []
        ax[0][0].set_title("Image", size=14)
        ax[0][1].set_title("Ground truth", size=14)
        ax[0][2].set_title("Thresholding", size=14)
        ax[0][3].set_title("Morph. closing", size=14)

        for n in range(n_steps):
            # print(cam_method, n)

            ax[n][0].imshow(images[n][...,:3])
            ax[n][1].imshow(masks[n], interpolation="bilinear", cmap="gray")

            cpp_true, width_true = cracks_per_patch_max_width(masks[n], min_area=50)
            area_true = masks[n].sum()

            if cam_method == "unet":
                expl_simple, expl_final, conf = get_segmentation(torch.Tensor(images[n].transpose(2, 0, 1)))
            elif cam_method != "explainer":
                expl_simple, expl_final, conf = get_explanation(torch.Tensor(images[n].transpose(2, 0, 1)), map, run, cam_method)
            else:
                expl_simple, expl_final, conf = get_explanation_explainer(torch.Tensor(images[n].transpose(2, 0, 1)))

            ax[n][2].imshow(expl_simple, interpolation="nearest", cmap="gray")
            ax[n][3].imshow(expl_final, interpolation="nearest", cmap="gray")
            cpp, width = cracks_per_patch_max_width(expl_final, min_area=50)
            area = expl_final.sum()

            results.append({
                "cam_method": cam_method,
                "idx_neg": idx_neg,
                "idx_pos": idx_pos,
                "experiment": i,
                "step": n,
                "cpp_true": cpp_true,
                "width_true": width_true,
                "area_true": area_true,
                "cpp_pred": cpp,
                "width_pred": width,
                "area_pred": area,
                "confidence": conf
            })

        [axis.axis('off') for axis in ax.ravel()];
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.1, hspace=0);
        plt.savefig(f"tmp/growth-{cam_method}-{i}-{idx_neg}-{idx_pos}.svg", bbox_inches="tight")
        plt.close()

    return results


def main(
        dataset="DICCracksDataset",
        data_path="/home/florent/imos/data/DIC_crack_dataset",
        fold="test",
        imagesize: int = 256,
        **kwargs
    ):

    Dataset = data.__dict__[dataset]

    dataset_neg = Dataset(
        root=data_path,
        fold=fold,
        transform=get_val_transform(add_inverse=False, size=imagesize),
        segmentation=False,
        negative_only=True
    )

    dataset_pos = Dataset(
        root=data_path,
        fold=fold,
        transform=get_val_transform(add_inverse=False, size=imagesize),
        segmentation=True,
        positive_only=True
    )

    num_classes = 2
    model_type = "vgg11_128"
    model_path = "checkpoints/output/dic-vgg11-128-alphabeta/epoch=283-val_accuracy=0.96.ckpt"
    # model_type = "vit"
    # model_path = "checkpoints/output-vit/vitb16-lr5/epoch=23-val_accuracy=0.92.ckpt"
    model = Classifier.load_from_checkpoint(model_path, model=model_type, num_classes=num_classes).eval()

    # model_type = "vgg11_128_bcos"
    # model_path = "checkpoints/output/dic-vgg11-128-bcos/epoch=96-val_accuracy=0.82.ckpt"
    # model_bcos = Classifier.load_from_checkpoint(model_path, model=model_type, num_classes=num_classes).eval()
    model_bcos = None

    # model_explainer = ExplainerClassifier(
    #     num_classes=2,
    #     classifier_type="vgg11_128",
    #     explainer_type="unet11"
    # ).load_from_checkpoint(
    #     "checkpoints/output/dic-explainer-unet11-vgg11-128-minmaxarea/last.ckpt",
    #     num_classes=2,
    #     classifier_type="vgg11_128",
    #     explainer_type="unet11"
    # ).eval()
    model_explainer = None

    # num_classes = 1
    # model_type = "unet11"
    # model_path = "checkpoints/output/dic-unet11/epoch=141-val_f1=0.79.ckpt"
    # model_unet = Segmenter.load_from_checkpoint(model_path, model=model_type, num_classes=num_classes).eval()
    model_unet = None

    N_EXPERIMENTS = 100

    results = []
    for i in range(N_EXPERIMENTS):
        print(f"{i+1}/{N_EXPERIMENTS}")
        results.extend(growth_experiment(model, model_bcos, model_explainer, model_unet, dataset_neg, dataset_pos, i))

    df = pd.DataFrame(results)
    print(df)
    df.to_csv("results_growth.csv")

if __name__ == "__main__":
    Fire(main)
