import os
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import figure
from scipy.signal import medfilt2d
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import cv2

from constants import SEED
from model.beta import BetaMixtureModel


def quantile_post(cam: np.ndarray, quantile: float = 0.8) -> np.ndarray:
    return np.where(cam > np.quantile(cam, q=quantile), 1, 0)


def threshold_post(cam: np.ndarray, threshold: float = 0.5, **kwargs) -> np.ndarray:
    return np.where(cam > threshold, 1, 0)


def simple_post(cam: np.ndarray, threshold: float = 1e-6, **kwargs) -> np.ndarray:

    norm = MinMaxScaler()
    shape_cam = cam.shape
    cam = cam.flatten().reshape(-1, 1)
    cam = norm.fit_transform(cam)
    cam = cam.reshape(shape_cam)

    past_mean = -float("inf")
    mean = np.mean(cam)
    cam_left = cam[cam < mean]
    cam_right = cam[cam >= mean]

    while abs(mean - past_mean) >= threshold:
        past_mean = mean
        mean = (np.mean(cam_left) + np.mean(cam_right)) / 2

        cam_left = cam[cam < mean]
        cam_right = cam[cam >= mean]

    map_post = np.where(cam > mean, 1, 0)
    if map_post.mean() > 0.5:
        map_post = 1 - map_post
    return map_post


def closing_post(cam: np.ndarray, structuring_element="ellipse", radius=3, iterations=1, **kwargs) -> np.ndarray:

    map_post = cam.astype(np.uint8) * 255

    if structuring_element == "rect":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (radius,radius))
    elif structuring_element == "ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius,radius))
    elif structuring_element == "cross":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (radius,radius))

    return cv2.morphologyEx(map_post, cv2.MORPH_CLOSE, kernel, iterations) / 255


def opening_post(cam: np.ndarray, structuring_element="ellipse", radius=3, iterations=1, **kwargs) -> np.ndarray:

    map_post = cam.astype(np.uint8) * 255

    if structuring_element == "rect":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (radius,radius))
    elif structuring_element == "ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius,radius))
    elif structuring_element == "cross":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (radius,radius))

    return cv2.morphologyEx(map_post, cv2.MORPH_OPEN, kernel, iterations) / 255


def GMM_post(
    cam: np.ndarray, n_components: int = 3, kernel_size: int = 3, **kwargs
) -> np.ndarray:

    norm = MinMaxScaler()
    cam = medfilt2d(cam, kernel_size=kernel_size)
    shape_cam = cam.shape
    cam = cam.flatten().reshape(-1, 1)
    cam = norm.fit_transform(cam)
    argmax = cam.argmax()

    gmm_model = GaussianMixture(
        n_components=n_components,
        max_iter=1000,
        init_params="kmeans",
        random_state=SEED,
    )
    gmm_model.fit(cam)
    com_pred = gmm_model.predict_proba(cam[argmax].reshape(-1, 1))

    max_component = com_pred.argmax()
    pred = gmm_model.predict_proba(cam)[:, max_component]

    cam = np.where(pred > 0.5, 1, 0)
    map_post = cam.reshape(shape_cam)
    if map_post.mean() > 0.5:
        map_post = 1 - map_post
    return map_post


def BMM_post(cam: np.ndarray, kernel_size: int = 3, **kwargs) -> np.ndarray:

    cam = medfilt2d(cam, kernel_size=kernel_size)
    shape_cam = cam.shape
    cam = cam.flatten().reshape(-1, 1)
    cam = np.where(cam > 0, cam, 0)
    th_low = np.median(cam)
    cam_selection = cam[cam > th_low].reshape(-1, 1)

    norm = MinMaxScaler()

    try:
        cam_selection = norm.fit_transform(cam_selection)
        cam = norm.transform(cam)
    except ValueError:
        return np.zeros(shape_cam)

    # pdf at 0 or 1 might be not defined
    cam_selection[cam_selection > (1 - 1e-4)] = 1 - 1e-4
    cam_selection[cam_selection < (1e-4)] = 1e-4

    # pdf at 0 or 1 might be not defined
    cam[cam > (1 - 1e-4)] = 1 - 1e-4
    cam[cam < (1e-4)] = 1e-4

    beta_model = BetaMixtureModel(**kwargs)
    beta_model.fit(cam_selection)
    pred = beta_model.predict(cam)

    map_post = pred.reshape(shape_cam)
    if map_post.mean() > 0.5:
        map_post = 1 - map_post
    return map_post


def otsu_post(cam: np.ndarray, **kwargs):

    norm = MinMaxScaler()
    shape_cam = cam.shape
    cam = cam.flatten().reshape(-1, 1)
    cam = norm.fit_transform(cam)
    cam = cam.reshape(shape_cam)

    cam = (cam*255).astype(np.uint8)
    _, mask = cv2.threshold(cam, 0, 255, cv2.THRESH_OTSU)

    return mask / 255


def area_opening_post(cam: np.ndarray, min_area: int = 250, **kwargs):

    norm = MinMaxScaler()
    shape_cam = cam.shape
    cam = cam.flatten().reshape(-1, 1)
    cam = norm.fit_transform(cam)
    cam = cam.reshape(shape_cam)

    cam = (cam*255).astype(np.uint8)

    # find all of the connected components (white blobs in your image).
    # im_with_separated_blobs is an image where each detected blob has a different pixel value ranging from 1 to nb_blobs - 1.
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(cam)
    # stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information. 
    # here, we're interested only in the size of the blobs, contained in the last column of stats.
    sizes = stats[:, -1]
    # the following lines result in taking out the background which is also considered a component, which I find for most applications to not be the expected output.
    # you may also keep the results as they are by commenting out the following lines. You'll have to update the ranges in the for loop below. 
    sizes = sizes[1:]
    nb_blobs -= 1

    # output image with only the kept components
    im_result = np.zeros_like(im_with_separated_blobs)
    # for every component in the image, keep it only if it's above min_size
    for blob in range(nb_blobs):
        if sizes[blob] >= min_area:
            # see description of im_with_separated_blobs above
            im_result[im_with_separated_blobs == blob + 1] = 1

    return im_result


def post_factory(method: str) -> Callable[[np.ndarray, Any], np.ndarray]:
    if method == "quantile":
        return quantile_post
    elif method == "simple":
        return simple_post
    elif method == "gmm":
        return GMM_post
    elif method == "bmm":
        return BMM_post
    elif method == "closing":
        return closing_post
    elif method == "opening":
        return opening_post
    elif method == "otsu":
        return otsu_post
    elif method == "area_opening":
        return area_opening_post
    elif method == "threshold":
        return threshold_post
    else:
        raise NotImplementedError()
