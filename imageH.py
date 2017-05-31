import numpy as np
import os
import cv2
import collections
import pandas as pd

from PIL import Image
from scipy.cluster.vq import kmeans2
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import euclidean
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1976 as dist_lab
from matplotlib import colors as mcolors
from munkres import Munkres

np.random.seed(2902)

color_hist_names = ['black', 'blue', 'brown', 'grey', 'green', 'orange',
                    'pink', 'purple', 'red', 'white', 'yellow']
color_hist_values = np.array([[0, 0, 0], [0, 0, 1], [0.5, 0.4, 0.25], [0.5, 0.5, 0.5], [0, 1, 0], [1, 0.8, 0],
                              [1, 0.5, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0]]) * 255

target_size = [500, 500]

# 0:black 1:blue 2:brown 3:grey 4:green 5:orange 6:pink 7:purple 8:red 9:white 10:yellow
CSS4_to_color_names = [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 9, 9, 9, 6, 6, 6, 2, 8, 2,
                       8, 8, 6, 6, 6, 6, 6, 5, 6, 2, 9, 2, 2, 2, 6, 2, 9, 6, 5, 2,
                       9, 2, 6, 6, 6, 6, 5, 2, 9, 9, 2, 10, 9, 10, 10, 10, 10, 4, 9, 9,
                       10, 10, 4, 10, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                       4, 4, 4, 4, 1, 1, 1, 4, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 9, 1, 3, 3, 3, 3, 1, 1, 1, 9, 7, 7, 1, 1,
                       1, 1, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 7, 7, 6, 6, 6,
                       6, 6, 6, 6, 6, 8, 6, 6]

XKCD_to_color_names = [10, 10, 9, 10, 7, 1, 8, 1, 2, 3, 2, 6, 8, 7, 7, 6, 6, 5, 5, 3,
                       1, 2, 6, 4, 4, 1, 7, 10, 10, 7, 3, 4, 10, 10, 7, 4, 1, 1, 8, 6,
                       2, 4, 2, 1, 0, 10, 1, 1, 1]


class ImageH(object):
    """An image.

    Attributes:
        img_rgb: An RGB PIL image.
        img_grey: A grey PIL Image.
        size: Image pixel size w x h
        compute_mask: Tru or False if consider the mask in the image processing
        mask: Binary mask of size w x h if compute mask is true
        bg_color: 'white' or 'black' if compute_mask is True
        color_hist_index: index of the 11 bin colors histogram with the colors sorted by relevance
        color_hist_scores: scores of the 11 bin colors histogram with the colors sorted by relevance
        color_hist_names: names of the 11 bin colors histogram with the colors sorted by relevance
        color_hist_real_rgb: real rgb values of the 11 bin colors histogram with the colors sorted by relevance
        color_hist_rgb: rgb values of the 11 bin colors histogram with the colors sorted by relevance
        color_clusters_center: rgb values of color clusters centers
        color_clusters_weights: percentage of each color cluster in the image
        color_clusters_names: names of each color cluster in the image
        color_clusters_index: index of each color cluster in the image
    """

    def __init__(self, path, compute_mask=False, compute_hist=False, compute_clusters=False, num_cluster_color=5,
                 type_cluster_color='XKCD', type_cluster_distance='LAB'):
        """
        Return an ImageH object.
        :param path: A string path to the image file
        :param compute_mask: True or False value if you want or not to extract the mask (def=False)
        :param compute_hist: True or False value if you want or not to extract the 11 bin color hist (def=False)
        :param compute_clusters: True or False value if you want or not to extract clusters from RGB pixels (def=False)
        :param num_cluster_color: Number of color clusters (def=5)
        :param type_cluster_color: Color map to use; 'BASE', 'CSS4', 'XKCD', 'IM2COLOR', 'ALL'
        :param type_cluster_distance: distance to be used 'EUCRGB' for euclidean distance between rgb values or 'LAB'
        for delta_e_cie1976 distance between lab values or 'EUCLAB' for euclidean distance between lab values or 'XKCD'
        for xkcd colors
        """
        assert os.path.isfile(path), 'The path is not a file'
        self.img_rgb = Image.open(path)
        self.img_grey = self.img_rgb.convert('L')
        self.size = self.img_rgb.size
        self.compute_mask = compute_mask

        self.mask = None
        self.bg_color = None
        if compute_mask:
            self.mask = get_mask(self)

        self.color_hist_index = None
        self.color_hist_scores = None
        self.color_hist_names = None
        self.color_hist_rgb = None
        self.color_hist_real_rgb = None
        if compute_hist:
            self.color_hist_index, self.color_hist_scores, self.color_hist_names, \
            self.color_hist_real_rgb, self.color_hist_rgb = get_color_hist(self)

        self.color_clusters_center = None
        self.color_clusters_weights = None
        self.color_clusters_names = None
        self.num_cluster_color = num_cluster_color
        self.type_cluster_color = type_cluster_color
        self.type_cluster_distance = type_cluster_distance
        if compute_clusters:
            self.color_clusters_center, self.color_clusters_weights, self.color_clusters_names, \
            self.color_clusters_index = pixel_clustering(self, typec=self.type_cluster_color,
                                                         typed=self.type_cluster_distance)

    def set_mask(self, mask):
        self.mask = mask

    def set_num_cluster_color(self, num_cluster_color):
        self.num_cluster_color = num_cluster_color

    def set_type_cluster_color(self, type_cluster_color):
        self.type_cluster_color = type_cluster_color


def get_color_hist_names():
    """
    Get the colors names of the color hist
    :return: ['black', 'blue', 'brown', 'grey', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']
    """
    return color_hist_names


def get_color_hist_values():
    """
    Get the colors RGB values of the color hist
    :return: [[0, 0, 0], [0, 0, 1], [0.5, 0.4, 0.25], [0.5, 0.5, 0.5], [0, 1, 0], [1, 0.8, 0],
                         [1, 0.5, 1], [1, 0, 1], [1, 0, 0], [1, 1, 1], [1, 1, 0]]
    """
    return color_hist_values


def get_mask(Ih):
    """
    Return the mask of an ImageH with black or white background
    :param Ih: ImageH image
    :return: mask image as ndarray n x m
    """
    # Add two pixel padding
    new_size = (Ih.size[0] + 4, Ih.size[1] + 4)
    new_im = Image.new("L", new_size, "white")
    new_im.paste(Ih.img_grey, (int((new_size[0] - Ih.size[0]) / 2), int((new_size[1] - Ih.size[1]) / 2)))

    # Get grey image
    im_in = np.array(new_im)
    # check if background is black or white
    Ih.bg_color = 'white'
    if im_in[1, 1] < 250:
        Ih.bg_color = 'black'
        im_in = 255 - im_in

    # Threshold.
    # Set values equal to or above 250 to 0.
    # Set values below 250 to 255.
    th, im_th = cv2.threshold(im_in, 240, 255, cv2.THRESH_BINARY_INV)
    # Copy the threshed image.
    im_floodfill = im_th.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    # Invert floodfill image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    # Set the mask in the ImageH object
    im_out = im_out[2:-2, 2:-2]
    Ih.set_mask(im_out)

    return im_out


def pixel_clustering(Ih, typec='XKCD', typed='LAB'):
    """
    Pixel clustering of the image Ih
    :param Ih: ImageH image
    :param typec: color map to use; 'BASE', 'CSS4', 'XKCD', 'IM2COLOR', 'ALL'
    :param typed: distance to be used 'EUCRGB' for euclidean distance between rgb values or 'LAB' for delta_e_cie1976
    distance between lab values or 'EUCLAB' for euclidean distance between lab values or 'XKCD' for xkcd colors
    :return:
    1) RGB values of the the clusters' centers
    1) The percentage of each color (cluster) in the image Ih. If the mask is enabled, the percentage does not consider
    the background
    2) The name of each color (cluster) considering the typec color map
    3) The color index of each color (cluster) in the corresponding color map
    """
    # If there is a background color with the mask do not consider this cluster
    num_cluster_color = Ih.num_cluster_color
    if Ih.bg_color is not None:
        num_cluster_color += 1

    img = Ih.img_rgb.resize((target_size[0], target_size[1]), Image.ANTIALIAS)
    img2 = np.array(img)
    img3 = np.reshape(img2, (img2.shape[0] * img2.shape[1], img2.shape[2]))
    try:
        color_clusters = kmeans2(np.float32(img3), num_cluster_color, minit='points')
        cc_counting = np.bincount(color_clusters[1])
        ii = np.nonzero(cc_counting + 1)[0]
        cc_most_important = [i[0] for i in sorted(enumerate(cc_counting), key=lambda x: x[1])]
        cc_most_important = cc_most_important[::-1]

        cc_test = np.array(color_clusters[0])
        cc_test2 = cc_test[ii[cc_most_important]]
        weights = np.array(cc_counting[cc_most_important])
    except:
        color_clusters = kmeans(np.float32(img3), num_cluster_color)
        cc_test2 = color_clusters[0]
        weights = np.ones(color_clusters[0].shape[0])/num_cluster_color

    color_clusters_center = np.array(cc_test2)

    color_clusters_names = []
    color_clusters_ind = []
    ind_to_remove = []
    for cci, ccc in enumerate(color_clusters_center):
        cn, cval, cind = get_color_name(ccc, typec=typec, typed=typed)
        color_clusters_ind.append(cind)
        color_clusters_names.append(cn)
        if Ih.bg_color is not None and cn == Ih.bg_color:
            ind_to_remove.append(cci)

    if len(ind_to_remove) > 0:
        color_clusters_names = np.delete(color_clusters_names, ind_to_remove[0])
        color_clusters_center = np.delete(color_clusters_center, ind_to_remove[0], 0)
        color_clusters_ind = np.delete(color_clusters_ind, ind_to_remove[0], 0)
        weights = np.delete(weights, ind_to_remove[0])

    weights = (weights + 1) / np.sum(weights + 1)

    return color_clusters_center, weights, color_clusters_names, color_clusters_ind


def get_color_name(rgb_values, typec='XKCD', typed='LAB'):
    """
    Get the color name of an RGB value
    :param rgb_values: ndarray with 3 numbers RGB from 0 to 255
    :param typec: color map to use; 'BASE', 'CSS4', 'XKCD', 'IM2COLOR', 'ALL'
    :param typed: distance to be used 'EUCRGB' for euclidean distance between rgb values or 'LAB' for delta_e_cie1976
    distance between lab values or 'EUCLAB' for euclidean distance between lab values or 'XKCD' for xkcd colors
    :return:
    1) The color name string of the rgb_values in the corresponding color map
    2) The color RGB values of the rgb_values in the corresponding color map
    3) The color index of the rgb_values in the corresponding color map
    """

    rgb_values = np.array(rgb_values / 256 * 255)
    if typec == 'BASE':
        color_name, color_value, color_ind = get_color_BASE(rgb_values, typed)
    elif typec == 'CSS4':
        color_name, color_value, color_ind = get_color_CSS4(rgb_values, typed)
    elif typec == 'XKCD':
        color_name, color_value, color_ind = get_color_XKCD(rgb_values, typed)
    elif typec == 'IM2COLOR':
        color_name, color_value, color_ind = get_color_IM2COLOR(rgb_values, typed)
    else:
        color_name1, color_value1, color_ind1 = get_color_BASE(rgb_values, typed)
        color_name2, color_value2, color_ind2 = get_color_CSS4(rgb_values, typed)
        color_name3, color_value3, color_ind3 = get_color_XKCD(rgb_values, typed)
        color_name4, color_value4, color_ind4 = get_color_IM2COLOR(rgb_values, typed)
        color_name = [color_name1, color_name2, color_name3, color_name4]
        color_value = [color_value1, color_value2, color_value3, color_value4]
        color_ind = [color_ind1, color_ind2, color_ind3, color_ind4]

    return color_name, color_value, color_ind


def get_color_XKCD(rgb_values, typed='LAB'):
    """
    Get the color info of an RGB value in the corresponding XKCD colors
    :param rgb_values: ndarray with 3 numbers RGB from 0 to 255
    :param typed: distance to be used, 'EUCRGB' for euclidean distance between rgb values or 'LAB' for delta_e_cie1976
    distance between lab values or 'EUCLAB' for euclidean distance between lab values
    :return:
    1) The color name string of the rgb_values in the corresponding XKCD colors
    2) The color RGB values of the rgb_values in the corresponding XKCD colors
    3) The color index of the rgb_values in the corresponding XKCD colors
    """

    overlap = {name for name in mcolors.CSS4_COLORS
               if "xkcd:" + name in mcolors.XKCD_COLORS}

    rgb_names = sorted(overlap, reverse=True)
    rgb_vals = []
    for j, n in enumerate(rgb_names):
        rgb_vals.append(mcolors.to_rgb(mcolors.XKCD_COLORS["xkcd:" + n].upper()))
    #
    if typed == 'LAB' or typed == 'EUCLAB':
        rgb1 = sRGBColor(rgb_values[0], rgb_values[1], rgb_values[2], is_upscaled=True)
        lab1 = convert_color(rgb1, LabColor)
    dist = []
    for ci, c in enumerate(rgb_vals):
        cc = np.array(c) * 255
        if typed == 'LAB' or typed == 'EUCLAB':
            rgb2 = sRGBColor(cc[0], cc[1], cc[2], is_upscaled=True)
            lab2 = convert_color(rgb2, LabColor)
            if typed == 'EUCLAB':
                d = euclidean(np.asarray(lab1.get_value_tuple()), np.asarray(lab2.get_value_tuple()))
            else:
                d = dist_lab(lab1, lab2)
        elif typed == 'EUCRGB':
            d = euclidean(rgb_values, cc)
        dist.append(d)

    most_important = [i[0] for i in sorted(enumerate(dist), key=lambda x: x[1])]
    color_name = rgb_names[most_important[0]]

    return color_name, np.array(rgb_vals[most_important[0]]) * 255, most_important[0]


def get_color_CSS4(rgb_values, typed='LAB'):
    """
    Get the color info of an RGB value in the corresponding CSS4_COLORS
    :param rgb_values: ndarray with 3 numbers RGB from 0 to 255
    :param typed: distance to be used, 'EUCRGB' for euclidean distance between rgb values or 'LAB' for delta_e_cie1976
    distance between lab values or 'EUCLAB' for euclidean distance between lab values
    :return:
    1) The color name string of the rgb_values in the corresponding CSS4_COLORS
    2) The color RGB values of the rgb_values in the corresponding CSS4_COLORS
    3) The color index of the rgb_values in the corresponding CSS4_COLORS
    """

    colors = dict(mcolors.CSS4_COLORS)
    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color)[:3])), name)
                    for name, color in colors.items())
    rgb_vals = [np.array(mcolors.hsv_to_rgb(color[0])) for color in by_hsv]
    rgb_names = [color[1] for color in by_hsv]
    #
    if typed == 'LAB' or typed == 'EUCLAB':
        rgb1 = sRGBColor(rgb_values[0], rgb_values[1], rgb_values[2], is_upscaled=True)
        lab1 = convert_color(rgb1, LabColor)
    dist = []
    for ci, c in enumerate(rgb_vals):
        cc = np.array(c) * 255
        if typed == 'LAB' or typed == 'EUCLAB':
            rgb2 = sRGBColor(cc[0], cc[1], cc[2], is_upscaled=True)
            lab2 = convert_color(rgb2, LabColor)
            if typed == 'EUCLAB':
                d = euclidean(np.asarray(lab1.get_value_tuple()), np.asarray(lab2.get_value_tuple()))
            else:
                d = dist_lab(lab1, lab2)
        elif typed == 'EUCRGB':
            d = euclidean(rgb_values, cc)
        dist.append(d)

    most_important = [i[0] for i in sorted(enumerate(dist), key=lambda x: x[1])]
    color_name = rgb_names[most_important[0]]

    return color_name, np.array(rgb_vals[most_important[0]]) * 255, most_important[0]


def get_color_BASE(rgb_values, typed='LAB'):
    """
    Get the color info of an RGB value in the corresponding BASE_COLORS
    :param rgb_values: ndarray with 3 numbers RGB from 0 to 255
    :param typed: distance to be used, 'EUCRGB' for euclidean distance between rgb values or 'LAB' for delta_e_cie1976
    distance between lab values or 'EUCLAB' for euclidean distance between lab values
    :return:
    1) The color name string of the rgb_values in the corresponding BASE_COLORS
    2) The color RGB values of the rgb_values in the corresponding BASE_COLORS
    3) The color index of the rgb_values in the corresponding BASE_COLORS
    """

    colors = dict(mcolors.BASE_COLORS)
    by_rgb = [np.array(mcolors.to_rgb(color)[:3]) for name, color in colors.items()]
    by_rgb_names = [name for name, color in colors.items()]

    for ri, rn in enumerate(by_rgb_names):
        if rn == 'w':
            by_rgb_names[ri] = 'white'
        elif rn == 'g':
            by_rgb_names[ri] = 'green'
        elif rn == 'b':
            by_rgb_names[ri] = 'blue'
        elif rn == 'k':
            by_rgb_names[ri] = 'black'
        elif rn == 'r':
            by_rgb_names[ri] = 'red'
        elif rn == 'y':
            by_rgb_names[ri] = 'yellow'
        elif rn == 'c':
            by_rgb_names[ri] = 'cyan'
        elif rn == 'm':
            by_rgb_names[ri] = 'magenta'

    if typed == 'LAB' or typed == 'EUCLAB':
        rgb1 = sRGBColor(rgb_values[0], rgb_values[1], rgb_values[2], is_upscaled=True)
        lab1 = convert_color(rgb1, LabColor)

    dist = []
    for ci, c in enumerate(by_rgb):
        c = np.array(c) * 255
        if typed == 'LAB' or typed == 'EUCLAB':
            rgb2 = sRGBColor(c[0], c[1], c[2], is_upscaled=True)
            lab2 = convert_color(rgb2, LabColor)
            if typed == 'EUCLAB':
                d = euclidean(np.asarray(lab1.get_value_tuple()), np.asarray(lab2.get_value_tuple()))
            else:
                d = dist_lab(lab1, lab2)
        elif typed == 'EUCRGB':
            d = euclidean(rgb_values, c)
        dist.append(d)

    most_important = [i[0] for i in sorted(enumerate(dist), key=lambda x: x[1])]
    color_name = by_rgb_names[most_important[0]]

    return color_name, np.array(by_rgb[most_important[0]]) * 255, most_important[0]


def get_color_IM2COLOR(rgb_values, typed='LAB'):
    """
    Get the color info of an RGB value in the corresponding IM2COLOR
    :param rgb_values: ndarray with 3 numbers RGB from 0 to 255
    :param typed: distance to be used 'EUCRGB' for euclidean distance between rgb values or 'LAB' for delta_e_cie1976
    distance between lab values or 'EUCLAB' for euclidean distance between lab values
    :return:
    1) The color name string of the rgb_values in the corresponding IM2COLOR
    2) The color RGB values of the rgb_values in the corresponding IM2COLOR
    3) The color index of the rgb_values in the corresponding IM2COLOR
    """

    original_color_hist_names = get_color_hist_names()
    colors = dict(zip(get_color_hist_names(), get_color_hist_values()))
    by_rgb = [np.array(color / 255) for name, color in colors.items()]
    by_rgb_names = [name for name, color in colors.items()]

    if typed == 'LAB' or typed == 'EUCLAB':
        rgb1 = sRGBColor(rgb_values[0], rgb_values[1], rgb_values[2], is_upscaled=True)
        lab1 = convert_color(rgb1, LabColor)
    dist = []
    for ci, c in enumerate(by_rgb):
        c = np.array(c) * 255
        if typed == 'LAB' or typed == 'EUCLAB':
            rgb2 = sRGBColor(c[0], c[1], c[2], is_upscaled=True)
            lab2 = convert_color(rgb2, LabColor)
            if typed == 'EUCLAB':
                d = euclidean(np.asarray(lab1.get_value_tuple()), np.asarray(lab2.get_value_tuple()))
            else:
                d = dist_lab(lab1, lab2)
        elif typed == 'EUCRGB':
            d = euclidean(rgb_values, c)
        dist.append(d)

    most_important = [i[0] for i in sorted(enumerate(dist), key=lambda x: x[1])]
    color_name = by_rgb_names[most_important[0]]
    orig_ind = original_color_hist_names.index(color_name)

    return color_name, np.array(by_rgb[most_important[0]]) * 255, orig_ind


def get_color_hist(Ih):
    """
    Same as color naming function as described in 'Learning color names for real-world applications. TIP 2009'
    :param Ih:  ImageH object image
    :return: 11 bin hist with the percentage of each color in the image Ih

    """
    if Ih.compute_mask:
        mask = Ih.mask
        if mask is None:
            mask = get_mask(Ih)
    else:
        mask = np.ones(Ih.size)  # mask of ones

    img = np.array(Ih.img_rgb)
    if np.sum(mask) < 1:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    RR = (img[:, :, 0]).astype('float').flatten()
    GG = (img[:, :, 1]).astype('float').flatten()
    BB = (img[:, :, 2]).astype('float').flatten()

    index_im = np.floor(RR / 8) + 32 * np.floor(GG / 8) + 32 * 32 * np.floor(BB / 8)

    w2c = pd.read_hdf('w2c.h5', 'w2c').as_matrix()
    w2cM = np.argmax(w2c, axis=1)

    out = w2cM[index_im.flatten().astype('int32')]
    mask = mask.flatten().astype('bool')
    CNI = out[mask]
    RR = RR[mask]
    GG = GG[mask]
    BB = BB[mask]
    H = collections.Counter(CNI)
    normH = []
    for i in range(11):
        normH.append(H[i] / len(CNI))

    color_hist = np.array(normH)

    ranked_color_hist = [i[0] for i in sorted(enumerate(color_hist), key=lambda x: x[1])]
    index_ranked_color_hist = np.array(ranked_color_hist[::-1])

    scores_ranked_color_hist = np.array(color_hist)[index_ranked_color_hist]
    names_ranked_color_hist = np.array(color_hist_names)[index_ranked_color_hist]
    rgb_ranked_color_hist = np.array(color_hist_values)[index_ranked_color_hist]

    real_rgb = []
    def_rgb_val = get_color_hist_values()
    for i in index_ranked_color_hist:
        iii = CNI == i
        if np.isnan(np.median(RR[iii])) or np.isnan(np.median(RR[iii])) or np.isnan(np.median(RR[iii])):
            Rmean = def_rgb_val[i][0]
            Gmean = def_rgb_val[i][1]
            Bmean = def_rgb_val[i][2]
        else:
            Rmean = int(np.median(RR[iii]))
            Gmean = int(np.median(GG[iii]))
            Bmean = int(np.median(BB[iii]))
        real_rgb.append([Rmean, Gmean, Bmean])


    return index_ranked_color_hist, scores_ranked_color_hist, names_ranked_color_hist, real_rgb, rgb_ranked_color_hist


# def image_color_distance(Ih1, Ih2):
#     ccc1 = Ih1.color_clusters_center
#     ccw1 = Ih1.color_clusters_weights
#     ccn1 = Ih1.color_clusters_names
#
#     ccc2 = Ih2.color_clusters_center
#     ccw2 = Ih2.color_clusters_weights
#     ccn2 = Ih2.color_clusters_names
#
#     ccct = np.vstack((ccc1, ccc2))
#     ccwt = np.hstack((ccw1, ccw2))
#     ccnt = np.asarray(list(ccn1) + list(ccn2))
#
#     dist = np.zeros((ccct.shape[0], ccct.shape[0]))
#     for ci1, cc1 in enumerate(ccct):
#         for ci2, cc2 in enumerate(ccct):
#             rgb1 = sRGBColor(cc1[0], cc1[1], cc1[2], is_upscaled=True)
#             lab1 = convert_color(rgb1, LabColor)
#             rgb2 = sRGBColor(cc2[0], cc2[1], cc2[2], is_upscaled=True)
#             lab2 = convert_color(rgb2, LabColor)
#             dist[ci1, ci2] = dist_lab(lab1, lab2) * np.max(
#                 (ccwt[ci1] / (ccwt[ci2] + 1), ccwt[ci2] / (ccwt[ci1] + 1))) * (
#                                  ccwt[ci1] / (Ih1.size[0] * Ih1.size[1])) * (ccwt[ci2] / (Ih2.size[0] * Ih2.size[1]))
#
#     if dist.shape[0] < Ih1.num_cluster_color * 2:
#         np.hstak(dist, np.ones(Ih1.num_cluster_color, 1) * np.max(dist) * 2)
#
#     d3 = dist.copy()
#     d3 = d3[Ih1.num_cluster_color:, 0:Ih1.num_cluster_color]
#     m = Munkres()
#     indexes = m.compute(np.array(d3))
#
#     res_dist = 0
#     for ix, yx in indexes:
#         res_dist += d3[ix, yx]
#     res_dist = res_dist / Ih1.num_cluster_color
#
#     return res_dist
#
#
# def image_color_distance2(Ih1, Ih2):
#     top_colors = 3
#     ccc1 = Ih1.color_clusters_center[0:top_colors]
#     ccw1 = Ih1.color_clusters_weights[0:top_colors]
#     ccn1 = Ih1.color_clusters_names[0:top_colors]
#
#     ccc2 = Ih2.color_clusters_center[0:top_colors]
#     ccw2 = Ih2.color_clusters_weights[0:top_colors]
#     ccn2 = Ih2.color_clusters_names[0:top_colors]
#
#     dist = np.zeros((ccc1.shape[0]))
#     for ci1, cc1 in enumerate(ccn1):
#         for ci2, cc2 in enumerate(ccn2):
#             if True:  # cc1 == cc2:
#                 rgb1 = sRGBColor(ccc1[ci1][0], ccc1[ci1][1], ccc1[ci1][2], is_upscaled=True)
#                 lab1 = convert_color(rgb1, LabColor)
#                 rgb2 = sRGBColor(ccc2[ci2][0], ccc2[ci2][1], ccc2[ci2][2], is_upscaled=True)
#                 lab2 = convert_color(rgb2, LabColor)
#                 dist[ci1] += dist_lab(lab1, lab2) * (ccw1[ci1] + ccw2[ci2]) / 2
#
#     res_dist = np.sum(dist)
#
#     return res_dist
#
#
# def image_color_distance3(Ih1, Ih2):
#     ccc1 = Ih1.color_clusters_center
#     ccw1 = Ih1.color_clusters_weights
#     ccn1 = Ih1.color_clusters_names
#
#     ccc2 = Ih2.color_clusters_center
#     ccw2 = Ih2.color_clusters_weights
#     ccn2 = Ih2.color_clusters_names
#
#     dist = np.zeros((ccc1.shape[0]))
#     for ci1, cc1 in enumerate(ccn1):
#         cnt = 0
#         for ci2, cc2 in enumerate(ccn2):
#             if cc1 == cc2:
#                 cnt += 1
#                 rgb1 = sRGBColor(ccc1[ci1][0], ccc1[ci1][1], ccc1[ci1][2], is_upscaled=True)
#                 lab1 = convert_color(rgb1, LabColor)
#                 rgb2 = sRGBColor(ccc2[ci2][0], ccc2[ci2][1], ccc2[ci2][2], is_upscaled=True)
#                 lab2 = convert_color(rgb2, LabColor)
#                 dist[ci1] += dist_lab(lab1, lab2) * (np.abs(ccw1[ci1] - ccw2[ci2]) / ccw1[ci1])
#         if cnt > 0:
#             dist[ci1] = dist[ci1] / cnt
#
#     res_dist = np.sum(dist) / dist.shape[0]
#
#     return res_dist
