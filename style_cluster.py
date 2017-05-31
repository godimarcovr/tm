import os
import time
import random
import glob
import h5py
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from PIL import Image as pil_image
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import AffinityPropagation
from scipy.spatial.distance import cosine as cosdist
from sklearn.metrics.pairwise import pairwise_distances

src_path = os.path.join("data", "tbs500_nomatch")
dst_basefold = "tbs_nomatch_500"
feat_path = os.path.join("saves", dst_basefold, "features_vgg.hdf5")
bin_feat_path = os.path.join("saves", dst_basefold, "features_bin_vgg.hdf5")
f = h5py.File(feat_path, "r")
f_bin = h5py.File(bin_feat_path, "r")
jpgs = glob.glob(os.path.join(src_path, "*", "*.jpg"))
ids = [x.split("\\")[-1].split(".")[0] for x in jpgs]
features = f["feats"][:]
#features_bin = f_bin["feats"][:]
f.close()
f_bin.close()

outfits = {}
assoc_list = {}
catnames = {}

# with open(os.path.join(src_path, "info.csv"), "r") as csvinfo:
#     lines = csvinfo.readlines()
#     for line in lines:
#         pieces = line.split(";")
#         cloth_id = pieces[0]
#         raw_list = pieces[5]
#         catname = pieces[6].rstrip()
#         assoc_list[cloth_id] = ids.index(cloth_id)
#         catnames[cloth_id] = catname
#         if len(raw_list) <= 2:
#             continue
#         outfit_list = []
#         for raw_list_element in raw_list[1:-1].split(","):
#             outfit_list.append(raw_list_element)
#         outfits[cloth_id] = outfit_list

# for epsmod in ((np.random.rand(10) * 2.0) - 1.0).tolist():
cat_path = os.path.join("saves", dst_basefold, "classes.hdf5")
f = h5py.File(cat_path, "r")
classes = f["classes"][:]
f.close()
class_names = set(classes.tolist())

train_path = src_path
cat_labels = [f for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))]

class_nums = [i for i in range(len(class_names))]
random.shuffle(class_nums)
features_orig = np.copy(features)
jpgs_orig = jpgs.copy()

feat_len = features.shape[1]
col_len = 11
vgg_len = feat_len - col_len
wvgg = 1.0
wcol = 1.0 - wvgg

for cat in class_nums:
    # eps = 5
    # m = 3
    print(cat_labels[cat])
    # print(eps)
    # print(m)
    inds = [i for i, c in enumerate(classes) if c == cat]
    features = features_orig[inds]
    print(features.shape)
    jpgs = [jpgs_orig[i] for i in inds]

    dist_path_vgg = os.path.join("saves", dst_basefold, cat_labels[cat] + "_dist_vgg.hdf5")
    dist_path_col = os.path.join("saves", dst_basefold, cat_labels[cat] + "_dist_col.hdf5")
    dist_path_sum = os.path.join("saves"
                                 , dst_basefold, cat_labels[cat] + "_dist_sum_" + str(wvgg) + ".hdf5")

    if not os.path.isfile(dist_path_vgg):
        distmat = pairwise_distances(features[:, :vgg_len], metric="cosine")
        f = h5py.File(dist_path_vgg, "w")
        f["dists"] = distmat
        f.close()

    if not os.path.isfile(dist_path_col):
        distmat = pairwise_distances(features[:, vgg_len:], metric="cosine")
        f = h5py.File(dist_path_col, "w")
        f["dists"] = distmat
        f.close()

    if not os.path.isfile(dist_path_sum):
        f_vgg = h5py.File(dist_path_vgg, "r")
        distmat_vgg = f_vgg["dists"][:]
        f_vgg.close()
        f_col = h5py.File(dist_path_col, "r")
        distmat_col = f_col["dists"][:]
        f_col.close()
        distmat = distmat_vgg * wvgg + distmat_col * wcol
        f = h5py.File(dist_path_sum, "w")
        f["dists"] = distmat
        f.close()
        # for count in range(3):
        #     i1 = int(random.random() * distmat_vgg.shape[0])
        #     i2 = int(random.random() * distmat_vgg.shape[0])
        #     fig1 = plt.figure(1)
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(pil_image.open(jpgs[inds[i1]]))
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(pil_image.open(jpgs[inds[i2]]))
        #     plt.title("vgg: " + str(distmat_vgg[i1, i2]) + "//col: " + str(distmat_col[i1, i2]))
        #     plt.show()
    f = h5py.File(dist_path_sum, "r")
    distmat = f["dists"][:]
    f.close()
    def weighted_cos_dist(feats):
        return distmat

    # #mostra elementi di quella classe
    # for idc in inds[:10]:
    #     fig1 = plt.figure(1)
    #     plt.imshow(pil_image.open(jpgs[idc]))
    #     plt.show()
    st = time.time()
    #db = DBSCAN(eps=eps, min_samples=m)
    #db = MiniBatchKMeans(n_clusters=features.shape[0]//20)
    db = AgglomerativeClustering(n_clusters=15, affinity = weighted_cos_dist, linkage = 'average')
    #db = Birch()
    #db = MeanShift(bandwidth=9)
    db = db.fit(features[:, :vgg_len])
    tt = time.time() - st
    print("Time elapsed ", tt)
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("**********************")
    print("n clusters: ", n_clusters_)
    n_outliers = len([x for x in range(features.shape[0]) if labels[x] == -1])
    print("n outliers: ", n_outliers)

    for cl in range(n_clusters_):
        cl_indexes = [x for x in range(features.shape[0]) if labels[x] == cl]
        print("cluster ", cl, ":", len(cl_indexes))
        #random.shuffle(cl_indexes)
        fig1 = plt.figure(1)
        fig1.canvas.set_window_title('cluster ' + str(cl)) 
        for i, ngind in enumerate(cl_indexes[:9]):
            plt.subplot(3, 3, i + 1)
            plt.imshow(pil_image.open(jpgs[ngind]))
        plt.show()
        features2 = features[cl_indexes]
        jpgs2 = [jpgs[i] for i in cl_indexes]
        f_col = h5py.File(dist_path_col, "r")
        distmat = f_col["dists"][cl_indexes]
        distmat = distmat[:, cl_indexes]
        f_col.close()
        #db2 = AgglomerativeClustering(n_clusters=5, affinity = weighted_cos_dist, linkage = 'average')
        #db2 = MeanShift(bandwidth=10)
        #db2 = DBSCAN(eps=0.4, min_samples=3)
        #db2 = AffinityPropagation(damping=0.5)
        #db2 = Birch(n_)
        db2 = MiniBatchKMeans(n_clusters=5)
        db2.fit(features2[:, vgg_len:])
        labels2 = db2.labels_
        n_clusters_2 = len(set(labels2)) - (1 if -1 in labels2 else 0)
        print("n clusters2: ", n_clusters_2)
        for cl2 in range(n_clusters_2):
            cl_indexes2 = [x for x in range(features2.shape[0]) if labels2[x] == cl2]
            print("cluster2 ", cl2, ":", len(cl_indexes2))
            random.shuffle(cl_indexes2)
            fig1 = plt.figure(1)
            fig1.canvas.set_window_title('cluster2 ' + str(cl)) 
            for i2, ngind2 in enumerate(cl_indexes2[:9]):
                plt.subplot(3, 3, i2 + 1)
                plt.imshow(pil_image.open(jpgs2[ngind2]))
            plt.show()


# print("********************")

#jpgs[0].split("\\")[-1].split(".")[0]

# for epsmod in ((np.random.rand(10) * 2.0) - 1.0).tolist():
# for cat in list(set(catnames.values())):
#     if cat == '':
#         continue
#     cat_ids = [k for k, v in catnames.items() if v == cat]
#     cat_ids = [assoc_list[k] for k in cat_ids]
#     # for idc in cat_ids:
#     #     fig1 = plt.figure(1)
#     #     plt.imshow(pil_image.open(jpgs[idc]))
#     #     plt.show()
#     eps = 7
#     m = 3
#     print(eps)
#     print(m)
#     #db = DBSCAN(eps=eps, min_samples=m).fit(features[cat_ids])
#     #db = KMeans(n_clusters=50).fit(features)
#     db = AgglomerativeClustering(10).fit(features[cat_ids])
#     labels = db.labels_
#     # Number of clusters in labels, ignoring noise if present.
#     n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     print("n clusters: ", n_clusters_)
#     n_outliers = len([x for x in range(features[cat_ids].shape[0]) if labels[x] == -1])
#     print("n outliers: ", n_outliers)
#     for cl in range(n_clusters_):
#         cl_indexes = [x for x in range(features[cat_ids].shape[0]) if labels[x] == cl]
#         print("cluster ", cl, ":", len(cl_indexes))
#         random.shuffle(cl_indexes)
#         fig1 = plt.figure(1)
#         fig1.canvas.set_window_title('cluster ' + str(cl)) 
#         for i, ngind in enumerate(cl_indexes[:9]):
#             plt.subplot(3, 3, i + 1)
#             try:
#                 plt.imshow(pil_image.open([jpgs[i] for i in cat_ids][ngind]))
#             except Exception:
#                 continue
#         plt.show()
#     # print("********************")

#     #jpgs[0].split("\\")[-1].split(".")[0]
#     print("**********************")