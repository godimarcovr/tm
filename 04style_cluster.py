import os
import time
import random
import glob
import h5py
import tkinter as tk
import pickle
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
from sklearn.cluster import SpectralClustering
from sklearn.cluster import spectral_clustering

dst_basefold = "tbs500_nomatch"
src_path = os.path.join("data", dst_basefold)
feat_path = os.path.join("saves", dst_basefold, "features_vgg.hdf5")
col_feat_path = os.path.join("saves", dst_basefold, "features_color.hdf5")
dist_vgg_path = os.path.join("saves", dst_basefold, "dist_vgg.hdf5")
dsinfo_path = os.path.join("saves", dst_basefold, "dsinfo.pickle")
clustering_info_path = os.path.join("saves", dst_basefold, "clustering_info.hdf5")

f = h5py.File(feat_path, "r")
f_col = h5py.File(col_feat_path, "r")
jpgs = glob.glob(os.path.join(src_path, "*", "*.jpg"))
ids = [x.split("\\")[-1].split(".")[0] for x in jpgs]
features = f["feats"][:]
features_col = f_col["feats"][:]
f.close()
f_col.close()

cat_path = os.path.join("saves", dst_basefold, "classes.hdf5")
f = h5py.File(cat_path, "r")
classes = f["classes"][:]
f.close()
class_names = set(classes.tolist())

with open(dsinfo_path, 'rb') as handle:
    dsinfo, assoc = pickle.load(handle)

cat_labels = [f for f in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, f))]

class_nums = [i for i in range(len(class_names))]
random.shuffle(class_nums)
features_orig = np.copy(features)
jpgs_orig = jpgs.copy()


gccount = 0
cluster_dict = []
gclabels = np.ones(classes.shape[0], dtype=np.int32) * (-1)

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

    f = h5py.File(dist_vgg_path, "r")
    distmat = f["dists"][:]
    f.close()

    def weighted_cos_dist(feats):
        return distmat
    st = time.time()
    ncluster_by_cat = [15, 20, 20, 25, 25, 20, 15, 15, 20, 20, 15, 20, 20, 25]
    #db = DBSCAN(eps=eps, min_samples=m)
    #db = MiniBatchKMeans(n_clusters=features.shape[0]//20)
    #db = AgglomerativeClustering(n_clusters=15, affinity = weighted_cos_dist, linkage = 'average')
    #db = Birch()
    #db = MeanShift(bandwidth=9)
    db = SpectralClustering(n_clusters=int(ncluster_by_cat[cat] * 0.75), affinity='precomputed')
    db = db.fit(1.0 - distmat[inds][:, inds])
    
    #MENO DISTMAT???? CI VA AFFINITY
    #labels = spectral_clustering(distmat, n_clusters=ncluster_by_cat[cat])
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
        inds_cl = [inds[x] for x in cl_indexes]
        print("cluster ", cl, ":", len(cl_indexes))
        # random.shuffle(cl_indexes)
        # fig1 = plt.figure(1)
        # fig1.canvas.set_window_title('cluster ' + str(cl)) 
        # for i, ngind in enumerate(cl_indexes[:9]):
        #     plt.subplot(3, 3, i + 1)
        #     plt.imshow(pil_image.open(jpgs[ngind]))
        # plt.show()

        features2 = features[cl_indexes]
        jpgs2 = [jpgs[i] for i in cl_indexes]
        features_col_ncl = features_col[inds_cl]
        #db2 = AgglomerativeClustering(n_clusters=min([]), affinity = 'euclidean', linkage = 'average')
        db2 = MeanShift(bandwidth=0.4)
        #db2 = DBSCAN(eps=0.4, min_samples=3)
        #db2 = AffinityPropagation(damping=0.5)
        #db2 = Birch(n_)
        #db2 = MiniBatchKMeans(n_clusters=5)
        db2.fit(features_col_ncl)
        labels2 = db2.labels_
        n_clusters_2 = len(set(labels2)) - (1 if -1 in labels2 else 0)
        print("n clusters2: ", n_clusters_2)
        for cl2 in range(n_clusters_2):
            cl_indexes2 = [x for x in range(features2.shape[0]) if labels2[x] == cl2]
            inds_cl2 = [inds_cl[x] for x in cl_indexes2]
            print("cluster2 ", cl2, ":", len(cl_indexes2))
            # random.shuffle(cl_indexes2)
            # fig1 = plt.figure(1)
            # fig1.canvas.set_window_title('cluster2 ' + str(cl)) 
            # for i2, ngind2 in enumerate(cl_indexes2[:min([9, len(cl_indexes2)])]):
            #     plt.subplot(3, 3, i2 + 1)
            #     plt.imshow(pil_image.open(jpgs2[ngind2]))
            # plt.show()
            cluster_dict.append([cat, cl, cl2])
            gclabels[inds_cl2] = gccount
            gccount += 1

f = h5py.File(clustering_info_path, "w")
f["gclabels"] = gclabels
f["cluster_dict"] = np.asarray(cluster_dict)
f.close()