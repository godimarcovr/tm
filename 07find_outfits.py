import os
import glob
import h5py
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image as pil_image
import random

dst_basefold = "tbs400"
src_path = os.path.join("data", dst_basefold)
feat_path = os.path.join("saves", dst_basefold, "features_vgg.hdf5")
col_feat_path = os.path.join("saves", dst_basefold, "features_color.hdf5")
dist_vgg_path = os.path.join("saves", dst_basefold, "dist_vgg.hdf5")
dsinfo_path = os.path.join("saves", dst_basefold, "dsinfo.pickle")
clustering_info_path = os.path.join("saves", dst_basefold, "clustering_info.hdf5")
cat_path = os.path.join("saves", dst_basefold, "classes.hdf5")
stylematrix_path = os.path.join("saves", dst_basefold, "stylematrix.hdf5")

dst_basefold_test = "tbs400_test"
src_path_test = os.path.join("data", dst_basefold_test)
feat_path_test = os.path.join("saves", dst_basefold_test, "features_vgg.hdf5")
col_feat_path_test = os.path.join("saves", dst_basefold_test, "features_color.hdf5")
dist_vgg_path_test = os.path.join("saves", dst_basefold_test, "dist_vgg.hdf5")
dsinfo_path_test = os.path.join("saves", dst_basefold_test, "dsinfo.pickle")
cat_path_test = os.path.join("saves", dst_basefold_test, "classes.hdf5")
clustering_info_path_test = os.path.join("saves", dst_basefold_test, "clustering_info.hdf5")


#carico stylematrix
f = h5py.File(stylematrix_path, "r")
style_matrix = f["stylematrix"][:]
f.close()
#carico dsinfo
with open(dsinfo_path_test, 'rb') as handle:
    dsinfo, assoc = pickle.load(handle)
#carico immagini
jpgs = glob.glob(os.path.join(src_path, "*", "*.jpg"))

#carico dsinfo test
with open(dsinfo_path_test, 'rb') as handle:
    dsinfo_test, assoc_test = pickle.load(handle)
#carico cluster info test
f = h5py.File(clustering_info_path_test, "r")
gclabels_test = f["gclabels"][:]
cluster_dict = f["cluster_dict"][:]
f.close()

#carico immagini test
jpgs_test = glob.glob(os.path.join(src_path_test, "*", "*.jpg"))

inds = [x for x in range(len(jpgs_test))]
random.shuffle(inds)
for ind in inds:
    gcl_test = gclabels_test[ind]
    smrow = style_matrix[gcl_test, :]
    pairing_rank = np.argsort(smrow)[::-1].tolist()
    pairing_rank = [x for x in pairing_rank if smrow[x] > 0]
    if len(pairing_rank) >= 3:
        fig1 = plt.figure(1)
        plt.subplot(3, 3, 1)
        plt.imshow(pil_image.open(jpgs_test[ind]))
        c1 = -1
        i = 4
        for c in range(3):
            candidates = []
            while len(candidates) == 0:
                c1 += 1
                candidates = [i for i, x in enumerate(gclabels_test) if x == pairing_rank[c1]]
            #fig1.canvas.set_window_title('cluster ' + str(cl))
            for _, ngind in enumerate(candidates[:min(2, len(candidates))]):
                plt.subplot(3, 3, i)
                i += 1
                plt.imshow(pil_image.open(jpgs_test[ngind]))
        plt.show()

















