import os
import glob
import h5py
import numpy as np
import pickle


dst_basefold = "tbs400"
src_path = os.path.join("data", dst_basefold)
feat_path = os.path.join("saves", dst_basefold, "features_vgg.hdf5")
col_feat_path = os.path.join("saves", dst_basefold, "features_color.hdf5")
dist_vgg_path = os.path.join("saves", dst_basefold, "dist_vgg.hdf5")
dsinfo_path = os.path.join("saves", dst_basefold, "dsinfo.pickle")
clustering_info_path = os.path.join("saves", dst_basefold, "clustering_info.hdf5")
cat_path = os.path.join("saves", dst_basefold, "classes.hdf5")

#carico cluster
f = h5py.File(clustering_info_path, "r")
gclabels = f["gclabels"][:]
cluster_dict = f["cluster_dict"][:]
f.close()
f = h5py.File(cat_path, "r")
classes = f["classes"][:]
f.close()
class_names = set(classes.tolist())

#carico dsinfo
with open(dsinfo_path, 'rb') as handle:
    dsinfo, assoc = pickle.load(handle)

style_matrix = np.zeros((cluster_dict.shape[0], cluster_dict.shape[0]), dtype=np.int32)

for ind in range(len(assoc)):
    zid = assoc[ind]
    cinfo = dsinfo[zid]
    gcl = gclabels[ind]
    for pairid in cinfo["outfit"]:
        if pairid in dsinfo:
            pairinfo = dsinfo[pairid]
            gcl2 = gclabels[pairinfo["index"]]
            style_matrix[gcl, gcl2] += 1
            style_matrix[gcl2, gcl] += 1

stylematrix_path = os.path.join("saves", dst_basefold, "stylematrix.hdf5")
f = h5py.File(stylematrix_path, "w")
smh5 = f.create_dataset("stylematrix", (cluster_dict.shape[0], cluster_dict.shape[0]), dtype=np.int32)
smh5[:] = style_matrix[:]
f.close()






























