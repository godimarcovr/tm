import os
import h5py
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

dst_basefold = "tbs500_nomatch"
src_path = os.path.join("data", dst_basefold)
feat_path = os.path.join("saves", dst_basefold, "features_vgg.hdf5")
col_feat_path = os.path.join("saves", dst_basefold, "features_col.hdf5")
dist_vgg_path = os.path.join("saves", dst_basefold, "dist_vgg.hdf5")

f = h5py.File(feat_path, "r")

features = f["feats"][:]
f.close()

if not os.path.isfile(dist_vgg_path):
    distmat = pairwise_distances(features[:], metric="cosine")
    f = h5py.File(dist_vgg_path, "w")
    f["dists"] = distmat
    f.close()











