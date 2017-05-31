import os
import h5py
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cosine as cosdist

dst_basefold = "tbs500_nomatch"
src_path = os.path.join("data", dst_basefold)
feat_path = os.path.join("saves", dst_basefold, "features_vgg.hdf5")
col_feat_path = os.path.join("saves", dst_basefold, "features_color.hdf5")
dist_vgg_path = os.path.join("saves", dst_basefold, "dist_vgg.hdf5")
dsinfo_path = os.path.join("saves", dst_basefold, "dsinfo.pickle")
clustering_info_path = os.path.join("saves", dst_basefold, "clustering_info.hdf5")
cat_path = os.path.join("saves", dst_basefold, "classes.hdf5")

dst_basefold_test = "tbs500_nomatch"
src_path_test = os.path.join("data", dst_basefold_test)
feat_path_test = os.path.join("saves", dst_basefold_test, "features_vgg.hdf5")
col_feat_path_test = os.path.join("saves", dst_basefold_test, "features_color.hdf5")
dist_vgg_path_test = os.path.join("saves", dst_basefold_test, "dist_vgg.hdf5")
dsinfo_path_test = os.path.join("saves", dst_basefold_test, "dsinfo.pickle")
cat_path_test = os.path.join("saves", dst_basefold_test, "classes.hdf5")

#carico features
f = h5py.File(feat_path, "r")
f_col = h5py.File(col_feat_path, "r")
jpgs = glob.glob(os.path.join(src_path, "*", "*.jpg"))
ids = [x.split("\\")[-1].split(".")[0] for x in jpgs]
features = f["feats"][:]
features_col = f_col["feats"][:]
f.close()
f_col.close()
#carico cluster
f = h5py.File(clustering_info_path, "r")
gclabels = f["gclabels"][:]
cluster_dict = f["cluster_dict"][:]
f.close()
f = h5py.File(cat_path, "r")
classes = f["classes"][:]
f.close()
class_names = set(classes.tolist())

#carico le features di test
f = h5py.File(feat_path_test, "r")
f_col = h5py.File(col_feat_path_test, "r")
jpgs_test = glob.glob(os.path.join(src_path_test, "*", "*.jpg"))
ids_test = [x.split("\\")[-1].split(".")[0] for x in jpgs_test]
features_test = f["feats"][:]
features_col_test = f_col["feats"][:]
f.close()
f_col.close()
f = h5py.File(cat_path_test, "r")
classes_test = f["classes"][:]
f.close()

for cat in class_names:
    inds = [i for i, c in enumerate(classes) if c == cat]
    inds_test = [i for i, c in enumerate(classes_test) if c == cat]
    knnc = KNeighborsClassifier(n_neighbors=1, metric=cosdist)
    labels = cluster_dict[gclabels[inds]][:, 1]
    knnc.fit(features[inds], labels)
    knnc.predict()
    









