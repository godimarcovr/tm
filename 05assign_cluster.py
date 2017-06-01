import os
import glob
import h5py
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cosine as cosdist
import matplotlib.pyplot as plt
from PIL import Image as pil_image
import random

dst_basefold = "tbs500_nomatch"
src_path = os.path.join("data", dst_basefold)
feat_path = os.path.join("saves", dst_basefold, "features_vgg.hdf5")
col_feat_path = os.path.join("saves", dst_basefold, "features_color.hdf5")
dist_vgg_path = os.path.join("saves", dst_basefold, "dist_vgg.hdf5")
dsinfo_path = os.path.join("saves", dst_basefold, "dsinfo.pickle")
clustering_info_path = os.path.join("saves", dst_basefold, "clustering_info.hdf5")
cat_path = os.path.join("saves", dst_basefold, "classes.hdf5")

dst_basefold_test = "tbs400"
src_path_test = os.path.join("data", dst_basefold_test)
feat_path_test = os.path.join("saves", dst_basefold_test, "features_vgg.hdf5")
col_feat_path_test = os.path.join("saves", dst_basefold_test, "features_color.hdf5")
dist_vgg_path_test = os.path.join("saves", dst_basefold_test, "dist_vgg.hdf5")
dsinfo_path_test = os.path.join("saves", dst_basefold_test, "dsinfo.pickle")
cat_path_test = os.path.join("saves", dst_basefold_test, "classes.hdf5")
clustering_info_path_test = os.path.join("saves", dst_basefold_test, "clustering_info.hdf5")

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
class_names = list(set(classes.tolist()))

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

gclabels_test = np.ones(classes_test.shape[0], dtype=np.int32) * (-1)


#random.shuffle(class_names)
for cat in class_names:
    print(cat)
    inds = [i for i, c in enumerate(classes) if c == cat]
    inds_test = [i for i, c in enumerate(classes_test) if c == cat]
    knnc = KNeighborsClassifier(n_neighbors=1, metric=cosdist)
    labels = cluster_dict[gclabels[inds]][:, 1]
    knnc.fit(features[inds], labels)
    labels_test = knnc.predict(features_test[inds_test])
    for cl in set(labels_test.tolist()):
        print(cat, cl)
        cl_indexes = [x for x in range(labels.shape[0]) if labels[x] == cl]
        inds_cl = [inds[x] for x in cl_indexes]
        cl_indexes_test = [x for x in range(labels_test.shape[0]) if labels_test[x] == cl]
        inds_cl_test = [inds_test[x] for x in cl_indexes_test]
        knnc2 = KNeighborsClassifier(n_neighbors=1)
        labels2 = cluster_dict[gclabels[inds_cl]][:, 2]
        knnc2.fit(features_col[inds_cl], labels2)
        labels_test2 = knnc2.predict(features_col_test[inds_cl_test])
        # fig1 = plt.figure(1)
        # fig1.canvas.set_window_title('cluster ' + str(cl)) 
        # for i, ngind in enumerate(inds_cl_test[:min([9, len(inds_cl_test)])]):
        #     plt.subplot(3, 3, i + 1)
        #     plt.imshow(pil_image.open(jpgs_test[ngind]))
        # plt.show()
        for cl2 in set(labels_test2.tolist()):
            print(cat, cl, cl2)
            cl_indexes_test2 = [x for x in range(labels_test2.shape[0]) if labels_test2[x] == cl2]
            inds_cl_test2 = [inds_cl_test[x] for x in cl_indexes_test2]
            gclabels_test[inds_cl_test2] = int(np.where(np.all(cluster_dict == np.asarray([cat, cl, cl2]),axis=1))[0])
            
            # fig1 = plt.figure(1)
            # fig1.canvas.set_window_title('cluster2 ' + str(cl2)) 
            # for i, ngind in enumerate(inds_cl_test2[:min([9, len(inds_cl_test2)])]):
            #     plt.subplot(3, 3, i + 1)
            #     plt.imshow(pil_image.open(jpgs_test[ngind]))
            # plt.show()


f = h5py.File(clustering_info_path_test, "w")
f["gclabels"] = gclabels_test
f["cluster_dict"] = np.asarray(cluster_dict)
f.close()






