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

top_cats = [1, 2, 3, 4, 8, 9, 13]
bottom_cats = [5, 6, 7]
shoe_cats = [0, 10, 11, 12]

def get_big_cat(cat):
    if cat in top_cats:
        return 0
    elif cat in bottom_cats:
        return 1
    else:
        return 2

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

usertest_fold = os.path.join("usertest")
if not os.path.isdir(usertest_fold):
    os.mkdir(usertest_fold)

#carico immagini test
jpgs_test = glob.glob(os.path.join(src_path_test, "*", "*.jpg"))

inds = [x for x in range(len(jpgs_test))]
random.shuffle(inds)

usertest_groundtruth_path = os.path.join(usertest_fold, "ground_truth.hdf5")
usertest_number_of_pairings = 500
usertest_groundtruth = np.zeros((usertest_number_of_pairings, ), dtype=np.int8)
usertest_count = 0
for ind in inds:
    if usertest_count >= usertest_number_of_pairings:
        break
    gcl_test = gclabels_test[ind]
    cat_test = cluster_dict[gcl_test, 0]
    big_cat_test = get_big_cat(cat_test)
    smrow = style_matrix[gcl_test, :]
    pairing_rank = np.argsort(smrow)[::-1].tolist()
    pairing_rank = [x for x in pairing_rank if smrow[x] > 0 and not get_big_cat(cluster_dict[x, 0]) == big_cat_test]
    if len(pairing_rank) > 0:
        bestgc = pairing_rank[0]
        bestbigcat = get_big_cat(cluster_dict[bestgc, 0])
        candidates = [i for i, x in enumerate(gclabels_test) if x == bestgc]
        if len(candidates) > 0:
            random.shuffle(candidates)
            candidate_ind = candidates[0]
            classes = cluster_dict[gclabels_test[:]][:, 0]
            same_cat_inds = [i for i, x in enumerate(classes) if x == cluster_dict[bestgc, 0]]
            different_gc_inds = [i for i, x in enumerate(gclabels_test) if not x == bestgc]
            candidates2 = list(set(same_cat_inds) & set(different_gc_inds))
            random.shuffle(candidates2)
            fig1 = plt.figure(1)
            if big_cat_test < bestbigcat:
                p1 = 1
                p2 = 3
            else:
                p1 = 3
                p2 = 1
            plt.subplot(2, 2, p1)
            plt.imshow(pil_image.open(jpgs_test[ind]))
            plt.subplot(2, 2, p1 + 1)
            plt.imshow(pil_image.open(jpgs_test[ind]))

            flipflag = random.random() > 0.5
            plt.subplot(2, 2, p2 + 1 if flipflag else p2)
            plt.imshow(pil_image.open(jpgs_test[candidate_ind]))
            plt.subplot(2, 2, p2 if flipflag else p2 + 1)
            plt.imshow(pil_image.open(jpgs_test[candidates2[0]]))
            # plt.show()
            plt.savefig(os.path.join("usertest", str(usertest_count) + ".jpg"))
            plt.close()
            print(usertest_count)
            usertest_groundtruth[usertest_count] = 1 if flipflag else 0
            usertest_count += 1
            
f = h5py.File(usertest_groundtruth_path, "w")
f["usertest_groundtruth"] = usertest_groundtruth
f.close()




