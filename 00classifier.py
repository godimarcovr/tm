import os
import glob
import h5py
import numpy as np
from scipy import misc

dst_basefold = "tbs400_test"
src_path = os.path.join("data", dst_basefold)
if not os.path.isdir(os.path.join("saves", dst_basefold)):
    os.mkdir(os.path.join("saves", dst_basefold))
dst_path = os.path.join("saves", dst_basefold, "classes.hdf5")

cat_labels = [f for f in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, f))]

jpgs = glob.glob(os.path.join(src_path, "*", "*.jpg"))
f = h5py.File(dst_path, "w")
f.create_dataset("classes", shape=(len(jpgs), ), dtype=np.int8)


for i, jpg_path in enumerate(jpgs):
    print(i)
    classname = jpg_path.split("\\")[-2]
    classind = cat_labels.index(classname)
    f["classes"][i] = classind
f.close()