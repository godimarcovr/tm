import os
import glob
import h5py
import numpy as np
from scipy import misc
from  keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from imageH import ImageH

dst_basefold = "tbs500_nomatch"
src_path = os.path.join("data", dst_basefold)
dst_path = os.path.join("saves", dst_basefold, "features_vgg.hdf5")
dst_path_color = os.path.join("saves", dst_basefold, "features_color.hdf5")

model = VGG16(weights='imagenet', include_top=True)
model = Model(input=model.input, output=model.get_layer('fc1').output)
jpgs = glob.glob(os.path.join(src_path, "*", "*.jpg"))

f = h5py.File(dst_path, "w")
f_col = h5py.File(dst_path_color, "w")
vgg_len = model.get_layer('fc1').output_dim
col_len = 11
feat_len = vgg_len + col_len
f.create_dataset("feats", shape=(len(jpgs), vgg_len), dtype=np.float32)
f_col.create_dataset("feats", shape=(len(jpgs), col_len), dtype=np.float32)
for i, jpg_path in enumerate(jpgs):
    print(i)
    I1 = ImageH(jpg_path, compute_mask=True, compute_hist=True, compute_clusters=False, num_cluster_color=5)
    jpg = misc.imread(jpg_path)
    f_col["feats"][i] = I1.color_hist_scores[np.argsort(I1.color_hist_index)]
    if len(jpg.shape) == 3:
        jpg = misc.imresize(jpg, (224, 224), interp='nearest')
        jpg = jpg / 255.0
        jpg = jpg[np.newaxis, :]
        feat = model.predict(jpg, batch_size=1)
        f["feats"][i] = feat
    else:
        print(jpg_path)
        break


f.close()
f_col.close()