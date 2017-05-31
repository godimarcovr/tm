import os
import glob
import h5py
import numpy as np
from scipy import misc
from  keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

src_path = os.path.join("data", "tbs500_nomatch")
dst_basefold = "tbs_nomatch_500"
dst_path = os.path.join("saves", dst_basefold, "classes.hdf5")
# dst_path_bin = os.path.join("saves", "features_bin_tb_vgg.hdf5")

# cnn_fold = os.path.join("networks", "1494413733.8207455classifier")
# model_path = os.path.join(cnn_fold, "model.hdf5")
# weights_path = os.path.join(cnn_fold, "weights_shocl.19-0.92-0.26_0.97-0.13.hdf5")

train_path = src_path
cat_labels = [f for f in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, f))]
# model = load_model(model_path)
# model.load_weights(weights_path)

jpgs = glob.glob(os.path.join(src_path, "*", "*.jpg"))
f = h5py.File(dst_path, "w")
f.create_dataset("classes", shape=(len(jpgs), ), dtype=np.int8)


for i, jpg_path in enumerate(jpgs):
    print(i)
    classname = jpg_path.split("\\")[-2]
    classind = cat_labels.index(classname)
    f["classes"][i] = classind
    # jpg = misc.imread(jpg_path)
    # if len(jpg.shape) == 3:
    #     jpg = misc.imresize(jpg, (224, 224), interp='nearest')
    #     jpg = jpg / 255.0
    #     jpg = jpg[np.newaxis, :]
    #     pred = model.predict(jpg, batch_size=1)
    #     f["classes"][i] = np.argmax(pred)
    # else:
    #     print(jpg_path)

f.close()