import os
import glob
import h5py
import numpy as np
from scipy import misc
from  keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from imageH import ImageH

src_path = os.path.join("data", "tbs500_nomatch")
dst_basefold = "tbs_nomatch_500"
dst_path = os.path.join("saves", dst_basefold, "features_vgg.hdf5")
dst_path_bin = os.path.join("saves", dst_basefold, "features_bin_vgg.hdf5")

# cnn_fold = os.path.join("networks", "1494025781.1287217")
# model_path = os.path.join(cnn_fold, "model.hdf5")
# weights_path = os.path.join(cnn_fold, "weights_shocl.13-0.87-0.38_0.91-0.26.hdf5")

# model = load_model(model_path)
# model.load_weights(weights_path)
model = VGG16(weights='imagenet', include_top=True)

# test_path = "data/dataset_abbinamenti100"
# validation_datagen = ImageDataGenerator(rescale=1./255)
# validation_generator = validation_datagen.flow_from_directory(directory=test_path,
#                                                                 target_size=(224, 224),
#                                                                 batch_size=64,
#                                                                 class_mode='categorical')
model = Model(input=model.input, output=model.get_layer('fc1').output)
#model = Model(input=model.input, output=model.get_layer('predictions').output)
#model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#print(model.metrics_names)
# res = model.evaluate_generator(validation_generator, val_samples=(2000 // 64) * 64)
# print(res)
jpgs = glob.glob(os.path.join(src_path, "*", "*.jpg"))

f = h5py.File(dst_path, "w")
f_bin = h5py.File(dst_path_bin, "w")
vgg_len = model.get_layer('fc1').output_dim
col_len = 11
feat_len = vgg_len + col_len
f.create_dataset("feats", shape=(len(jpgs), feat_len), dtype=np.float32)
f_bin.create_dataset("feats", shape=(len(jpgs), vgg_len), dtype=np.int8)
for i, jpg_path in enumerate(jpgs):
    print(i)
    I1 = ImageH(jpg_path, compute_mask=True, compute_hist=True, compute_clusters=False, num_cluster_color=5)
    jpg = misc.imread(jpg_path)
    f["feats"][i, vgg_len:] = I1.color_hist_scores[np.argsort(I1.color_hist_index)]
    if len(jpg.shape) == 3:
        jpg = misc.imresize(jpg, (224, 224), interp='nearest')
        jpg = jpg / 255.0
        jpg = jpg[np.newaxis, :]
        feat = model.predict(jpg, batch_size=1)
        bin_feat = np.copy(feat)
        bin_feat[bin_feat < 0.001] = 0
        bin_feat[bin_feat >= 0.001] = 1
        f["feats"][i, :vgg_len] = feat
        f_bin["feats"][i] = bin_feat
    else:
        print(jpg_path)


f.close()
f_bin.close()