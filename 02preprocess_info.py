import os
import glob
import pickle

dst_basefold = "tbs400_test"
src_path = os.path.join("data", dst_basefold)
feat_path = os.path.join("saves", dst_basefold, "features_vgg.hdf5")
col_feat_path = os.path.join("saves", dst_basefold, "features_col.hdf5")
dsinfo_path = os.path.join("saves", dst_basefold, "dsinfo.pickle")

jpgs = glob.glob(os.path.join(src_path, "*", "*.jpg"))
ids = [x.split("\\")[-1].split(".")[0] for x in jpgs]

dsinfo = {}
assoc = {}

with open(os.path.join(src_path, dst_basefold + ".csv"), "r") as csvinfo:
    lines = csvinfo.readlines()
    for line in lines:
        pieces = line.split(";")
        cloth_id = pieces[0]
        raw_list = pieces[5]
        catname = pieces[6].rstrip()
        if cloth_id in ids:
            dsinfo[cloth_id] = {}
            dsinfo[cloth_id]["index"] = ids.index(cloth_id)
            assoc[ids.index(cloth_id)] = cloth_id
            dsinfo[cloth_id]["cat"] = catname
            dsinfo[cloth_id]["outfit"] = []
            if len(raw_list) <= 2:
                continue
            outfit_list = []
            for raw_list_element in raw_list[1:-1].split(","):
                outfit_list.append(raw_list_element)
            dsinfo[cloth_id]["outfit"] = outfit_list

with open(dsinfo_path, 'wb') as handle:
    pickle.dump((dsinfo, assoc), handle, protocol=pickle.HIGHEST_PROTOCOL)