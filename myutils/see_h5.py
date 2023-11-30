import h5py
import os.path as osp
import numpy as np

data_path = "/work2/08264/baagee/frontera/gns-meshnet-data/gns-data/datasets/pipe/"
split = "train"

f = h5py.File(f"{data_path}/{split}.h5", 'r')
f.keys()
dataset = f["0"]
cells = dataset.keys()
cells.keys()
