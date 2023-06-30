import sys, os, distutils.core
import torch, detectron2
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

from os import path
import random
import shutil
import time
import json
import cv2
import torch.nn.functional as F

import numpy as np
import pandas as pd

from pycocotools.coco import COCO
from pycocotools import mask as cocomask

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import Visualizer

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data.datasets import load_coco_json
from detectron2.structures import BoxMode, Boxes, Instances
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.modeling import build_model

from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo

print("Setting paths")

#setup path to the annotations and images
PATH_TRAIN_SUBSET_ANNOTATIONS =  "./data/annotations_comb.json"
PATH_TRAIN_SUBSET_IMAGES = "./data/images/"

# choose your model from the model zoo
MODEL_PATH = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"


def load_my_dataset(json_file):
    with open(json_file) as f:
        dataset = json.load(f)
    return dataset

# Register your dataset with detectron2
json_file = PATH_TRAIN_SUBSET_ANNOTATIONS
dataset_name = 'my_dataset_train'

from detectron2.data.datasets import register_coco_instances

# Register the COCO dataset with detectron2
register_coco_instances(
    "my_dataset_train",
    {},
    PATH_TRAIN_SUBSET_ANNOTATIONS,
    PATH_TRAIN_SUBSET_IMAGES
)

# Get the metadata of the registered dataset
metadata = MetadataCatalog.get("my_dataset_train")
dataset_dicts = DatasetCatalog.get("my_dataset_train")

print("Setting config")

coco=COCO(PATH_TRAIN_SUBSET_ANNOTATIONS)
class_count = len(coco.getCatIds())

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(MODEL_PATH))

cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 6

# alternatively one can train on the CPU
#cfg.MODEL.DEVICE='cpu'

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_PATH)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.001 
cfg.SOLVER.MAX_ITER = 10000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, enough for this dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = class_count  # number of food classes + 1
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

print("Starting training")
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# TIMESTAMP: uncomment if you want to add a timestamp to your model name
# curr_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
# os.rename('./output/model_final.pth', f'./output/model_final_{curr_time}_{class_count}_cl.pth')

# YAML: uncomment if you want to dump the config values as yaml
# with open(f"./output/config_{curr_time}_{class_count}_cl.yaml", "w") as f:
#    f.write(cfg.dump())

# ARCHIVE: uncomment if you want copy your model and weights to an archive folder
# src_model = './output/'
# dest_folder = f"./model_archive/{curr_time}_{class_count}_cl/"
# shutil.copytree(src_model, dest_folder)