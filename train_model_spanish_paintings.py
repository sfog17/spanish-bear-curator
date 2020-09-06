#!/usr/bin/env python
# coding: utf-8

from fastai.vision.all import *
from pathlib import Path

path = Path('data/raw')
fns = get_image_files(path)
paintings = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
	batch_tfms=aug_transforms()
)
dls = paintings.dataloaders(path)
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(8)
learn.export('classifier.pkl')