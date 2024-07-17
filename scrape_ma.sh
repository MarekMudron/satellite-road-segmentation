#!/bin/sh

wget -nd -r -P data/train_sat_temp -A tiff https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/index.html
python resize_images.py -i data/train_sat_temp -o data/train -e tiff -t sat

wget -nd -r -P data/train_mask_temp -A tif https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/index.html
python resize_images.py -i data/train_mask_temp -o data/train -e tif -t mask
