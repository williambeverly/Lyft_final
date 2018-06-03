#!/bin/bash
# Shell script to download data sets for Lyft challenge, obtained from various sources

# Exit immediately if a command exits with a non-zero status.
set -e

# Go to the data directory and download the original dataset
DATASET_DIR="data"
ORIGINAL_DATASET="https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Lyft_Challenge/Training+Data/lyft_training_data.tar.gz"
ORIG_DIR="lyft_training_data.tar.gz"

cd "${DATASET_DIR}"
wget "${ORIGINAL_DATASET}"
tar -xf "${ORIG_DIR}"
# now remove the original dataset
rm "${ORIG_DIR}"

# Go into the train directory, and download more
TRAIN_DIR="Train"
ADD_DATASET_1="https://www.dropbox.com/s/1etgf32uye2iy8q/world_2_w_cars.tar.gz"
ADD_DIR_1="world_2_w_cars.tar.gz"
ADD_DATASET_2="https://github.com/ongchinkiat/LyftPerceptionChallenge/releases/download/v0.1/carla-capture-20180528.zip"
ADD_DIR_2="carla-capture-20180528.zip"
cd "${TRAIN_DIR}"
wget "${ADD_DATASET_1}"
tar -xf "${ADD_DIR_1}"
# now remove the original dataset
rm "${ADD_DIR_1}"

ADD_2_FOLDER='Carla_Capture'
mkdir -p "${ADD_2_FOLDER}"
cd "${ADD_2_FOLDER}"
wget "${ADD_DATASET_2}"
unzip "${ADD_DIR_2}"
# now remove the original dataset
rm "${ADD_DIR_2}"
