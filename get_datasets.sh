#!/bin/bash
# Shell script to download data sets for Lyft challenge, obtained from various sources

# Exit immediately if a command exits with a non-zero status.
set -e

# Go to the data directory and download the original dataset
DATASET_DIR="data"
ORIGINAL_DATASET="https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Lyft_Challenge/Training+Data/lyft_training_data.tar.gz"
ORIG_DIR=""

cd "${DATASET_DIR}"
wget "${ORIGINAL_DATASET}"
tar -xf "${ORIG_DIR}"


