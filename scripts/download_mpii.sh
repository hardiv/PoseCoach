#!/bin/bash
# MPII Human Pose Dataset Download Instructions
#
# IMPORTANT: MPII requires manual download due to licensing requirements.
# You must visit the official website and agree to their terms.
#
# Steps:
# 1. Visit: http://human-pose.mpi-inf.mpg.de/
# 2. Navigate to "Download" section
# 3. Fill out the request form and agree to the license
# 4. Download the following files:
#    - mpii_human_pose_v1.tar.gz (images, ~12.9 GB)
#    - mpii_human_pose_v1_u12_2.zip (annotations)
#
# Expected directory structure after extraction:
#
# data/mpii/
#   images/
#     000001163.jpg
#     000003072.jpg
#     ...
#   annotations/
#     mpii_human_pose_v1_u12_1.mat  (or .json if converted)
#
# Note: MPII annotations are in MATLAB format (.mat).
# You may need to convert them to JSON using the following Python script:
#
# import scipy.io as sio
# import json
# 
# mat = sio.loadmat('mpii_human_pose_v1_u12_1.mat', struct_as_record=False)
# # Extract and convert to JSON format expected by pose_bench
# # See src/pose_bench/datasets/mpii.py for expected format
#
# Configuration:
# Update config.yaml:
#   dataset:
#     name: "mpii"
#     images_root: "data/mpii/images"
#     annotations_json: "data/mpii/annotations/mpii_annotations.json"

echo "MPII Human Pose Dataset requires manual download."
echo ""
echo "Please visit: http://human-pose.mpi-inf.mpg.de/"
echo "Follow the instructions above to download and set up the dataset."
echo ""
echo "Target directory structure:"
echo "  data/mpii/images/          - MPII images"
echo "  data/mpii/annotations/     - Converted JSON annotations"
echo ""
echo "After setup, update config.yaml to use MPII dataset."
