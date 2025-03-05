#!/bin/bash

# copy data to HOIHO
# 	scp *.py tariq@10.16.126.42:/u/tariq/projects/vulcan
#
# open the vpn tunnel
#   sudo openfortivpn remote1.callaghaninnovation.govt.nz:10443 -u t.abuhashim
#   password: /home/mrt/Desktop/callaghan-innovation/admin/NEW_PASSWORD
#
# log in to Hoiho
#	ssh -X tariq@10.16.126.42

# run command in DETR directory
#ssh h8
cd /u/tariq/projects/nauss/detr/
source /u/tariq/anaconda3/etc/profile.d/conda.sh
conda activate detr
CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
        --dataset_file custom_dataset \
        --coco_path /u/tariq/projects/vulcan \
        --lr=1e-4 \
        --batch_size=2 \
        --num_workers=8 \
        --output_dir="/u/tariq/projects/vulcan/outputs/" \
        --backbone="resnet101" \
        --start_epoch=1400 \
        --epochs=2000 \
        --resume=/u/tariq/projects/vulcan/outputs/checkpoint1399.pth \
        --lr_drop=1800

#python -m torch.distributed.launch \
#       --nproc_per_node=3 \
#       --use_env main.py \
#       --dataset_file your_dataset \
#       --coco_path /u/tariq/projects/nauss/data/ms/exclude_coco \
#       --lr=1e-4 \
#       --batch_size=4 \
#       --num_queries=100 \
#       --num_workers=8  \
#       --output_dir="outputs" \
#       --backbone="resnet101" \
#       --epochs=500 \
#       --resume=outputs/checkpoint.pth

# copy checkpoint back to gannit
scp tariq@10.16.126.42:/u/tariq/projects/nauss/detr/outputs/checkpoint.pth ./

# Create TensorRT model:
#conda activate detr
#cd ~/catkin_ws/src/detr/src
#python build_onnx.py
#trtexec --onnx=detr.onnx --saveEngine=detr.trt
