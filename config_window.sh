#!/bin/bash

echo "Building model for window detection (1 class) ..."

#vulkan		1273x800
#DJI/FOV	1422x800

cd src
python3 build_model.py --w 1274 --h 800 --num_classes 1 --checkpoint "../detr/outputs/window_res101_cp1999.pth"
python3 serialise_engine.py --max_batch_size 1

mv model.engine window.engine
