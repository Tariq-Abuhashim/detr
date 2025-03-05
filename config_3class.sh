#!/bin/bash

echo Building model for 3class detection - 3 classes ...

#vulkan		1274x800
#DJI/FOV	1422x800

cd src
python3 build_model.py --w 1422 --h 800 --num_classes 3 --checkpoint "../detr/outputs/3classes_res101_cpt1630.pth"
python3 serialise_engine.py --max_batch_size 1

mv model.engine 3class.engine
