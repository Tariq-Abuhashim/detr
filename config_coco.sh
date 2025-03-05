#!/bin/bash

echo Building model for COCO dataset - 90 classes ...

#vulkan		1274x800
#DJI/FOV	1422x800

cd src
python3 build_model.py --w 1422 --h 800 --num_classes 90 --checkpoint "../weights/detr-r101-2c7b67e5.pth"
python3 serialise_engine.py --max_batch_size 1

mv model.engine coco.engine
