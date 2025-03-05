#!/bin/bash

cd src

#FOV or DJI
python3 infer_engine.py --engine "3class.engine" --image "/media/mrt/Whale/data/mission-systems/DJI_longer/images/081142005978.png" --num_classes 3

#Vulcan
#python3 infer_engine.py --engine "model.engine" --image "../samples/recreated_image.png" --num_classes 3
