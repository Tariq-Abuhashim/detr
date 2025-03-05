port=`cat /usr/local/etc/weaver/config.json | python3 -c "import sys;import json;print(json.load(sys.stdin)['camera']['port'])"`
desired_width=1274
desired_height=800
center_x=$(( original_width/2  ))
center_y=$(( original_height/2  ))
source /opt/ros/humble/setup.bash && cd $HOME/src/hyperteaming/detr/src && io-cat tcp:localhost:$(( port )) | python3 $HOME/src/hyperteaming/detr/src/detect_service_geoimage.py --num_classes 3 --n bull --trained_width $desired_width --trained_height $desired_height --time_filter 5 --spatial_filter 0
