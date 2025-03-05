cd $HOME/src/repair-red/ && cat /run/user/1000/gvfs/smb-share:server=orac.local,share=share01/datasets/field-trials/hyperteaming/2024-05-singleton/vulcan-logs/2024_05_30_11_explore/cameras/alvium_1800_c240c/20240530T051055.838925.bin | cv-cat "convert-color=rgb,bgr" | python3 predict_red_channel.py | io-publish tcp:4001 --size=7062548 &
nc -l 4003 &
cd $HOME/src/hyperteaming/detr/docker && ./run_model // this does not work right now, need to open docker container without exec and then run ./detect ../configs/vulcan.yaml within the detr/src folder
