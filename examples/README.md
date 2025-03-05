# Example Detection Pipeline

The script in `repair_red_and_detect.sh` will commence a TCP server on port 4003. It will also commence a TCP io-publish service on port 4001 with a video from minivault logs, repair the red channel, then pipe this into the detection model.