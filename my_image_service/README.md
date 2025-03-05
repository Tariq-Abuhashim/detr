
Introduction:
-------------

This is a sample image service, and trigger/listen nodes.   

The image service will wait for a triggering signal ```std_srvs/srv/Trigger``` to publish an image on ```/arena_camera_node/resized/images```.   

The image trigger node will send the trigger on ```trigger_image``` and listen to the reponse published image.   

Once the image is received, the node will send another trigger, and the loop will continue.   


Build the Docker Image:
-----------------------
```
$ docker build -t my_image_service:humble.
```


Run the Docker Container (terminal 1):
--------------------------------------
```
$ docker run -it --rm -v /path/to/images/:/images my_image_service:humble
```


Setup terminal 2:
-----------------
```
$ docker ps  # Note the container ID of the running my_image_service:humble container
$ docker exec -it <container_id> /bin/bash
$ source /opt/ros/humble/setup.bash
$ source /ros2_ws/install/setup.bash
```


Run the trigger and listen node (in terminal 2):
---------------------------------------------
```
$ cd /ros2_ws/src/my_image_service/my_image_service
$ python3 trigger_image.py
```

Run DETR detection service (in terminal 2):
---------------------------------------------
```
$ cd /ros2_ws/src
$ git clone https://gitlab.com/missionsystems/hyperteaming/detr.git
$ cd detr/src
$ python3 detect_service.py
```

Manual triggering of the image service (in terminal 2):
-------------------------------------------------------
```
$ ros2 service call /trigger_image std_srvs/srv/Trigger "{}"
```

Author:
-------
Tariq Abuhashim for Mission-Systems  
28th of May, 2024
