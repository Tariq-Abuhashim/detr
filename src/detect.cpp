/**
* 
* Tariq Abuhashim for mission-systems hyperteaming
* ...
*
* 21-Oct-2022
*
*/

//#include <stdio.h>
//#include <stdlib.h>
//#include <algorithm>
//#include <fstream>
//#include <ctime>
//#include <sstream>
//#include <vector>

#include <Python.h>

#include <unistd.h>
#include <chrono>
#include <vector>
#include <string>
#include <thread>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "System.h"
#include "tcp_interface.hpp"

int main(int argc, char **argv)
{
    if(argc < 2)
    {
        std::cerr << std::endl << "Usage: ./detect config_file" << std::endl;
		std::cerr << std::endl << "Examples:" << std::endl;
		std::cerr << std::endl << "./detect vulcan.yaml" << std::endl;
        return 1;
    }

	std::cout << std::endl << "-------" << std::endl;
    std::cout.precision(12);

	// processing thread (syncs data in ldQueue and imQueue and runs python function)
	std::cout << "\n Start process thread ..." << std::endl;
	DETR::System DETR(argv[1]);

	// open port settings
	cv::FileStorage fSettings("../configs/ports.yaml", cv::FileStorage::READ);
	if(!fSettings.isOpened())
	{
		cerr << "Failed to open settings file at: configs/ports.yaml" << endl;
		exit(-1);
	}

	// set where to send the detection results
    // you can listen to this data using  $ nc -l 4003
    // TODO how to relate this connection to serial in COVINS
	DETR.setSocket(fSettings["host"], fSettings["det2d.port"]); /* ("127.0.0.1", 4003) */

	// Initialise readers
	dataQueue imageQueue; // for ImageData
    ImageReader image_reader(imageQueue,"127.0.0.1", 4001, "t,3ui,s[7062528]"); /* ("127.0.0.1", 4001) */

	// Start threads to read from the sensors
	std::cout << "\n Start data threads ..." << std::endl;
    std::thread t2(&ImageReader::readFromSocket, &image_reader); // fills imageQueue
	//usleep(1e6); // (not reliable) allow t1 and t2 to start
	std::this_thread::sleep_for(std::chrono::seconds(1)); // ensure that threads have started before moving on

    std::cout << "\n Start processing thread ..." << std::endl;
    while (image_reader.is_connected()) {  // continue as long as both readers are alive

		while (!image_reader.safeEmpty()) {
			std::shared_ptr<SensorData> image_ptr = image_reader.safeFront();
			std::shared_ptr<ImageData> actual_image_ptr = std::dynamic_pointer_cast<ImageData>(image_ptr);
			ImageData& image = *actual_image_ptr; // dereferencing

			// you should directly use the result of std::bind which is a callable object, and can be invoked with the () operator.
			auto func = std::bind(&DETR::System::processImageData, &DETR, std::placeholders::_1);
			func(image); // Invoke the function
			image_reader.safePop();
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Sleep to avoid busy waiting
	}

	std::cout << "\n Joining camera thread ..." << std::endl;
	t2.join();

    // Stop all threads
	std::cout << "\n Finalize the interpreter ..." << std::endl;
	DETR.finalize();

	std::cout << "\n Shutdown SLAM ..." << std::endl;
    //DETR.Shutdown();
    return 0;
}

