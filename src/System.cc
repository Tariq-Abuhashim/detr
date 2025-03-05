
#include "System.h"
#include <thread>
#include <iomanip>
#include <openssl/md5.h>
#include <fstream>
#include <opencv2/opencv.hpp>

long long lidar_index = 0;
int image_index = 0;

namespace DETR
{

	System::System(const string &strSettingsFile)
	{
		// Output welcome message
		cout << endl <<
		"---------------------------------------------------------------------------------------" << endl <<
		"DETR System interface Copyright (C) 2023-2024 Tariq Abuhashim, Mission Systems ltd." << endl <<
		"This is a fusion of pybind11, TCP interface with comma/snark, DETR model with tensorRT;" << endl  <<
		"---------------------------------------------------------------------------------------" << endl;

		/* Check settings file */
		
		cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
		if(!fsSettings.isOpened())
		{
		   cerr << "Failed to open settings file at: " << strSettingsFile << endl;
		   exit(-1);
		}
		
		/* Python interpreter */
			
		std::cout << "Starting Python interpreter ..." << std::endl;
		py::initialize_interpreter(); // Initialize first

		/* actual code */

		std::cout << "Import modules ..." << std::endl;
		PyObject* pName = PyUnicode_FromString("infer_engine"); // "infer_engine" was "detect_trt"
		pModule = PyImport_Import(pName);
		Py_DECREF(pName);

		if (pModule != nullptr) {
			// The script was loaded successfully.
			std::cout << "Done ..." << std::endl;
		} else {
			// Handle the error. Use PyErr_Print() to get info about the error.
			PyErr_Print();
			fprintf(stderr, "Failed to load \"%s\"\n", "infer_engine"); // "infer_engine" was "detect_trt"
			//return 1;
		}
		
		// Create an instance of the TensorRTInference class
		pClass = PyObject_GetAttrString(pModule, "TensorRTInference");
		pInstance = PyObject_CallObject(pClass, nullptr); // Add arguments if the constructor requires them
		Py_DECREF(pClass);
			
		if (pInstance == nullptr) {
			PyErr_Print();
			std::cerr << "Failed to create Python class instance" << std::endl;
			//return 1;  // Or handle the error as appropriate
		}

		/* Initialise */
		
		cout << "Initialise python thread ..." << endl;
		InitThread();
		PyEval_ReleaseThread(PyThreadState_Get());

	}

	void System::setSocket(const std::string& server, const std::string& port) {
		cout << "Initialise boost socket to send detections ..." << endl;
		// Resolve the server address and port
		boost::asio::io_service io_service;
		socket_ =  new boost::asio::ip::tcp::socket(io_service);
		boost::asio::ip::tcp::resolver resolver(io_service);
		boost::asio::ip::tcp::resolver::query query(server, port);
		boost::asio::ip::tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);
		// Try to connect to the server
		boost::asio::ip::tcp::resolver::iterator end;
	   	boost::system::error_code error = boost::asio::error::host_not_found;
		while (endpoint_iterator != end && error) {
			socket_->close();
			socket_->connect(*endpoint_iterator++, error);
		}
		if (error) {
			throw boost::system::system_error(error);
		}
	}

	// Example function to process synchronized data
	//void System::processSyncedData(const CloudData& data1, const ImageData& data2) {
	//	// Replace this with your actual processing code
	//	std::cout << "Processing data: Lidar time(" << data1.getTimestamp()/1e9 << "), Image time(" << data2.getTimestamp()/1e9 << ")" << std::endl;
	//}

	/* a strictly CloudData and ImageData function */
	void System::processSyncedData(const CloudData& lidar, const ImageData& image, dataQueue& imuQueue) {
		// get imu measurement
		double lidarTime = lidar.getTimestamp()/1e+9;
		double imuTime = -1;
		ImuData* imu = nullptr;
		while (!imuQueue.empty()) {
		//for (auto& imu_ptr : imuQueue) {
			//std::lock_guard<std::mutex> guard_cloud(reader1.mtxQueue);
			std::shared_ptr<SensorData> imu_ptr = imuQueue.front();
			std::shared_ptr<ImuData> actual_imu_ptr = std::dynamic_pointer_cast<ImuData>(imu_ptr);
			if (actual_imu_ptr) {
				imuTime = actual_imu_ptr->getTimestamp()/1e+9;

				std::cout << lidarTime << " " << imuTime << std::endl;

				if (std::abs(imuTime-lidarTime) < EPSILON) {
					imu = actual_imu_ptr.get(); // get raw pointer
					break;
				}
			}
			imuQueue.pop_front();  // Pop the checked IMU data from the front
		}
		if (!imu) {
			std::cerr << "No Imu match" << std::endl;
			return;
		}
    	// get the pointcloud
		std::vector<Point> pointCloud = lidar.getData().points;
		/*
			Add code here to align the point cloud using imu orientation and height over ground
		*/
		// get the image
		Image I = image.getData();
		cv::Mat Cvimage(I.height, I.width, CV_8UC3, I.data.data()); // does not copy the data. If vecData goes out of scope or is modified, the cv::Mat will be affected.
    	//std::cout << "Processing data: Lidar time(" << data1.getTimestamp()/1e9 << "), Image time(" << data2.getTimestamp()/1e9 << ")" << std::endl;
		//std::cout << "Processing data: Lidar data(" << pointCloud.size() << "), Image data(" << I.height << "x" << I.width << ")" << std::endl;
		// call python function
		call_python_function(image.getTimestamp(), Cvimage, pointCloud, imu->getData());
	}
	
	/*
	void System::processImageData (const ImageData& image) {
		Image I = image.getData();
		cv::Mat Cvimage(I.height, I.width, CV_8UC3, I.data.data()); // does not copy the data. If vecData goes out of scope or is modified, the cv::Mat will be affected.
    	std::cout << "Processing data: Image time(" << image.getTimestamp()/1e9 << ")" << std::endl;
		std::cout << "Processing data: Image data(" << Cvimage.rows << "x" << Cvimage.cols << ")" << std::endl;
		// call python function
		call_python_function(Cvimage);
	}
	*/

	void System::processImageData(const ImageData& image) {
    	// Extract image data
    	Image I = image.getData();
    	cv::Mat Cvimage(I.height, I.width, CV_8UC3, I.data.data());
    	
    	//std::cout << "Processing data: Image time(" << image.getTimestamp()/1e9 << ")" << std::endl;
		//std::cout << "Processing data: Image data(" << Cvimage.rows << "x" << Cvimage.cols << ")" << std::endl;

    	// Define a callback function to process the outputs from Python
    	auto callback = [&](const std::vector<std::vector<float> >& probas, const std::vector<cv::Rect2d>& bboxes_scaled) {
    		//std::cout << probas.size() << " x " << bboxes_scaled.size() << std::endl;
        	// Process the outputs here
        	// For example, print them or store them in member variables
        	
            std::stringstream ss;
			std::stringstream string_stream_prefix;
        	// string_stream_prefix << image.getTimestamp() // time
			//   << "," << bboxes_scaled.size(); // time, int
			



			for (size_t i = 0; i < bboxes_scaled.size(); ++i) {
				//auto mask = det.attr("mask").cast<Eigen::Vector3f>();
				auto prob = probas[i]; // 3xfloat
				auto bbox = bboxes_scaled[i]; // 4xfloat;
				auto maxIt = std::max_element(prob.begin(), prob.end()); // maximum probability 1xfloat
				int index = std::distance(prob.begin(), maxIt); // index of maximum probability 1xint
				ss << std::to_string(image_index) << "," << image.getTimestamp() << "," << std::to_string(index) << "," << std::to_string(*maxIt) << "," << bbox.x << "," << bbox.y << "," << bbox.width << "," << bbox.height << "\n";
			}
			// ss << "\n";  // Add newline to separate detections

			if(bboxes_scaled.size() > 0){ // only send output if there is a detection
		
				if (socket_->is_open()) {
					std::string data_to_send = ss.str();   
					boost::system::error_code error;
					boost::asio::write(*socket_, boost::asio::buffer(data_to_send), boost::asio::transfer_all(), error);
					if (error) {
						throw boost::system::system_error(error);
					}
				}
				else {
					std::cerr << "call_python_function: Socket_ is not open" << std::endl;
				}

				image_index += 1;

			}
		};

    	// Call Python function 'detect' and pass the callback function
    	call_python_function(Cvimage, callback);
	}
	
	
	// python function callback using image and lidar points
	void System::call_python_function(const cv::Mat& image) {
	    PyThreadStateLock PyThreadLock;
	    
		PyObject* pValue = PyObject_CallMethod(pInstance, "infer", "(s)", "/media/dv/Whale/Orin/annotate/images/test/20230210T081220.325959.png");   // "infer" was "detect"
		
		// Convert cv::Mat and std::vector<Eigen::Vector3d> to numpy arrays
		//auto np_Image = cvMatToNumpyArray(image);
		// Call the 'detect' method with the NumPy array as argument
		//PyObject* pValue = PyObject_CallMethod(pInstance, "detect", "(O)", np_Image);

		if (pValue != nullptr) {
    		// Process the return value
    		// ...
    		std::cout << "Call to method detect succeeded" << std::endl;

    		Py_DECREF(pValue);
		} else {
    		PyErr_Print();
    		std::cerr << "Call to method detect failed" << std::endl;
		}
	}
	
	// python function callback using image and lidar points
	void System::call_python_function(const cv::Mat& image, const ProcessImageDataFunc& callback) {
	    PyThreadStateLock PyThreadLock;
		
		// Convert cv::Mat and std::vector<Eigen::Vector3d> to numpy arrays
		//cv::Mat image = cv::imread("/home/mrt/dev/mission-systems/repair-red/recreated_image.png", cv::IMREAD_COLOR);
		auto np_Image = cvMatToNumpyArray(image);
		//std::cout << "Array dtype: " << py::cast<std::string>(py::str(np_Image.attr("dtype"))) << std::endl;
		//std::cout << "Array dtype: " << py::cast<std::string>(py::str(np_Image.attr("shape"))) << std::endl;
		
		// Call the 'detect' method with the NumPy array as argument
		PyObject* np_Image_pyobj = np_Image.ptr(); // Convert py::array to PyObject*
		PyObject* pResult = PyObject_CallMethod(pInstance, "infer", "(O)", np_Image_pyobj); // "infer" was "detect"
		if (pResult != nullptr) {
    		// Process the return value
    		// ...
    		// Extract data from the returned Python object
			PyObject* probas_obj = PyTuple_GetItem(pResult, 0); // Assuming probas is the first element
			PyObject* bboxes_scaled_obj = PyTuple_GetItem(pResult, 1); // Assuming bboxes_scaled is the second element

			// Convert Python objects to C++ vectors
			std::vector<std::vector<float> > parsed_probas = numpyArrayToVector(probas_obj);
			std::vector<cv::Rect2d> parsed_bboxes_scaled = numpyArrayToRect2dVector(bboxes_scaled_obj);

			// Call the callback function with the parsed data
			callback(parsed_probas, parsed_bboxes_scaled);

    		Py_DECREF(pResult);
		} else {
    		PyErr_Print();
    		std::cerr << "Call to method detect failed" << std::endl;
		}
	}
	
	// python function callback using image and lidar points
	void System::call_python_function(const double& Time, const cv::Mat& image, const std::vector<Point>& pointCloud, 
										const Imu& imu) {
		PyThreadStateLock PyThreadLock;
		//py::gil_scoped_acquire acquire;

		// Convert cv::Mat and std::vector<Eigen::Vector3d> to numpy arrays
		auto np_Image = cvMatToNumpyArray(image);

		std::cout << imu.orientation[0] << " " << imu.orientation[1] << " " << imu.orientation[2] << std::endl;

		// rotation angles (for mission systems)
		double r = 0;//imu.orientation[0] * M_PI/180.0;  // rotation around x-axis
		double p = 0;//imu.orientation[1] * M_PI/180.0;  // rotation around y-axis
		double y = 0;//imu.orientation[2] * M_PI/180.0;  // rotation around z-axis

		// construct rotation matrix using a roll-pitch-yaw convention (order in which you multiply the Eigen::AngleAxisd instances matters)
		Eigen::Matrix3d R;
		R = Eigen::AngleAxisd(r, Eigen::Vector3d::UnitX())
			* Eigen::AngleAxisd(p, Eigen::Vector3d::UnitY())
			* Eigen::AngleAxisd(y, Eigen::Vector3d::UnitZ());
		Eigen::Vector3d t(0.0, 0.0, 0.0);
    	auto np_PointCloud = vector3dToNumpyArray(pointCloud, R, t);

		// Serialise data to the hard-drive
		writeToPCD(pointCloud);
		//writeToBIN(pointCloud);
		writeToPNG(image);
		writeToNAV(imu);
		increment(); // moves the file name counter
		return;

		// Call the method with the numpy arrays
    	//obj.attr("my_method")(np_Image, np_PointCloud); // Call the method
		py::list detections = pySequence.attr("get_frame")(np_Image, np_PointCloud);

		if (detections.size()<1) return;

		std::stringstream ss;
        ss << Time  << "," << detections.size(); // Comma Se
		for (auto det : detections) {
			//auto mask = det.attr("mask").cast<Eigen::Vector3f>();
			auto bbox = det.attr("bbox").cast<Eigen::Vector4f>();
			auto label = det.attr("label").cast<int>();
			auto score = det.attr("score").cast<float>();
			if (score > 0.85f) {
				// Convert bbox, label, and score to string format and send to socket
                ss << "," << label << "," << bbox.transpose() << "," << score;
			}
		}
		ss << "\n";  // Add newline to separate detections

		if (socket_->is_open()) {
			std::string data_to_send = ss.str();   
		    boost::system::error_code error;
			boost::asio::write(*socket_, boost::asio::buffer(data_to_send), boost::asio::transfer_all(), error);
			if (error) {
				throw boost::system::system_error(error);
			}
		}
		else {
			std::cerr << "call_python_function: Socket_ is not open" << std::endl;
		}

	}
	
	
	// increments file name counter
	void System::increment(){
		++lidar_index;
	}

	// saves lidar data to pcd file
	void System::writeToPCD(const std::vector<Point>& input)
	{
		// Open file for writing
		char s[200];
		sprintf(s, "/home/mrt/data/mission_systems/output/pcd/%010lld.pcd", lidar_index); 
		std::ofstream file(s);
		if (!file) {
        	std::cerr << "Could not open file for writing\n";
    	}

		// Write the header
		file << "# .PCD v.7 - Point Cloud Data file format\n"
		     << "VERSION .7\n"
		     << "FIELDS x y z intensity\n"   // Add 'intensity' to FIELDS
		     << "SIZE 4 4 4 4\n"             // Add '4' to SIZE (assuming float intensity)
		     << "TYPE F F F F\n"             // Add 'F' to TYPE (assuming float intensity)
		     << "COUNT 1 1 1 1\n"            // Add '1' to COUNT
		     << "WIDTH " << input.size() << "\n"
		     << "HEIGHT 1\n"
		     << "VIEWPOINT 0 0 0 1 0 0 0\n"
		     << "POINTS " << input.size() << "\n"
		     << "DATA ascii\n";

		// get the max value
		float maxIntensity = -1;
		for (const auto& vec : input) {
			if (static_cast<float>(vec.intensity) > maxIntensity && vec.range > 2.0f) // TODO determine this minimum range
				maxIntensity = static_cast<float>(vec.intensity);
		}
		// Write the point data
		for (const auto& vec : input) {
			if (vec.range < 2.0f) continue; // TODO determine this minimum range
			float x = static_cast<float>(vec.x); //-(0*static_cast<float>(vec.x) + 0*static_cast<float>(vec.y) + 1*static_cast<float>(vec.z) + 0.25);
			float y = static_cast<float>(vec.y); //-(0*static_cast<float>(vec.x) - 1*static_cast<float>(vec.y) + 0*static_cast<float>(vec.z) + 0.10);
			float z = static_cast<float>(vec.z); // (1*static_cast<float>(vec.x) + 0*static_cast<float>(vec.y) + 0*static_cast<float>(vec.z) + 0.00);
			float i = (static_cast<float>(vec.intensity)/maxIntensity); // when using signal
			//float i =  (static_cast<float>(vec.intensity)/255.0f); // when using reflectivity
			//TODO there might be a geometric transformation here
			file << x << " " << y << " " << z << " " << i << "\n";
		}

		file.close();
	}

	// saves lidar data to bin file
	void System::writeToBIN(const std::vector<Point>& input) { 
		// Open file for writing
		char s[200];
		sprintf(s, "/home/mrt/data/mission_systems/output/bin/%010lld.bin", lidar_index); 
		std::ofstream bin_file(s, ios::out|ios::binary|ios::app);
		if (!bin_file.good()) {
        	std::cerr << "Couldn't open " << s << std::endl;
    	}
		// get the max value
		float maxIntensity = -1;
		for (const auto& vec : input) {
			if (static_cast<float>(vec.intensity) > maxIntensity && vec.range > 2.0f) // TODO determine this minimum range
				maxIntensity = static_cast<float>(vec.intensity);
		}
		std::cout << "maxIntensity=" << maxIntensity << std::endl;
		// Coordinate transformation and filling the file with data
		for (const auto& vec : input) {
			if (vec.range < 2.0f) continue; // TODO determine this minimum range
			float x = static_cast<float>(vec.x);
			float y = static_cast<float>(vec.y);
			float z = static_cast<float>(vec.z);
			//float x_ = ( 0*x + 0*y - 1*z + 0.25);
			//float y_ = (-1*x + 0*y + 0*z + 0.10);
			//float z_ = ( 0*x + 1*y + 0*z + 0.00);
			float x_ = ( 1*x + 0*y + 0*z + 0.18016);
			float y_ = ( 0*x + 1*y + 0*z + 0.03540);
			float z_ = ( 0*x + 0*y + 1*z + 0.10900);
			float i_ =  (static_cast<float>(vec.intensity)/maxIntensity);
			//float i_ =  (static_cast<float>(vec.intensity)/255.0f);
			bin_file.write((char*)&x_, sizeof(float));
			bin_file.write((char*)&y_, sizeof(float));
			bin_file.write((char*)&z_, sizeof(float));
    		bin_file.write((char*)&i_, sizeof(float));
		}
	}

	// saves image data to png file
	void System::writeToPNG(const cv::Mat& mat)
	{
		char s[200];
		sprintf(s, "/home/mrt/data/mission_systems/output/png/%010lld.png", lidar_index);
		cv::imwrite(s, mat);
	}

	void System::writeToNAV(const Imu& imu)
	{
		char s[200];
		sprintf(s, "/home/mrt/data/mission_systems/output/nav/%010lld.txt", lidar_index);
		std::ofstream file(s);
		if (!file) {
        	std::cerr << "Could not open file for writing\n";
    	}
		file<< setprecision(9) << 0.0 << " " << 0.0 << " " << 0.0 << " " 
			<< imu.orientation[0] << " " << imu.orientation[1] << " " << imu.orientation[2] << " " 
			<< 0.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << " " << 0.0 << " " 
			<< imu.accelerometer[0] << " " << imu.accelerometer[1] << " " << imu.accelerometer[2] << " " 
			<< 0.0 << " " << 0.0 << " " << 0.0 << " " 
			<< imu.gyroscope[0] << " " << imu.gyroscope[1] << " " << imu.gyroscope[2] << " " << 0.0 << " " 
			<< 0.0 << " " << " " << 0.0 << " " << 0.0 << endl;
		file.close();
	}

	// converts lidar data to numpy array
	py::array_t<float> System::vector3dToNumpyArray(const std::vector<Point>& input, const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
		py::array_t<float> result(input.size() * 4); // One 3D point is represented by three double values + intensity
		py::buffer_info bufInfo = result.request();
		float* ptr = static_cast<float*>(bufInfo.ptr);
		// get the max reflectivity
		float maxIntensity = -1;
		for (const auto& vec : input) {
			if (static_cast<float>(vec.intensity) > maxIntensity)
				maxIntensity = static_cast<float>(vec.intensity);
		}
		// Apply rotation and translation and store in py::array_t
		for (const auto& vec : input) {
			Eigen::Vector3d point(vec.x, vec.y, vec.z);
			Eigen::Vector3d transformedPoint = R * point + t;
		    *ptr++ = static_cast<float>(transformedPoint.x());
		    *ptr++ = static_cast<float>(transformedPoint.y());
		    *ptr++ = static_cast<float>(transformedPoint.z());
			*ptr++ = static_cast<float>(vec.intensity)/maxIntensity;
		}
		std::cout << "maxIntensity=" << maxIntensity << std::endl;
		std::vector<size_t> shape = {input.size(), 4};
		result.resize(shape); // Reshape to a 2D array
		return result;
	};

	// converts image data to numpy array
	py::array_t<unsigned char> System::cvMatToNumpyArray(const cv::Mat& mat) {
		std::vector<size_t> shape = {static_cast<size_t>(mat.rows), 
									static_cast<size_t>(mat.cols), 
									static_cast<size_t>(mat.channels())};
		//py::array_t<unsigned char> result = py::array_t<unsigned char>(shape, mat.data);
		py::array_t<unsigned char> result = py::array_t<unsigned char>(shape);
		std::memcpy(result.mutable_data(), mat.data, mat.total() * mat.elemSize());
		return result;
	};
	
	// Convert a NumPy array to a vector of Probabilities
	std::vector<std::vector<float>> System::numpyArrayToVector(PyObject* numpy_array) {
		py::handle handle(numpy_array);
		if (!py::isinstance<py::array>(handle)) {
		    throw std::runtime_error("Object is not a NumPy array");
		}

		py::array np_array = py::cast<py::array>(handle);
		//std::cout << "Array dtype: " << py::cast<std::string>(py::str(np_array.attr("dtype"))) << std::endl;
		//std::cout << "Array shape: " << py::cast<std::string>(py::str(np_array.attr("shape"))) << std::endl;
		if (!py::isinstance<py::array_t<float>>(np_array)) {
		    throw std::runtime_error("NumPy array should contain float");
		}
		
		// Nx3 for {window}, Nx5 for {person,car,window}
		if (np_array.ndim() != 2 || (np_array.shape(1) != 3 && np_array.shape(1) != 5)) {
		    throw std::runtime_error("NumPy array must have a shape of Nx3 or Nx5");
		}
		
		// Use safer access if unchecked is problematic
		auto r = np_array.unchecked<float, 2>(); // For 2D arrays of float
		std::vector<std::vector<float>> result;
		result.reserve(r.shape(0)); // Pre-allocate memory

		for (ssize_t i = 0; i < r.shape(0); ++i) {
		    std::vector<float> vec;
		    vec.push_back(r(i, 0));
		    vec.push_back(r(i, 1));
		    vec.push_back(r(i, 2));

		    if (r.shape(1) == 5) {
		        vec.push_back(r(i, 3));
		        vec.push_back(r(i, 5));
		    }
		    
		    result.emplace_back(vec);
		}

		return result;
	}


	// Convert a NumPy array to a vector of OpenCV Rectangles
	std::vector<cv::Rect2d> System::numpyArrayToRect2dVector(PyObject* numpy_array) {
		py::handle handle(numpy_array);
		if (!py::isinstance<py::array>(handle)) {
		    throw std::runtime_error("Object is not a NumPy array");
		}

		py::array np_array = py::cast<py::array>(handle);
		//std::cout << "Array dtype: " << py::cast<std::string>(py::str(np_array.attr("dtype"))) << std::endl;
		//std::cout << "Array shape: " << py::cast<std::string>(py::str(np_array.attr("shape"))) << std::endl;
		if (!py::isinstance<py::array_t<float>>(np_array)) {
		    throw std::runtime_error("NumPy array should contain float");
		}

		if (np_array.ndim() != 2 || np_array.shape(1) != 4) {
		    throw std::runtime_error("NumPy array must have a shape of Nx4");
		}
		
		// Use safer access if unchecked is problematic
		auto r = np_array.unchecked<float, 2>(); // For 2D arrays of double, if causing issues, remove <double>
		std::vector<cv::Rect2d> result;
		result.reserve(r.shape(0)); // Pre-allocate memory
		
		for (ssize_t i = 0; i < r.shape(0); ++i) {
			//std::cout << "Row " << i << ": " << r(i, 0) << ", " << r(i, 1) << ", " << r(i, 2) << ", " << r(i, 3) << std::endl;
		    double x = r(i, 0);
		    double y = r(i, 1);
		    double width = r(i, 2);
		    double height = r(i, 3);
		    result.emplace_back(x, y, width, height);
		}
		
		//for (ssize_t i = 0; i < result.size(); ++i) {
		//    const auto& lastRect = result[i];
		//    std::cout << "Row " << i << ": " << lastRect.x << ", " << lastRect.y << ", " << lastRect.width << ", " << lastRect.height << std::endl;
		//}

		return result;
	};


} //namespace DETR


