
#ifndef SYSTEM_H
#define SYSTEM_H

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <opencv2/core/core.hpp>

#include "tcp_interface.hpp" // mission-systems
#include <pybind11/embed.h>
#include <pybind11/eigen.h>

#include <functional> // for defining ProcessImageDataFunc

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

using namespace std;
using ProcessImageDataFunc = std::function<void(std::vector<std::vector<float> >&, const std::vector<cv::Rect2d>&)>;

namespace py = pybind11;
class PyThreadStateLock
{
public:
    PyThreadStateLock()
    {
        state = PyGILState_Ensure();
    }

    ~PyThreadStateLock()
    {
        PyGILState_Release(state);
    }
private:
    PyGILState_STATE state;
};

namespace DETR
{

class Verbose
{
public:
    enum eLevel
    {
        VERBOSITY_QUIET=0,
        VERBOSITY_NORMAL=1,
        VERBOSITY_VERBOSE=2,
        VERBOSITY_VERY_VERBOSE=3,
        VERBOSITY_DEBUG=4
    };

    static eLevel th;

public:
    static void PrintMess(string str, eLevel lev)
    {
        if(lev <= th)
            cout << str << endl;
    }

    static void SetTh(eLevel _th)
    {
        th = _th;
    }
};

class System
{

public:

    // Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
    System(const string &strSettingsFile);

    // pybind stuff
    inline void InitThread()
    {
        if (!PyEval_ThreadsInitialized())
        {
            PyEval_InitThreads();
        }
    };
    
	/* finalise the interpreter */
	inline void finalize() {
		Py_DECREF(pInstance);
		Py_DECREF(pModule);
        py::finalize_interpreter();
    };
    py::object pyCfg;
    py::object pyDecoder;
    py::object pySequence;
    
    PyObject* pModule;
    PyObject* pClass;
    PyObject* pInstance;

	boost::asio::ip::tcp::socket* socket_;

	//double roll=0, pitch=0, yaw=0; // drone orientation
	void setSocket(const std::string& server, const std::string& port);
	void call_python_function(const cv::Mat& image);
	void call_python_function(const double& Time, const cv::Mat& image, 
							const std::vector<Point>& pointCloud, const Imu& imu);
	void call_python_function(const cv::Mat& image, const ProcessImageDataFunc& callback);

	/* process the sensor data */
	void processImageData (const ImageData& image);
	void processSyncedData(const CloudData& lidar, const ImageData& image, dataQueue& imuQueue);
  
	/* Utilities */
	void increment();
	void writeToPCD(const std::vector<Point>& input); 
	void writeToBIN(const std::vector<Point>& input);
	void writeToPNG(const cv::Mat& mat);
	void writeToNAV(const Imu& imu);
	py::array_t<float> vector3dToNumpyArray(const std::vector<Point>& input, const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
	py::array_t<unsigned char> cvMatToNumpyArray(const cv::Mat& mat);
	
	std::vector<std::vector<float> > numpyArrayToVector(PyObject* probas_obj);

	// Convert a NumPy array to a vector of OpenCV Rectangles
	std::vector<cv::Rect2d> numpyArrayToRect2dVector(PyObject* numpy_array);
   
};


}// namespace DETR

#endif // SYSTEM_H
