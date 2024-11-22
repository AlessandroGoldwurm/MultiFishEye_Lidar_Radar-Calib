#ifndef SENSOR_CAMERAFE_CALIBRATOR_HPP_
#define SENSOR_CAMERAFE_CALIBRATOR_HPP_

#include "nonlinear_optimizer.hpp"
#include <ceres/ceres.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <vector>

#include "logging.hpp"

class Sensor_cameraFE_calibrator {
    private:
        NonlinearOptimizer optimizer;

};

#endif // SENSOR_CAMERA_CALIBRATOR_HPP_
