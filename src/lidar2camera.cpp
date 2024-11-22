/*
 * Copyright (C) 2022 by Autonomous Driving Group, Shanghai AI Laboratory
 * Limited. All rights reserved.
 * Yan Guohang <yanguohang@pjlab.org.cn>
 */

#include <fstream>
#include <iostream>
#include <sstream>

//#include "camera_calibrator.hpp"
//#include "cameraFE_calibrator.hpp"
#include "nonlinear_optimizer.cpp"

std::vector<SensorCamPointPair> readSensorCamPoints(const std::string& cam_file, const std::string& sensor_file) {
    std::vector<SensorCamPointPair> point_pairs;

    std::ifstream cam_file_stream(cam_file);
    std::ifstream sensor_file_stream(sensor_file);

    if (!cam_file_stream.is_open() || !sensor_file_stream.is_open()) {
        std::cerr << "Error: Unable to open input files." << std::endl;
        return point_pairs;
    }

    std::string cam_line, sensor_line;
    while (std::getline(cam_file_stream, cam_line) && std::getline(sensor_file_stream, sensor_line)) {
        std::istringstream cam_stream(cam_line);
        std::istringstream sensor_stream(sensor_line);

        float cam_x, cam_y;
        float sensor_x_mm, sensor_y_mm, sensor_z_mm;

        // Lire les points 2D (caméra)
        cam_stream >> cam_x >> cam_y;

        // Lire les points 3D (radar/sensor)
        sensor_stream >> sensor_x_mm >> sensor_y_mm >> sensor_z_mm;
        float sensor_x_m = sensor_x_mm / 100.0f;
        float sensor_y_m = sensor_y_mm / 100.0f;
        float sensor_z_m = sensor_z_mm / 100.0f;
        
        // Créer et ajouter un point pair
        SensorCamPointPair pair;
        pair.cam2d_point = cv::Point2f(cam_x, cam_y);
        pair.sensor3d_point = cv::Point3f(sensor_x_m, sensor_y_m, sensor_z_m);

        point_pairs.push_back(pair);
    }

    cam_file_stream.close();
    sensor_file_stream.close();

    return point_pairs;
}

void printSensorCamPoints(const std::vector<SensorCamPointPair>& point_pairs) {
    for (size_t i = 0; i < point_pairs.size(); ++i) {
        const auto& pair = point_pairs[i];
        std::cout << "Point Pair " << i + 1 << ":" << std::endl;
        std::cout << "  Camera 2D Point: (" << pair.cam2d_point.x << ", " << pair.cam2d_point.y << ")" << std::endl;
        std::cout << "  Sensor 3D Point: (" << pair.sensor3d_point.x << ", " << pair.sensor3d_point.y << ", " << pair.sensor3d_point.z << ")" << std::endl;
    }
}

bool readCameraParamsFromTxt(const std::string& calib_file, Eigen::Matrix3d& intrinsic, Eigen::VectorXd& distortion) {
    std::ifstream file(calib_file);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open calibration file: " << calib_file << std::endl;
        return false;
    }

    std::vector<double> values;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream stream(line);
        double value;
        while (stream >> value) {
            values.push_back(value);
        }
    }

    file.close();

    // Vérifier que le fichier contient suffisamment de valeurs
    if (values.size() < 12) {
        std::cerr << "Error: Calibration file does not contain enough data." << std::endl;
        return false;
    }

    // Remplir la matrice intrinsèque
    intrinsic << values[0], values[1], values[2],
                 values[3], values[4], values[5],
                 values[6], values[7], values[8];

    // Remplir le vecteur de distorsion
    distortion = Eigen::VectorXd(4);
    for (int i = 0; i < 4; ++i) {
        distortion(i) = values[9 + i];
    }

    return true;
}

int main(int argc, char **argv) {
    if (argc != 4) {
    std::cout << "Usage: ./main camera_txt_file sensor_txt_file camera_calib_txt_file"
                 "\nexample:\n\t"
                 "./bin/lidar2camera data/Radar_FE_point_pairs/cam_pts.txt data/Radar_FE_point_pairs/radar_pts.txt data/Radar_FE_point_pairs/calib_cam1.txt"
              << std::endl;
    return 0;
    }
  
    std::cout << "OpenCV version : " << CV_VERSION << std::endl;

    std::cout << "Start camera to sensor calibrator" << std::endl;

    const std::string cam_file = argv[1];
    const std::string sensor_file = argv[2];
    const std::string calib_file = argv[3];

    // Lire les points des fichiers
    std::vector<SensorCamPointPair> point_pairs = readSensorCamPoints(cam_file, sensor_file);

    if (point_pairs.empty()) {
        std::cerr << "No points read from the input files. Exiting." << std::endl;
        return 1;
    }

    printSensorCamPoints(point_pairs);

    //Calib
    Eigen::Matrix3d camera_intrinsic;
    Eigen::VectorXd distortion;

    if (!readCameraParamsFromTxt(calib_file, camera_intrinsic, distortion)) {
        std::cerr << "Failed to read camera parameters. Exiting." << std::endl;
        return 1;
    }

    // Afficher les paramètres de la caméra
    std::cout << "Camera Intrinsic Matrix:\n" << camera_intrinsic << std::endl;
    std::cout << "Camera Distortion Coefficients:\n" << distortion.transpose() << std::endl;

    // an inaccurate initial Lidar-camera extrinsic
    Eigen::Matrix<double, 3, 4> initial_extrinsic;
    initial_extrinsic << -0.0000667338, -0.9999999780, 0.0,
        -0.0010031200, -0.0000409491, 0.0001990681, 0.9999999793, 0.0,
        -0.9999999969, 0.0000667257, -0.0000409624, 0.0;

    initial_extrinsic << 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0;


    Eigen::VectorXd distort(2);
    distort << distortion(0, 0), distortion(1, 0);

    //std::cout << "Camera Distort Coefficients:\n" << distort << std::endl;

    SensorParams params, params_refined;
    params.camera_matrix = camera_intrinsic;
    params.k = distortion;
    params.extrinsic = initial_extrinsic;

    //Initialiser optimiseur
    NonlinearOptimizer optimizer;

    // Lancer l'optimisation
    std::cout << "Starting optimization..." << std::endl;
    optimizer.refine_sensor2camera_params(params, point_pairs, params_refined);

        // Afficher les paramètres optimisés
    std::cout << "Optimized Camera Intrinsic Matrix:\n" << params_refined.camera_matrix << std::endl;
    std::cout << "Optimized Distortion Coefficients:\n" << params_refined.k.transpose() << std::endl;
    std::cout << "Optimized Extrinsic Parameters:\n" << params_refined.extrinsic << std::endl;
    
    return 0;
}

/*int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Usage: ./main camera_dir csv file"
                 "\nexample:\n\t"
                 "./bin/lidar2camera data/intrinsic/ data/circle.csv"
              << std::endl;
    return 0;
  }

  std::cout << "Start camera calibrator" << std::endl;

  std::string image_dir = argv[1];
  std::string csv_file = argv[2];
  std::cout << csv_file << std::endl;
  std::ifstream fin(csv_file);
  std::string line;
  bool is_first = true;
  std::vector<std::vector<std::string>> lidar_3d_pts;
  while (getline(fin, line)) {
    if (is_first) {
      is_first = false;
      continue;
    }

    std::istringstream sin(line);
    std::vector<std::string> fields;
    std::string field;
    while (getline(sin, field, ',')) {
      fields.push_back(field);
    }
    lidar_3d_pts.push_back(fields);
  }

  std::cout << image_dir << std::endl;
  std::vector<cv::String> images;
  cv::glob(image_dir, images);
  std::vector<cv::Mat> vec_mat;
  std::vector<std::string> images_name;
  for (const auto &path : images) {
    std::cout << path << std::endl;
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    vec_mat.push_back(img);
    images_name.push_back(path);
  }

  //CameraCalibrator m;
  CameraFECalibrator m;
  cv::Mat camera_matrix = cv::Mat(3, 3, CV_32FC1, cv::Scalar::all(0));
  //cv::Mat k = cv::Mat(1, 5, CV_32FC1, cv::Scalar::all(0));
  //For fisheye
  cv::Mat k = cv::Mat(1, 4, CV_32FC1, cv::Scalar::all(0));
  std::vector<cv::Mat> tvecsMat;
  std::vector<cv::Mat> rvecsMat;
  
  //m.set_input(images_name, vec_mat, cv::Size{17, 7}, lidar_3d_pts);
  //m.set_input(images_name, vec_mat, cv::Size{8, 6}, lidar_3d_pts);
  //m.set_input(images_name, vec_mat, cv::Size{11,9}, lidar_3d_pts);
  m.set_input(images_name, vec_mat, cv::Size{9,9}, lidar_3d_pts);

  //m.get_result(camera_matrix, k, cv::Size{1920, 1200}, rvecsMat, tvecsMat);
  m.get_result(camera_matrix, k, cv::Size{1936, 1552}, rvecsMat, tvecsMat);
  return 0;
}*/