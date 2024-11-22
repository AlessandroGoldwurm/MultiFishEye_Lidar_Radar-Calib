#ifndef CAMERAFE_CALIBRATOR_HPP_
#define CAMERAFE_CALIBRATOR_HPP_

#include "nonlinear_optimizer.hpp"
#include <ceres/ceres.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include <vector>

#include "logging.hpp"

class CameraFECalibrator {
public:
  void set_input(const std::vector<std::string> images_name,
                 const std::vector<cv::Mat> &vec_mat_,
                 const cv::Size &chessboard_size_,
                 const std::vector<std::vector<std::string>> &lidar_3d_pts);

  void get_result(cv::Mat &camera_matrix, cv::Mat &k,
                  const cv::Size &image_size, std::vector<cv::Mat> &rvecsMat,
                  std::vector<cv::Mat> &tvecsMat);

private:
  void make_board_points(const cv::Size &chessboard_size_);

  void refine_all(Eigen::Matrix3d &camera_matrix_, Eigen::VectorXd &k_,
                  std::vector<Eigen::MatrixXd> &vec_extrinsics_);
  void refine_lidar2camera(Eigen::Matrix3d &camera_matrix_, Eigen::VectorXd &k_,
                           std::vector<Eigen::MatrixXd> &vec_extrinsics_,
                           Eigen::Matrix<double, 3, 4> &initial_extrinsic,
                           std::vector<LidarPointPair> &lidar_point_pairs);

  void lidar_projection(const Eigen::Matrix3d &camera_intrinsic,
                        const Eigen::Matrix<double, 3, 4> &extrinsic,
                        const cv::Point3f &pt, cv::Point2f &img_pt);

  void lidar_projectionFE(const Eigen::Matrix3d &camera_intrinsic,
                          const std::vector<double>& distortion_coeffs,
                          const Eigen::Matrix<double, 3, 4> &extrinsic, const cv::Point3f &pt,
                          std::vector<cv::Point2f>& img_pt);

  void DrawCross(cv::Mat &img, cv::Point point);

  void equidistantProjectPoints(const std::vector<cv::Point3f>& cam_points, 
                                const Eigen::Matrix3d& camera_intrinsic,
                                const std::vector<double>& distortion_coeffs,
                                std::vector<cv::Point2f>& imgpoints_cir);

  void applyExtrinsicTransformation(const std::vector<cv::Point3f>& points_3d,
                                    const cv::Mat& rvec, const cv::Mat& tvec,
                                    std::vector<cv::Point3f>& cam_points);

private:
  bool _b_disp_corners = false;
  std::vector<std::vector<cv::Point2f>> _imgs_pts;
  std::vector<std::vector<cv::Point2f>> _boards_pts;
  std::vector<std::vector<cv::Point3f>> _boards_pts_3d;
  std::vector<std::vector<cv::Point3f>> _boards_pts_cir; // scl
  std::vector<std::vector<cv::Point3f>> lidar_3d_pts_;
  std::vector<std::vector<cv::Point2f>> _imgs_pts_cir2D_true; // scl
  std::vector<cv::Mat> available_imgs;

  NonlinearOptimizer optimier;
};

#endif // CAMERA_CALIBRATOR_HPP_
