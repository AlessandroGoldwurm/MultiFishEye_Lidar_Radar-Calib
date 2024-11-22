#ifndef NONLINEAR_OPTIMIZER_HPP_
#define NONLINEAR_OPTIMIZER_HPP_

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>

#include "params.hpp"
#include "reprojection_error.hpp"
#include "termination_checking.hpp"

class NonlinearOptimizer {
public:
  NonlinearOptimizer() {}
  ~NonlinearOptimizer() {}

  void refine_H(const std::vector<cv::Point2f> &img_pts_,
                const std::vector<cv::Point2f> &board_pts_,
                const Eigen::Matrix3d &matrix_H_, Eigen::Matrix3d &refined_H_);

  void refine_all_camera_params(
      const Params &params_,
      const std::vector<std::vector<cv::Point2f>> &imgs_pts_,
      const std::vector<std::vector<cv::Point2f>> &bords_pts_,
      Params &refined_params_);

  void refine_lidar2camera_params(
      const LidarParams &params_,
      const std::vector<std::vector<cv::Point2f>> &imgs_pts_,
      const std::vector<std::vector<cv::Point2f>> &bords_pts_,
      const std::vector<LidarPointPair> lidar_point_pairs,
      LidarParams &refined_params_);

  void refine_lidar2FEcamera_params(
      const LidarParams &params_,
      const std::vector<std::vector<cv::Point2f>> &imgs_pts_,
      const std::vector<std::vector<cv::Point2f>> &bords_pts_,
      const std::vector<LidarPointPair> lidar_point_pairs,
      LidarParams &refined_params_);

  void refine_sensor2camera_params(
      const SensorParams &params_,
      const std::vector<SensorCamPointPair> sensor2cam_point_pairs,
      SensorParams &refined_params_);

private:
  void formate_data(const Eigen::VectorXd &v_camera_matrix_,
                    const Eigen::VectorXd &v_dist_,
                    const std::vector<Eigen::VectorXd> &v_rt_, Params &params_);
  void formate_data(const Eigen::VectorXd &v_camera_matrix_,
                    const Eigen::VectorXd &v_dist_,
                    const std::vector<Eigen::VectorXd> &v_rt_,
                    const Eigen::VectorXd &RT, LidarParams &params_);
  void formate_data(const Eigen::VectorXd &v_camera_matrix_,
                    const Eigen::VectorXd &v_dist_,
                    const Eigen::VectorXd &RT, SensorParams &params_);
  // Cost functor which computes symmetric geometric distance
  // used for homography matrix refinement.
  struct HomographySymmetricGeometricCostFunctor {
    HomographySymmetricGeometricCostFunctor(const Eigen::Vector2d &x,
                                            const Eigen::Vector2d &y)
        : x_(x), y_(y) {}

    template <typename T>
    bool operator()(const T *homography_parameters, T *residuals) const {
      typedef Eigen::Matrix<T, 3, 3> Mat3;
      typedef Eigen::Matrix<T, 2, 1> Vec2;

      Mat3 H(homography_parameters);
      Vec2 x(T(x_(0)), T(x_(1)));
      Vec2 y(T(y_(0)), T(y_(1)));

      Utils::SymmetricGeometricDistanceTerms<T>(H, x, y, &residuals[0],
                                                &residuals[2]);
      return true;
    }

    const Eigen::Vector2d x_;
    const Eigen::Vector2d y_;
  };

  struct ReprojectionError {
    ReprojectionError(const Eigen::Vector2d &img_pts_,
                      const Eigen::Vector2d &board_pts_)
        : _img_pts(img_pts_), _board_pts(board_pts_) {}

    template <typename T>
    bool operator()(const T *const instrinsics_, const T *const k_,
                    const T *const rt_, // 6 : angle axis and translation
                    T *residuls) const {
      //	Eigen::Vector3d hom_w(_board_pts(0), _board_pts(1), T(1.));
      T hom_w_t[3];
      hom_w_t[0] = T(_board_pts(0));
      hom_w_t[1] = T(_board_pts(1));
      hom_w_t[2] = T(1.);
      T hom_w_trans[3];
      ceres::AngleAxisRotatePoint(rt_, hom_w_t, hom_w_trans);
      hom_w_trans[0] += rt_[3];
      hom_w_trans[1] += rt_[4];
      hom_w_trans[2] += rt_[5];

      T c_x = hom_w_trans[0] / hom_w_trans[2];
      T c_y = hom_w_trans[1] / hom_w_trans[2];

      // distortion
      T r2 = c_x * c_x + c_y * c_y;
      T r4 = r2 * r2;
      T r_coeff = (T(1) + k_[0] * r2 + k_[1] * r4);
      T xd = c_x * r_coeff;
      T yd = c_y * r_coeff;

      // camera coord => image coord
      T predict_x = instrinsics_[0] * xd + instrinsics_[2];
      T predict_y = instrinsics_[3] * yd + instrinsics_[4];

      // residus

      residuls[0] = _img_pts(0) - predict_x;
      residuls[1] = _img_pts(1) - predict_y;

      return true;
    }
    const Eigen::Vector2d _img_pts;
    const Eigen::Vector2d _board_pts;
  };

  struct ReprojectionError_FE {
    ReprojectionError_FE(const Eigen::Vector2d &img_pts_,
                      const Eigen::Vector2d &board_pts_)
        : _img_pts(img_pts_), _board_pts(board_pts_) {}

    template <typename T>
    bool operator()(const T *const instrinsics_, const T *const k_,
                    const T *const rt_, // 6 : angle axis and translation
                    T *residuls) const {
      //	Eigen::Vector3d hom_w(_board_pts(0), _board_pts(1), T(1.));
      T hom_w_t[3];
      hom_w_t[0] = T(_board_pts(0));
      hom_w_t[1] = T(_board_pts(1));
      hom_w_t[2] = T(1.);
      T hom_w_trans[3];
      ceres::AngleAxisRotatePoint(rt_, hom_w_t, hom_w_trans);
      hom_w_trans[0] += rt_[3];
      hom_w_trans[1] += rt_[4];
      hom_w_trans[2] += rt_[5];

      T x = hom_w_trans[0];
      T y = hom_w_trans[1];
      T z = hom_w_trans[2];

      T chi = sqrt(x*x+y*y);
      T theta = atan2(chi, z);
      T rho = instrinsics_[0] * (theta + 
                  k_[0] * pow(theta, 3) + 
                  k_[1] * pow(theta, 5) + 
                  k_[2] * pow(theta, 7) + 
                  k_[3] * pow(theta, 9));

      T cx = instrinsics_[2];
      T cy = instrinsics_[3];

      T predict_x, predict_y;
      if (chi != T(0)) {
        predict_x = (rho * x) / chi + cx;
        predict_y = (rho * y) / chi + cy;
      } else {
        predict_x = cx;
        predict_y = cy;
      }

      // residus

      residuls[0] = _img_pts(0) - predict_x;
      residuls[1] = _img_pts(1) - predict_y;

      return true;
    }
    const Eigen::Vector2d _img_pts;
    const Eigen::Vector2d _board_pts;

  };

  struct LidarReprojectionError {
    LidarReprojectionError(const Eigen::Vector2d &img_pts_,
                           const Eigen::Vector3d &lidar_pts_)
        : _img_pts(img_pts_), _lidar_pts(lidar_pts_) {}

    template <typename T>
    bool operator()(const T *const instrinsics_, const T *const k_,
                    const T *const rt_, // 6 : angle axis and translation
                    T *residuls) const {
      T hom_w_t[3];
      hom_w_t[0] = T(_lidar_pts(0));
      hom_w_t[1] = T(_lidar_pts(1));
      hom_w_t[2] = T(_lidar_pts(2));
      T hom_w_trans[3];
      ceres::AngleAxisRotatePoint(rt_, hom_w_t, hom_w_trans);
      hom_w_trans[0] += rt_[3];
      hom_w_trans[1] += rt_[4];
      hom_w_trans[2] += rt_[5];

      T c_x = hom_w_trans[0] / hom_w_trans[2];
      T c_y = hom_w_trans[1] / hom_w_trans[2];

      T xd = c_x;
      T yd = c_y;
      // distortion
      // T r2 = c_x * c_x + c_y * c_y;
      // T r4 = r2 * r2;
      // T r_coeff = (T(1) + k_[0] * r2 + k_[1] * r4);
      // T xd = c_x * r_coeff;
      // T yd = c_y * r_coeff;

      // // camera coord => image coord
      T predict_x = instrinsics_[0] * xd + instrinsics_[2];
      T predict_y = instrinsics_[3] * yd + instrinsics_[4];

      T scale_factor = (T)60.0;
      residuls[0] = (_img_pts(0) - predict_x) * scale_factor;
      residuls[1] = (_img_pts(1) - predict_y) * scale_factor;

      return true;
    }
    const Eigen::Vector2d _img_pts;
    const Eigen::Vector3d _lidar_pts;
  };

  struct LidarFisheyeReprojectionError {
    LidarFisheyeReprojectionError(const Eigen::Vector2d &img_pts_,
                           const Eigen::Vector3d &lidar_pts_)
        : _img_pts(img_pts_), _lidar_pts(lidar_pts_){}
    template <typename T>
    bool operator()(const T *const instrinsics_, const T *const k_,
                    const T *const rt_, // 6 : angle axis and translation
                    T *residuls) const {

    T hom_w_t[3];
    hom_w_t[0] = T(_lidar_pts(0));
    hom_w_t[1] = T(_lidar_pts(1));
    hom_w_t[2] = T(_lidar_pts(2));
    T hom_w_trans[3];
    ceres::AngleAxisRotatePoint(rt_, hom_w_t, hom_w_trans);
    hom_w_trans[0] += rt_[3];
    hom_w_trans[1] += rt_[4];
    hom_w_trans[2] += rt_[5];

    T x = hom_w_trans[0];
    T y = hom_w_trans[1];
    T z = hom_w_trans[2];

    T chi = sqrt(x*x+y*y);
    T theta = atan2(chi, z);
    T rho = instrinsics_[0] * (theta + 
                k_[0] * pow(theta, 3) + 
                k_[1] * pow(theta, 5) + 
                k_[2] * pow(theta, 7) + 
                k_[3] * pow(theta, 9));

    T cx = instrinsics_[2];
    T cy = instrinsics_[3];

    T predict_x, predict_y;
    if (chi != T(0)) {
      predict_x = (rho * x) / chi + cx;
      predict_y = (rho * y) / chi + cy;
    } else {
      predict_x = cx;
      predict_y = cy;
    }
    // Étape 6: Calcul des résidus
    T scale_factor = T(60.0);
    residuls[0] = (_img_pts(0) - predict_x) * scale_factor;
    residuls[1] = (_img_pts(1) - predict_y) * scale_factor;

    return true;
    }
    const Eigen::Vector2d _img_pts;
    const Eigen::Vector3d _lidar_pts;
  };
  
  struct SensorFisheyeReprojectionError {
    SensorFisheyeReprojectionError(const Eigen::VectorXd &v_camera_matrix_, const Eigen::VectorXd &v_dist_, const Eigen::Vector2d &img_pts_,
                           const Eigen::Vector3d &sensor_pts_)
        : _v_camera_matrix(v_camera_matrix_), _v_dist(v_dist_), _img_pts(img_pts_), _3d_pts_(sensor_pts_){}
    template <typename T>
    bool operator()(const T *const rt_, // 6 : angle axis and translation
                    T *residuls) const {

    T hom_w_t[3];
    hom_w_t[0] = T(_3d_pts_(0));
    hom_w_t[1] = T(_3d_pts_(1));
    hom_w_t[2] = T(_3d_pts_(2));
    T hom_w_trans[3];
    ceres::AngleAxisRotatePoint(rt_, hom_w_t, hom_w_trans);
    hom_w_trans[0] += rt_[3];
    hom_w_trans[1] += rt_[4];
    hom_w_trans[2] += rt_[5];

    T x = hom_w_trans[0];
    T y = hom_w_trans[1];
    T z = hom_w_trans[2];

    T chi = sqrt(x*x+y*y);
    T theta = atan2(chi, z);

    /*T rho = instrinsics_[0] * (theta + 
                k_[0] * pow(theta, 3) + 
                k_[1] * pow(theta, 5) + 
                k_[2] * pow(theta, 7) + 
                k_[3] * pow(theta, 9));*/

    T rho = T(_v_camera_matrix(0))* (theta + 
                T(_v_dist(0)) * pow(theta, 3) + 
                T(_v_dist(1)) * pow(theta, 5) + 
                T(_v_dist(2)) * pow(theta, 7) + 
                T(_v_dist(3)) * pow(theta, 9));

    T cx = T(_v_camera_matrix(2));
    T cy = T(_v_camera_matrix(4));

    T predict_x, predict_y;
    if (chi != T(0)) {
      predict_x = (rho * x) / chi + cx;
      predict_y = (rho * y) / chi + cy;
    } else {
      predict_x = cx;
      predict_y = cy;
    }
    // Étape 6: Calcul des résidus
    T scale_factor = T(60.0);
    residuls[0] = (_img_pts(0) - predict_x) * scale_factor;
    residuls[1] = (_img_pts(1) - predict_y) * scale_factor;

    return true;
    }

    const Eigen::VectorXd _v_camera_matrix;
    const Eigen::VectorXd _v_dist;
    const Eigen::Vector2d _img_pts;
    const Eigen::Vector3d _3d_pts_;
  };

};



#endif // NONLINEAR_OPTIMIZER_HPP_
