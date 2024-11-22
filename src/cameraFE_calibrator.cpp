#include "cameraFE_calibrator.hpp"
#include <iomanip>

void CameraFECalibrator::set_input(
    const std::vector<std::string> images_name,
    const std::vector<cv::Mat> &vec_mat_, const cv::Size &chessboard_size_,
    const std::vector<std::vector<std::string>> &lidar_3d_pts) {
  // lidar
  std::vector<std::vector<cv::Point3f>> pts;
  for (auto src : lidar_3d_pts) {
    std::vector<cv::Point3f> pt;
    cv::Point3f left_top(std::stod(src[0]), std::stod(src[1]),
                         std::stod(src[2]));
    cv::Point3f right_top(std::stod(src[3]), std::stod(src[4]),
                          std::stod(src[5]));
    cv::Point3f left_bottom(std::stod(src[6]), std::stod(src[7]),
                            std::stod(src[8]));
    cv::Point3f right_bottom(std::stod(src[9]), std::stod(src[10]),
                             std::stod(src[11]));
    pt.push_back(left_top);
    pt.push_back(right_top);
    pt.push_back(left_bottom);
    pt.push_back(right_bottom);
    //On ajoute dans pts (en str) les coordonnées des centres des 4 cercles
    pts.push_back(pt);
  }

  std::cout << "start" << std::endl;
  _boards_pts.clear();
  _imgs_pts.clear();
  int i = 0;
  for (const auto &img : vec_mat_) {
    CHECK(1 == img.channels()) << "images must be gray";
    std::vector<cv::Point2f> corner_pts;
    int found = cv::findChessboardCorners(img, chessboard_size_, corner_pts,
                                          cv::CALIB_CB_ADAPTIVE_THRESH |
                                              cv::CALIB_CB_FAST_CHECK |
                                              cv::CALIB_CB_NORMALIZE_IMAGE);
    if (!found) {
      continue;
    }   
    cv::Mat img_copy = img.clone();
    available_imgs.push_back(img_copy);
    //Stockage dans lidar_3d_pts_ des centres des cercles pour les images dont on a détecté les coins 
    lidar_3d_pts_.push_back(pts[i]);
    std::cout << images_name[i] << std::endl;
    i++;

    //Affinement coins détectés
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);
    cv::cornerSubPix(img, corner_pts, chessboard_size_, cv::Size(-1, -1),
                     criteria);
    //Stockage dans _imgs_pts des coins du chessboard
    _imgs_pts.push_back(corner_pts);
    this->make_board_points(chessboard_size_);

    cv::drawChessboardCorners(img_copy, chessboard_size_, corner_pts, found);
    cv::namedWindow("Chessboard Corners", cv::WINDOW_NORMAL);  // Permet de redimensionner la fenêtre
    cv::resizeWindow("Chessboard Corners", 800, 600); 
    cv::imshow("Chessboard Corners", img_copy);
    cv::waitKey(0);  // Attend une touche pour passer à l'image suivante
  }
}

void CameraFECalibrator::get_result(cv::Mat &camera_matrix, cv::Mat &k,
                                  const cv::Size &image_size,
                                  std::vector<cv::Mat> &rvecsMat,
                                  std::vector<cv::Mat> &tvecsMat) {

  std::cout<<"start calib"<<std::endl;
  //Camera calibration, using board 3D and 2D points (_boards_pts_3d and _imgs_pts) -> Use fishEye calibration for fisheyes ?
  int calibration_flags = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC + cv::fisheye::CALIB_FIX_SKEW;
  double eps_criteria = 1e-6;
  int max_iter = 100;
  cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, max_iter, eps_criteria);
  double re_error =
    //   cv::calibrateCamera(_boards_pts_3d, _imgs_pts, image_size, camera_matrix,
    //                       k, rvecsMat, tvecsMat, cv::CALIB_FIX_PRINCIPAL_POINT);
        cv::fisheye::calibrate(_boards_pts_3d, _imgs_pts, image_size, camera_matrix,
                            k, rvecsMat, tvecsMat, calibration_flags, criteria);

                          
  std::cout << "first calibration" << std::endl;
  std::cout << "reprojection is " << re_error << std::endl;

  std::cout << "Camera Matrix (Intrinsic): " << std::endl;
  std::cout << camera_matrix << std::endl;

  std::cout << "Distortion Coefficients: " << std::endl;
  std::cout << k << std::endl;

  Eigen::Matrix3d camera_intrinsic;      
  camera_intrinsic << camera_matrix.at<float>(0, 0),
      camera_matrix.at<float>(0, 1), camera_matrix.at<float>(0, 2),
      camera_matrix.at<float>(1, 0), camera_matrix.at<float>(1, 1),
      camera_matrix.at<float>(1, 2), camera_matrix.at<float>(2, 0),
      camera_matrix.at<float>(2, 1), camera_matrix.at<float>(2, 2); 

  Eigen::VectorXd distort(2);
  distort << k.at<double>(0, 0), k.at<double>(0, 1);

  std::vector<Eigen::MatrixXd> vec_extrinsics;

  for (size_t i = 0; i < rvecsMat.size(); i++) {
    std::cout << " image " << i+1 << ":" << std::endl; 
    cv::Mat rvec = rvecsMat[i];
    cv::Mat tvec = tvecsMat[i];

    cv::Mat rot;
    cv::Rodrigues(rvec, rot);

    // Parcourir les 4 points LiDAR correspondants à chaque image
    std::cout << "Points 3D des centres des cercles vue Lidar 3D " << i+1 << ":" << std::endl; 
    for (size_t j = 0; j < lidar_3d_pts_[i].size(); ++j) {
      const cv::Point3f& pt = lidar_3d_pts_[i][j];
      std::cout << "Point " << j + 1 << ": (" 
                << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
    }

    std::cout << "Points 3D des centres de cercles (_boards_pts_cir) vue de l'image " << i+1 << ":" << std::endl;
    for (const auto& point3D : _boards_pts_cir[i]) {
      std::cout << "(" << point3D.x << ", " << point3D.y << ", " << point3D.z << ")" << std::endl;
    }

    std::vector<cv::Point3f> cam_points;
    std::vector<cv::Point2f> imgpoints_cir;
    applyExtrinsicTransformation(_boards_pts_cir[i], rvecsMat[i], tvecsMat[i], cam_points);
    equidistantProjectPoints(cam_points, camera_intrinsic, k, imgpoints_cir);

    // cv::fisheye::projectPoints(_boards_pts_cir[i], rvecsMat[i], tvecsMat[i],
    //                  camera_matrix, k, imgpoints_cir, 0);

    // Clone de l'image pour ne pas modifier l'originale
    cv::Mat img_copy = available_imgs[i].clone();
    std::cout << "Points projetés 2D (imgpoints_cir) des centres de cerlces, Simulés, pour l'image " << i+1 << ":" << std::endl;
    for (const auto& point2D : imgpoints_cir) {
      std::cout << "(" << point2D.x << ", " << point2D.y << ")" << std::endl;
      cv::circle(img_copy, point2D, 5, cv::Scalar(0, 0, 255), -1); // Cercles rouges de rayon 5
    }
    // // Afficher l'image avec les points
    // cv::imshow("Image avec points", img_copy);
    // // Attendre que l'utilisateur appuie sur une touche pour fermer la fenêtre
    // cv::waitKey(0);
    // // Fermer la fenêtre après appui sur une touche
    // cv::destroyAllWindows();

    // std::vector<cv::Point3f> test_img_board_3d;
    // std::vector<cv::Point2f> test_img_board;
    // applyExtrinsicTransformation(_boards_pts_3d[i], rvecsMat[i], tvecsMat[i], test_img_board_3d);
    // equidistantProjectPoints(test_img_board_3d, camera_intrinsic, k, focal, test_img_board);

    // img_copy = available_imgs[i].clone();
    // for (const auto& point2D : test_img_board) {
    //   cv::circle(img_copy, point2D, 5, cv::Scalar(0, 0, 255), -1); // Cercles rouges de rayon 5
    // }
    // cv::imshow("Image avec points", img_copy);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    // double total_error = 0.0;
    // std::vector<cv::Point2f> detected_points = _imgs_pts[i]; // Points détectés (coins de l'image)
    // // Calculer la distance euclidienne entre chaque paire de points
    // for (size_t j = 0; j < test_img_board.size(); j++) {
    //     double error = cv::norm(test_img_board[j] -  detected_points[j]);
    //     total_error += error;
    // }
    // double mean_error = total_error / static_cast<double>(test_img_board.size());
    // std::cout << "Erreur de reprojection pour l'image " << i + 1 << " : " << mean_error << std::endl;


    // size_t y_min = imgpoints_cir[0].y;
    // size_t y_max = imgpoints_cir[0].y;
    // size_t x_min = imgpoints_cir[0].x;
    // size_t x_max = imgpoints_cir[0].x;
    // for (size_t i = 1; i < 4; i++) {
    //   if (imgpoints_cir[i].y > y_max)
    //     y_max = imgpoints_cir[i].y;
    //   if (imgpoints_cir[i].y < y_min)
    //     y_min = imgpoints_cir[i].y;
    //   if (imgpoints_cir[i].x > x_max)
    //     x_max = imgpoints_cir[i].x;
    //   if (imgpoints_cir[i].x < x_min)
    //     x_min = imgpoints_cir[i].x;
    // }

    // std::vector<cv::Point2f> imgpoints_cir1; // scl
    // for (size_t i = 0; i < 4; i++) {
    //   if ((imgpoints_cir[i].x - x_min) <= 50 &&
    //       (imgpoints_cir[i].y - y_min) <= 50)
    //     imgpoints_cir1.push_back(imgpoints_cir[i]);
    // }
    // for (size_t i = 0; i < 4; i++) {
    //   if ((x_max - imgpoints_cir[i].x) <= 50 &&
    //       (imgpoints_cir[i].y - y_min) <= 50)
    //     imgpoints_cir1.push_back(imgpoints_cir[i]);
    // }
    // for (size_t i = 0; i < 4; i++) {
    //   if ((imgpoints_cir[i].x - x_min) <= 50 &&
    //       (y_max - imgpoints_cir[i].y) <= 50)
    //     imgpoints_cir1.push_back(imgpoints_cir[i]);
    // }
    // for (size_t i = 0; i < 4; i++) {
    //   if ((x_max - imgpoints_cir[i].x) <= 50 &&
    //       (y_max - imgpoints_cir[i].y) <= 50)
    //     imgpoints_cir1.push_back(imgpoints_cir[i]);
    // }
    // if (imgpoints_cir1.size() != 4) {
    //   std::cout << "imgpoints_cir1.size() must = 4" << std::endl;
    //   return;
    // }

    // for (const auto& point2D : imgpoints_cir1) {
    //   std::cout << "(" << point2D.x << ", " << point2D.y << ")" << std::endl;
    // }
    //_imgs_pts_cir2D_true.push_back(imgpoints_cir1);
    _imgs_pts_cir2D_true.push_back(imgpoints_cir);

    std::cout << "Points sélectionnés dans _imgs_pts_cir2D_true pour l'image " << i << ":" << std::endl;
    for (const auto& point2D : _imgs_pts_cir2D_true[i]) {
      std::cout << "(" << point2D.x << ", " << point2D.y << ")" << std::endl;
    }
    // // Clone de l'image pour ne pas modifier l'originale
    // cv::Mat img_copy = available_imgs[i].clone();
    // // Parcours des points et ajout de cercles rouges
    // for (const auto &point : _imgs_pts_cir2D_true[i]) {
    //     cv::circle(img_copy, point, 5, cv::Scalar(0, 0, 255), -1); // Cercles rouges de rayon 5
    // }
    // // Afficher l'image avec les points
    // cv::imshow("Image avec points", img_copy);
    // // Attendre que l'utilisateur appuie sur une touche pour fermer la fenêtre
    // cv::waitKey(0);
    // // Fermer la fenêtre après appui sur une touche
    // cv::destroyAllWindows();

    std::cout << "--------------------------" << std::endl;
    
    Eigen::Vector3d r0, r1, r2, t;
    r0 << rot.at<double>(0, 0), rot.at<double>(1, 0), rot.at<double>(2, 0);
    r1 << rot.at<double>(0, 1), rot.at<double>(1, 1), rot.at<double>(2, 1);
    r2 << rot.at<double>(0, 2), rot.at<double>(1, 2), rot.at<double>(2, 2);
    t << tvec.at<double>(0, 0), tvec.at<double>(0, 1), tvec.at<double>(0, 2);

    Eigen::MatrixXd RT(3, 4);
    RT.block<3, 1>(0, 0) = r0;
    RT.block<3, 1>(0, 1) = r1;
    RT.block<3, 1>(0, 2) = r2;
    RT.block<3, 1>(0, 3) = t;
    vec_extrinsics.push_back(RT);
  }

  // // Refining calib intrinsic params with optimisation using points of Board -> Modify for FishEyes ?
  // this->refine_all(camera_intrinsic, distort, vec_extrinsics);

  // std::cout << "Matrice intrinsèque affinée :\n" << camera_intrinsic << std::endl;
  // std::cout << "Coefficients de distorsion affinés :\n" << distort.transpose() << std::endl;
  // for (size_t i = 0; i < vec_extrinsics.size(); ++i) {
  //     std::cout << "Extrinsèques affinés pour l'image " << i << " :\n" << vec_extrinsics[i] << std::endl;
  // }


  // Construction of pairs of Circle center points (3D Lidar space points, and known circle centers points from 2D Image )
  std::vector<LidarPointPair> lidar_point_pairs;
  for (size_t i = 0; i < _imgs_pts_cir2D_true.size(); i++) {
    cv::Mat img = available_imgs[i].clone();
    cv::Mat undistort_img = img;
    std::vector<cv::Point2f> left_top_src, right_top_src, left_bottom_src,
        right_bottom_src;

    LidarPointPair lidar_point_pair;
    lidar_point_pair.img_index = i;
    // left top
    lidar_point_pair.lidar_2d_point[0] = _imgs_pts_cir2D_true[i][0];
    lidar_point_pair.lidar_3d_point[0] = lidar_3d_pts_[i][0];
    // right top
    lidar_point_pair.lidar_2d_point[1] = _imgs_pts_cir2D_true[i][1];
    lidar_point_pair.lidar_3d_point[1] = lidar_3d_pts_[i][1];

    // left bottom;
    lidar_point_pair.lidar_2d_point[2] = _imgs_pts_cir2D_true[i][2];
    lidar_point_pair.lidar_3d_point[2] = lidar_3d_pts_[i][3];

    // right bottom
    lidar_point_pair.lidar_2d_point[3] = _imgs_pts_cir2D_true[i][3];
    lidar_point_pair.lidar_3d_point[3] = lidar_3d_pts_[i][2];

    if (false) { // Show real LiDAR pixels
      DrawCross(undistort_img, lidar_point_pair.lidar_2d_point[0]);
      DrawCross(undistort_img, lidar_point_pair.lidar_2d_point[1]);
      DrawCross(undistort_img, lidar_point_pair.lidar_2d_point[2]);
      DrawCross(undistort_img, lidar_point_pair.lidar_2d_point[3]);
      std::string save_name = "original" + std::to_string(i) + ".png";
      cv::imwrite(save_name, undistort_img);
    }

    // ??
    if (i > 20) // no corresponding circle center for more than 20
    {
      continue;
    }

    lidar_point_pairs.push_back(lidar_point_pair);
  }

  // an inaccurate initial Lidar-camera extrinsic
  Eigen::Matrix<double, 3, 4> initial_extrinsic;
  initial_extrinsic << -0.0000667338, -0.9999999780, 0.0001990654,
      -0.0010031200, -0.0000409491, 0.0001990681, 0.9999999793, 0.5912607639,
      -0.9999999969, 0.0000667257, -0.0000409624, 2.5079706347;

  initial_extrinsic << 0.0, -1.0, 0.0,
  0.0, 0.0, 0.0, 1.0, 1.0,
  -1.0, 0.0, 0.0, 2.50;
  
  std::cout << "Matrice extrinsèque initialisée :\n" << initial_extrinsic << std::endl;

  cv::Mat img = available_imgs[1].clone();
  cv::Mat undistort_img = img;
  std::vector<cv::Point2f> img_pt;
  lidar_projectionFE(camera_intrinsic, k, initial_extrinsic,
                    lidar_point_pairs[1].lidar_3d_point[0], img_pt);
  cv::circle(undistort_img, img_pt[0], 8, (0, 255, 0), 8);

  lidar_projectionFE(camera_intrinsic, k, initial_extrinsic, 
                    lidar_point_pairs[1].lidar_3d_point[1], img_pt);
  cv::circle(undistort_img, img_pt[0], 8, (0, 255, 0), 8);

  lidar_projectionFE(camera_intrinsic, k, initial_extrinsic,
                    lidar_point_pairs[1].lidar_3d_point[2], img_pt);
  cv::circle(undistort_img, img_pt[0], 8, (0, 255, 0), 8);

  lidar_projectionFE(camera_intrinsic, k, initial_extrinsic,
                    lidar_point_pairs[1].lidar_3d_point[3], img_pt);
  cv::circle(undistort_img, img_pt[0], 8, (0, 255, 0), 8);
  
  DrawCross(undistort_img, lidar_point_pairs[1].lidar_2d_point[0]);
  DrawCross(undistort_img, lidar_point_pairs[1].lidar_2d_point[1]);
  DrawCross(undistort_img, lidar_point_pairs[1].lidar_2d_point[2]);
  DrawCross(undistort_img, lidar_point_pairs[1].lidar_2d_point[3]);
  
  // Afficher l'image avec les points
  cv::namedWindow("Image avec points", cv::WINDOW_NORMAL);  // Permet de redimensionner la fenêtre
  cv::resizeWindow("Image avec points", 800, 600); 
  cv::imshow("Image avec points", undistort_img);
  // Attendre que l'utilisateur appuie sur une touche pour fermer la fenêtre
  cv::waitKey(0);
  // Fermer la fenêtre après appui sur une touche
  cv::destroyAllWindows();

  std::exit(0);

  // Ici optimisation des paramètres intrinsèques/extrinsèques et extrinsèques Cam2Lidar avec points 3D Lidar/2D Image des centres des cercles et points du chessboard
  // À modifier pour le cas Fisheye
  this->refine_lidar2camera(camera_intrinsic, distort, vec_extrinsics,
                            initial_extrinsic, lidar_point_pairs);

  double lidar_reprojection_error = 0;
  int number = 0;
  if (true) // Show the optimized projection effect
  {
    for (size_t i = 0; i < lidar_point_pairs.size(); i++) {
      int img_index = lidar_point_pairs[i].img_index;
      cv::Mat img = available_imgs[img_index].clone();
      cv::Mat undistort_img = img;
      // image_undistort(img, undistort_img, camera_intrinsic, distort);
      cv::Point2f img_pt;
      lidar_projection(camera_intrinsic, initial_extrinsic,
                       lidar_point_pairs[i].lidar_3d_point[0], img_pt);
      cv::circle(undistort_img, img_pt, 8, (0, 255, 0), 8);

      cv::Point2f pt = lidar_point_pairs[i].lidar_2d_point[0];
      double error1 = std::sqrt((img_pt.x - pt.x) * (img_pt.x - pt.x) +
                                (img_pt.y - pt.y) * (img_pt.y - pt.y));

      lidar_projection(camera_intrinsic, initial_extrinsic,
                       lidar_point_pairs[i].lidar_3d_point[1], img_pt);
      cv::circle(undistort_img, img_pt, 8, (0, 255, 0), 8);

      pt = lidar_point_pairs[i].lidar_2d_point[1];
      double error2 = std::sqrt((img_pt.x - pt.x) * (img_pt.x - pt.x) +
                                (img_pt.y - pt.y) * (img_pt.y - pt.y));

      lidar_projection(camera_intrinsic, initial_extrinsic,
                       lidar_point_pairs[i].lidar_3d_point[2], img_pt);
      cv::circle(undistort_img, img_pt, 8, (0, 255, 0), 8);

      pt = lidar_point_pairs[i].lidar_2d_point[2];
      double error3 = std::sqrt((img_pt.x - pt.x) * (img_pt.x - pt.x) +
                                (img_pt.y - pt.y) * (img_pt.y - pt.y));

      lidar_projection(camera_intrinsic, initial_extrinsic,
                       lidar_point_pairs[i].lidar_3d_point[3], img_pt);
      cv::circle(undistort_img, img_pt, 8, (0, 255, 0), 8);

      pt = lidar_point_pairs[i].lidar_2d_point[3];
      double error4 = std::sqrt((img_pt.x - pt.x) * (img_pt.x - pt.x) +
                                (img_pt.y - pt.y) * (img_pt.y - pt.y));
      //
      DrawCross(undistort_img, lidar_point_pairs[i].lidar_2d_point[0]);
      DrawCross(undistort_img, lidar_point_pairs[i].lidar_2d_point[1]);
      DrawCross(undistort_img, lidar_point_pairs[i].lidar_2d_point[2]);
      DrawCross(undistort_img, lidar_point_pairs[i].lidar_2d_point[3]);
      
      // Afficher l'image avec les points
      cv::imshow("Image avec points", undistort_img);
      // Attendre que l'utilisateur appuie sur une touche pour fermer la fenêtre
      cv::waitKey(0);
      // Fermer la fenêtre après appui sur une touche
      cv::destroyAllWindows();

      std::string save_name = "refine" + std::to_string(i) + ".png";
      cv::imwrite(save_name, undistort_img);
      double error = (error1 + error2 + error3 + error4) / 4;
      lidar_reprojection_error += error;
      number++;
    }
  }
  std::cout << "lidar reprojection error: " << lidar_reprojection_error / number
            << std::endl;
}

void CameraFECalibrator::DrawCross(cv::Mat &img, cv::Point point) {
  double cx = point.x;
  double cy = point.y;
  double len = 10;
  cv::line(img, cv::Point(cx - len, cy), cv::Point(cx + len, cy), (0, 255, 255),
           3);
  cv::line(img, cv::Point(cx, cy - len), cv::Point(cx, cy + len), (0, 255, 255),
           3);
}

void CameraFECalibrator::lidar_projection(
    const Eigen::Matrix3d &camera_intrinsic,
    const Eigen::Matrix<double, 3, 4> &extrinsic, const cv::Point3f &pt,
    cv::Point2f &img_pt) {
  Eigen::Matrix<double, 4, 1> lidar_point;
  lidar_point << pt.x, pt.y, pt.z, 1.0;
  // Eigen::Matrix<float, 3, 1> pro_pt;
  auto pro_pt = camera_intrinsic * extrinsic * lidar_point;
  img_pt.x = pro_pt(0) / pro_pt(2);
  img_pt.y = pro_pt(1) / pro_pt(2);
}

void CameraFECalibrator::lidar_projectionFE(
    const Eigen::Matrix3d &camera_intrinsic,
    const std::vector<double>& distortion_coeffs,
    const Eigen::Matrix<double, 3, 4> &extrinsic, const cv::Point3f& pt,
    std::vector<cv::Point2f>& img_pt) {

  Eigen::Vector4d lidar_point(pt.x, pt.y, pt.z, 1.0);
  Eigen::Vector3d transformed_point = extrinsic * lidar_point;

  std::vector<cv::Point3f> cam_points = {cv::Point3f(
      static_cast<float>(transformed_point(0)),
      static_cast<float>(transformed_point(1)),
      static_cast<float>(transformed_point(2))
  )};

  // Étape 2: Projeter le point transformé à l'aide de la fonction `equidistantProjectPoints`
  this->equidistantProjectPoints(cam_points, camera_intrinsic, distortion_coeffs, img_pt);
}

void CameraFECalibrator::make_board_points(const cv::Size &chessboard_size_) {
  std::vector<cv::Point2f> vec_points;
  std::vector<cv::Point3f> vec_points_3d;
  for (int r = 0; r < chessboard_size_.height; ++r) {
    for (int c = 0; c < chessboard_size_.width; ++c) {
      vec_points.emplace_back(c, r);
      vec_points_3d.emplace_back(c, r, 0);
    }
  }
  _boards_pts.push_back(vec_points);
  _boards_pts_3d.push_back(vec_points_3d);

  std::vector<cv::Point3f> vec_points_cir;
  // vec_points_cir.emplace_back(1.82f, -3.12f, 0.0f);  // sim
  // vec_points_cir.emplace_back(14.18f, -3.12f, 0.0f); // sim
  // vec_points_cir.emplace_back(1.82f, 9.12f, 0.0f);   // sim
  // vec_points_cir.emplace_back(14.18f, 9.12f, 0.0f);  // sim

  // vec_points_cir.emplace_back(1.0f, -2.0f, 0.0f);  // sim
  // vec_points_cir.emplace_back(6.0f, -2.0f, 0.0f); // sim
  // vec_points_cir.emplace_back(1.0f, 7.0f, 0.0f);   // sim
  // vec_points_cir.emplace_back(6.0f, 7.0f, 0.0f);  // sim


  // vec_points_cir.emplace_back(-5.0f, -6.0f, 0.0f);  // sim
  // vec_points_cir.emplace_back(12.0f, -6.0f, 0.0f); // sim
  // vec_points_cir.emplace_back(-5.0f, 10.0f, 0.0f);   // sim
  // vec_points_cir.emplace_back(12.0f, 10.0f, 0.0f);  // sim

  vec_points_cir.emplace_back(-2.0f, 2.33f, 0.0f);  // sim
  vec_points_cir.emplace_back(10.0f, 2.33f, 0.0f); // sim
  vec_points_cir.emplace_back(-2.0f, 5.660f, 0.0f);   // sim
  vec_points_cir.emplace_back(10.0f, 5.66f, 0.0f);  // sim


  _boards_pts_cir.push_back(vec_points_cir); // scl
}

void CameraFECalibrator::refine_all(
    Eigen::Matrix3d &camera_matrix_, Eigen::VectorXd &k_,
    std::vector<Eigen::MatrixXd> &vec_extrinsics_) {

  Params params, params_refined;
  params.camera_matrix = camera_matrix_;
  params.k = k_;
  params.vec_rt = vec_extrinsics_;
  optimier.refine_all_camera_params(params, _imgs_pts, _boards_pts,
                                    params_refined);
  camera_matrix_ = params_refined.camera_matrix;
  k_ = params_refined.k;
  vec_extrinsics_ = params_refined.vec_rt;
}

void CameraFECalibrator::refine_lidar2camera(
    Eigen::Matrix3d &camera_matrix_, Eigen::VectorXd &k_,
    std::vector<Eigen::MatrixXd> &vec_extrinsics_,
    Eigen::Matrix<double, 3, 4> &initial_extrinsic,
    std::vector<LidarPointPair> &lidar_point_pairs) {
  LidarParams params, params_refined;
  params.camera_matrix = camera_matrix_;
  params.k = k_;
  params.vec_rt = vec_extrinsics_;
  params.extrinsic = initial_extrinsic;

  // optimier.refine_lidar2camera_params(params, _imgs_pts, _boards_pts,
  //                                     lidar_point_pairs, params_refined);

  optimier.refine_lidar2FEcamera_params(params, _imgs_pts, _boards_pts,
                                      lidar_point_pairs, params_refined);
  //
  camera_matrix_ = params_refined.camera_matrix;
  k_ = params_refined.k;
  vec_extrinsics_ = params_refined.vec_rt;
  initial_extrinsic = params_refined.extrinsic;
}

void CameraFECalibrator::equidistantProjectPoints(const std::vector<cv::Point3f>& cam_points, 
                                                  const Eigen::Matrix3d& camera_intrinsic,
                                                  const std::vector<double>& distortion_coeffs,
                                                  std::vector<cv::Point2f>& imgpoints_cir){
  // Coefficients de distorsion
  Eigen::VectorXd coefficients = Eigen::Map<const Eigen::VectorXd>(distortion_coeffs.data(), distortion_coeffs.size());
  Eigen::VectorXd power = Eigen::VectorXd::LinSpaced(coefficients.size(), 3, 2 * coefficients.size() + 1);

  // Récupération du centre optique à partir de la matrice intrinsèque
  double cx = camera_intrinsic(0, 2);
  double cy = camera_intrinsic(1, 2);
  double focal = camera_intrinsic(0, 0);

  imgpoints_cir.clear();  // Assurez-vous que le vecteur est vide avant de commencer
  for (const auto& point : cam_points) {
    double x = point.x;
    double y = point.y;
    double z = point.z;

    // Calcul de chi (distance radiale dans le plan image)
    double chi = std::sqrt(x * x + y * y);

    // Calcul de theta (angle entre le point 3D et l'axe optique Z)
    double theta = std::atan2(chi, z);

    // Calcul de rho (distance projetée dans l'image)
    double rho = focal * (theta + distortion_coeffs[0] * std::pow(theta, 3) +
                             distortion_coeffs[1] * std::pow(theta, 5) +
                             distortion_coeffs[2] * std::pow(theta, 7) +
                             distortion_coeffs[3] * std::pow(theta, 9));

    // Projection dans le plan image
    cv::Point2f img_point;
    if (chi != 0) {
      img_point.x = (rho * x) / chi + cx; 
      img_point.y = (rho * y) / chi + cy;
    } else {
      img_point.x = cx;
      img_point.y = cy;
    }

    imgpoints_cir.push_back(img_point);  // Stockage dans le vecteur passé en argument
  }
}

void CameraFECalibrator::applyExtrinsicTransformation(const std::vector<cv::Point3f>& points_3d, 
                                                      const cv::Mat& rvec, const cv::Mat& tvec,
                                                      std::vector<cv::Point3f>& cam_points){
  cv::Mat R;
  cv::Rodrigues(rvec, R); // Conversion du vecteur de rotation en matrice de rotation

  std::cout << "Matrice de rotation " << R << std::endl;
  cv::Mat RtR = R * R.t();
  std::cout << "R * R^T = " << RtR << std::endl;
  std::cout << "Det(R) = " << cv::determinant(R) << std::endl;
  cam_points.clear();  // Assurez-vous que le vecteur est vide avant de commencer

  for (const auto& point : points_3d) {
      cv::Mat p_3d = (cv::Mat_<double>(3, 1) << point.x, point.y, point.z);
      cv::Mat cam_point = R * p_3d + tvec; // Rotation et translation
      cam_points.emplace_back(cam_point.at<double>(0), cam_point.at<double>(1), cam_point.at<double>(2));
  }
}