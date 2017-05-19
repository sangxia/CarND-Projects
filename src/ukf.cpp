#include "ukf.h"
#include <math.h>
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  Acoeff_ = sqrt(n_aug_+lambda_);
  weights_ = VectorXd(2*n_aug_+1);
  weights_(0) = lambda_ / (lambda_+n_aug_);
  for (int i=1; i<2*n_aug_+1; i++) {
    weights_(i) = 0.5/(lambda_ + n_aug_);
  }

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;
  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(n_x_);
  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.4;

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    P_.fill(0.0);
    for (int i=0; i<n_x_; i++) {
      P_(i,i) = 1;
    }
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      float rho = meas_package.raw_measurements_(0);
      float theta = meas_package.raw_measurements_(1);
      float rho_dot = meas_package.raw_measurements_(2);
      x_ << rho*cos(theta), rho*sin(theta), rho_dot, theta, 0;
      P_(0,0) = std_radr_ * std_radr_;
      P_(1,1) = std_radr_ * std_radr_;
      P_(2,2) = std_radrd_ * std_radrd_;
      P_(3,3) = std_radphi_ * std_radphi_;
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_(0), 
                 meas_package.raw_measurements_(1), 0, 0, 0;
      P_(0,0) = std_laspx_ * std_laspx_;
      P_(1,1) = std_laspy_ * std_laspy_;
    }
    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;
    return;
  }

  double delta_t = (meas_package.timestamp_ - time_us_) * 1e-6;
  Prediction(delta_t);
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
  time_us_ = meas_package.timestamp_;
  return;
}

double UKF::get_yawdd(int i) {
  if (i==n_aug_) {
    return Acoeff_*std_yawdd_;
  } else if (i==n_aug_*2) {
    return -Acoeff_*std_yawdd_;
  } else {
    return 0;
  }
}

double UKF::get_nu_a(int i) {
  if (i==n_aug_-1) {
    return Acoeff_*std_a_;
  } else if (i==n_aug_*2-1) {
    return -Acoeff_*std_a_;
  } else {
    return 0;
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  MatrixXd A = P_.llt().matrixL();
  Xsig_pred_ = x_.replicate(1, 2*n_aug_+1);
  Xsig_pred_.block(0,1,n_x_,n_x_) += Acoeff_*A;
  Xsig_pred_.block(0,n_aug_+1,n_x_,n_x_) -= Acoeff_*A;

  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    double yawd = Xsig_pred_(4,i);
    double nu_a = get_nu_a(i);
    double nu_yawdd = get_yawdd(i);
    //predicted x,y
    double px_p, py_p;
    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }
    // predicted speed and yaw
    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;
    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;
    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;
    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
  // predict mean and covariance
  x_ = Xsig_pred_ * weights_;
  P_.fill(0.0);
  for (int i = 0; i < 2*n_aug_ + 1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    while (x_diff(4)> M_PI) x_diff(4)-=2.*M_PI;
    while (x_diff(4)<-M_PI) x_diff(4)+=2.*M_PI;
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  int n_z = 2;
  MatrixXd Zsig = Xsig_pred_.block(0,0,n_z,2*n_aug_+1);
  VectorXd z_pred = Zsig * weights_;
  MatrixXd S = MatrixXd(n_z, n_z);
  MatrixXd Tc = MatrixXd(n_x_, n_z); //cross correlation matrix Tc
  S.fill(0.0);
  Tc.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++) { 
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S = S + weights_(i) * z_diff * z_diff.transpose();
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  //add measurement noise covariance matrix
  S(0,0) += std_laspx_*std_laspx_;
  S(1,1) += std_laspy_*std_laspy_;

  MatrixXd K = Tc * S.inverse(); //Kalman gain K;
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred; //residual

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1);
  for (int i = 0; i < 2*n_aug_+1; i++) {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    double vx = cos(yaw)*v;
    double vy = sin(yaw)*v;
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = vx*cos(Zsig(1,i)) + vy*sin(Zsig(1,i));   //r_dot
  }
  // prediction mean and covariance
  VectorXd z_pred = Zsig * weights_;
  MatrixXd S = MatrixXd(n_z, n_z);
  MatrixXd Tc = MatrixXd(n_x_, n_z); //cross correlation matrix Tc
  S.fill(0.0);
  Tc.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++) { 
    VectorXd z_diff = Zsig.col(i) - z_pred;
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    S = S + weights_(i) * z_diff * z_diff.transpose();
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    while (x_diff(4)> M_PI) x_diff(4)-=2.*M_PI;
    while (x_diff(4)<-M_PI) x_diff(4)+=2.*M_PI;
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  //add measurement noise covariance matrix
  S(0,0) += std_radr_*std_radr_;
  S(1,1) += std_radphi_*std_radphi_;
  S(2,2) += std_radrd_*std_radrd_;

  MatrixXd K = Tc * S.inverse(); //Kalman gain K;
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred; //residual
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}

