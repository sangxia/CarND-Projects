#include "kalman_filter.h"
#include <math.h>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Matrix;
using std::cout;
using std::endl;

KalmanFilter::KalmanFilter() {
  I4 = Matrix<double, 4, 4>::Identity();
}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::UpdateF(double dt) {
  F_(0, 2) = dt;
  F_(1, 3) = dt;
}

void KalmanFilter::UpdateQ(double dt) {
  double noise_ax = 9;
  double noise_ay = 9;

  double dt_2 = dt*dt;
  double dt_3 = dt_2*dt;
  double dt_4 = dt_3*dt;

  Q_(0, 0) = dt_4/4*noise_ax;
  Q_(0, 2) = dt_3/2*noise_ax;
  Q_(1, 1) = dt_4/4*noise_ay;
  Q_(1, 3) = dt_3/2*noise_ay;
  Q_(2, 0) = dt_3/2*noise_ax;
  Q_(2, 2) = dt_2*noise_ax;
  Q_(3, 1) = dt_3/2*noise_ay;
  Q_(3, 3) = dt_2*noise_ay;
}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::UpdateLaser(const VectorXd &z) {
  /**
    * update the state by using Kalman Filter equations
  */
  VectorXd y = z - H_ * x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();
  x_ = x_ + K*y;
  P_ = (I4 - K*H_) * P_;
}

void KalmanFilter::UpdateRadar(const VectorXd &z) {
  /**
    * update the state by using Extended Kalman Filter equations
  */
  VectorXd x_polar(3);
  x_polar(0) = sqrt(x_(0)*x_(0)+x_(1)*x_(1));
  if (abs(x_(0))<1e-6) {
    if (x_(1) > 0) {
      x_polar(1) = M_PI/2;
    }
    else {
      x_polar(1) = -M_PI/2;
    }
  }
  else {
    if (x_(0) > 0) {
      x_polar(1) = atan(x_(1)/x_(0));
    }
    else if (x_(1) > 0) {
      x_polar(1) = atan(x_(1)/x_(0)) + M_PI;
    }
    else {
      x_polar(1) = atan(x_(1)/x_(0)) - M_PI;
    }
  }
  double ux = cos(x_polar(1));
  double uy = sin(x_polar(1));
  x_polar(2) = ux*x_(2) + uy*x_(3);
  VectorXd y = z - x_polar;
  while (y(1) < -M_PI-1e-3) {
    y(1) += 2*M_PI;
  }
  while (y(1) > M_PI+1e-3) {
    y(1) -= 2*M_PI;
  }
  H_(0,0) = cos(x_polar(1));
  H_(0,1) = sin(x_polar(1));
  H_(2,2) = H_(0,0);
  H_(2,3) = H_(0,1);
  if (x_polar(0) > 1e-6) {
    H_(1,0) = - x_(1) / (x_polar(0)*x_polar(0));
    H_(1,1) = x_(0) / (x_polar(0)*x_polar(0));
    H_(2,0) = x_(1) * (x_(2)*x_(1) - x_(0)*x_(3)) / (x_polar(0)*x_polar(0)*x_polar(0));
    H_(2,1) = x_(0) * (x_(0)*x_(3) - x_(1)*x_(2)) / (x_polar(0)*x_polar(0)*x_polar(0));
  }
  else {
    H_(1,0) = 0; H_(1,1) = 0;
    H_(2,0) = 0; H_(2,1) = 0;
  }
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();
  x_ = x_ + K*y;
  P_ = (I4 - K*H_) * P_;
}

