#include <math.h>
#include <cstring>
#include <algorithm>
#include "PID.h"


PID::PID() {}

PID::~PID() {
  if (i_error_buffer != nullptr) {
    delete[] i_error_buffer;
  }
}

void PID::init(double Kp, double Ki, double Kd, double tol,
    double speed_rate, double speed, double throttle, double brake, double cruise,
    int steer_hist_max, bool verb) {
  this->verbose = verb;
  this->Kp = Kp;
  this->Ki = Ki;
  this->Kd = Kd;
  this->cte_tol = tol;
  this->speed_rate = speed_rate;
  this->speed_limit = speed;
  this->brake_throttle = brake;
  this->normal_throttle = throttle;
  this->soft_throttle = cruise;

  this->steer_hist_max = steer_hist_max;
  p_error = 0;
  i_error = 0;
  d_error = 0;
  i_error_buffer = new double[buffer_size];
  std::memset(i_error_buffer, 0, buffer_size * sizeof(double));
  steer_hist_size = 0;
  steer_hist_index = 0;
  steer_hist_sum = 0;
  steer_history = new double[steer_hist_max];
  std::memset(steer_history, 0, steer_hist_max * sizeof(double));
}

void PID::updateError(double cte, double speed) {
  if (cte > 0) {
    cte = std::max(0.0, cte-cte_tol);
  } else if (cte < 0) {
    cte = std::min(0.0, cte+cte_tol);
  }
  d_error = cte - p_error;
  i_error += (cte-i_error_buffer[buffer_ptr]);
  i_error_buffer[buffer_ptr] = cte;
  buffer_ptr = (buffer_ptr+1) % buffer_size;
  p_error = cte;

  if (speed > speed_rate) {
    steer_ctrl = sqrt(speed/speed_rate)*(
        -Kp*p_error - Kd*(1+0.1*sqrt(speed/speed_rate))*d_error - Ki*i_error);
  } else {
    steer_ctrl = -Kp*p_error - Kd*d_error - Ki*i_error;
  }
  steer_ctrl = clip_f(steer_ctrl);

  steer_hist_sum += (steer_ctrl-steer_history[steer_hist_index]);
  steer_history[steer_hist_index] = steer_ctrl;
  steer_hist_index = (steer_hist_index+1) % steer_hist_max;
  steer_hist_size = (steer_hist_size==steer_hist_max) ? steer_hist_max : steer_hist_size+1;

  steer_ctrl = steer_hist_sum / steer_hist_size;
  throttle_ctrl = (speed>speed_limit) ? 0 : normal_throttle;
  if (cte > 0.5 || steer_ctrl > 0.5) {
    if (speed > 40) {
      throttle_ctrl = brake_throttle;
    } else if (speed > 15) {
      throttle_ctrl = soft_throttle / 2;
    }
    else {
      throttle_ctrl = soft_throttle;
    }
  }
}

