#include <math.h>
#include <cstring>
#include <algorithm>
#include "PID.h"

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {
  if (i_error_buffer != nullptr) {
    delete[] i_error_buffer;
  }
}

void PID::init(double Kp, double Ki, double Kd,
    double speed_rate, double speed, double throttle) {
  this->Kp = Kp;
  this->Ki = Ki;
  this->Kd = Kd;
  this->speed_rate = speed_rate;
  this->speed_limit = speed;
  this->normal_throttle = throttle;
  p_error = 0;
  i_error = 0;
  d_error = 0;
  i_error_buffer = new double[buffer_size];
  std::memset(i_error_buffer, 0, buffer_size * sizeof(double));
}

void PID::updateError(double cte, double speed) {
  d_error = cte - p_error;
  i_error += (cte-i_error_buffer[buffer_ptr]);
  i_error_buffer[buffer_ptr] = cte;
  buffer_ptr = (buffer_ptr+1) % buffer_size;
  p_error = cte;

  steer_ctrl = -Kp*p_error - Kd*d_error - Ki*i_error;
  if (speed > speed_rate) {
    steer_ctrl *= sqrt(speed/speed_rate);
  }
  steer_ctrl = clip_f(steer_ctrl);
  throttle_ctrl = (speed>speed_limit) ? 0 : normal_throttle;
  if (cte > 0.5 || steer_ctrl > 0.5) {
    if (speed > 40) {
      throttle_ctrl = -0.5;
    } else if (speed > 15) {
      throttle_ctrl = 0.;
    }
    else {
      throttle_ctrl = 0.5;
    }
  }
}

