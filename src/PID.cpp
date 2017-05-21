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

void PID::init(double Kp, double Ki, double Kd) {
  this->Kp = Kp;
  this->Ki = Ki;
  this->Kd = Kd;
  p_error = 0;
  i_error = 0;
  d_error = 0;
  i_error_buffer = new double[buffer_size];
}

double PID::updateError(double cte) {
  d_error = cte - p_error;
  i_error += (cte-i_error_buffer[buffer_ptr]);
  i_error_buffer[buffer_ptr] = cte;
  buffer_ptr = (buffer_ptr+1) % buffer_size;
  p_error = cte;
  return -Kp*p_error - Kd*d_error - Ki*i_error;
}

