#include "PID.h"
#include <algorithm>

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::init(double Kp, double Ki, double Kd) {
  this->Kp = Kp;
  this->Ki = Ki;
  this->Kd = Kd;
  p_error = 0;
  i_error = 0;
  d_error = 0;
}

double PID::updateError(double cte) {
  d_error = cte - p_error;
  i_error = (i_error+cte) * (1.0-Ki*1e-1);
  p_error = cte;
  return max(-1.0, min(1.0, -Kp*p_error - Kd*d_error - Ki*i_error));
}

