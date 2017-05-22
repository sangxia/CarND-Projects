#ifndef PID_H
#define PID_H

#include <algorithm>

class PID {
  
  inline double clip_f(double val) {
    return std::min(1.0, std::max(-1.0, val));
  }

public:
  /*
  * Errors
  */
  double p_error; // this is the previous cte error
  double i_error;
  double d_error;

  int buffer_size = 100;
  int buffer_ptr = 0;
  double *i_error_buffer = nullptr;

  double speed_rate = 15;
  double speed_limit = 60;
  double normal_throttle = 0.5;
  double brake_throttle = -0.5;

  double steer_ctrl;
  double throttle_ctrl;

  int steer_hist_max, steer_hist_size, steer_hist_index;
  double steer_hist_sum;
  double *steer_history;

  /*
  * Coefficients
  */ 
  double Kp;
  double Ki;
  double Kd;

  /*
  * Constructor
  */
  PID();

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  void init(double Kp, double Ki, double Kd, double speed_rate,
      double speed, double throttle, double brake, int steer_hist_max);

  /*
  * Update the PID error variables given cross track error.
  * Return a new steering angle
  */
  void updateError(double cte, double speed);

};

#endif /* PID_H */
