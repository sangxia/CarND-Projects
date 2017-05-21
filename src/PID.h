#ifndef PID_H
#define PID_H

class PID {
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
  void init(double Kp, double Ki, double Kd);

  /*
  * Update the PID error variables given cross track error.
  * Return a new steering angle
  */
  double updateError(double cte);

};

#endif /* PID_H */
