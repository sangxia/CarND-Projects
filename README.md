# PID Control Project

Use PID control to steer a car around a track in a simulator using 
Cross Track Error (CTE) as feedback.

## Algorithm

The simulator supplies the algorithm with telemetry and CTE. The 
current implementation uses both CTE and speed to control steering and 
throttle / brake.

The steering command is determined by CTE, the difference between two 
consecutive CTEs, and the total CTE over the previous 100 timeframes.

The best way to use the controller is to provide a configuration file. 
For example, run `build/pid params.conf` in the root directory of the 
repository. The configuration file `params.conf` in the repository 
contains two settings, one more conservative and the other more aggressive. 
Add a # sign at the beginning of a line to comment out a line. 
No empty line is allowed in the configuration file.

### Details of the parameters

The gains for CTE, derivative and integral (`Kp`, `Kd` and `Ki`) are 
tuned by hand following essentially *twiddle*.

The parameters `const_throttle`, `brake` and `cruise` controls the 
throttle command under various conditions.

It is possible to use the average of the last several steering command 
as the final steering, and the size of this window is given by 
`steer_hist`.

The parameter `speed_max` controls the maximum speed. Once maximum 
speed is reached, the throttle is always set to 0. Using the aggressive
setting, one can achieve a maximum speed of 80+ on the test track.
The parameters `speed_r` and `cte_tol` are for improved stability under
high speed. Steering and other parameters are amplified under high speed
and the amount by which it is amplified is governed by `speed_r`.
To reduce oscillation which can cause big problems under high speed, the
algorithm accepts a range of CTE as acceptable and this range is given
by `cte_tol`. In other words, the PID controller only attempts to steer
so that CTE is within `[-cte_tol, cte_tol]`.

Speed and other drive information output can be turned off by setting
`verbose` to 0.

## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets) master branch
* Simulator. You can download these from the [project intro page](https://github.com/udacity/CarND-PID-Control-Project/releases) in the classroom.

