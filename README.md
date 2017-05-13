# Extended Kalman Filter Project Starter Code

Utilize a kalman filter to estimate the state of a moving object of interest with noisy lidar and radar measurements. 

The program supports either taking input from files or communicating with the Term 2 Simulator using uWebSocketIO.

## Command line interface

For I/O using files, use `./ExtendedEKF file [-l] [-r] <path/to/input> <path/to/output>`.
Use `-l` and `-r` to *exclude* lidar or radar measurements from the file.

For communicating with the simulator, use `./ExtendedEKF socket`.

## Simulator Protocol

Here is the main protcol that main.cpp uses for uWebSocketIO in communicating with the simulator.

INPUT: values provided by the simulator to the c++ program

["sensor_measurement"] => the measurment that the simulator observed (either lidar or radar)

OUTPUT: values provided by the c++ program to the simulator

["estimate_x"] <= kalman filter estimated position x
["estimate_y"] <= kalman filter estimated position y
["rmse_x"]
["rmse_y"]
["rmse_vx"]
["rmse_vy"]

## Results

| Source               | RMSE px | RMSE py | RMSE vx | RMSE vy |
|:-------------------- | -------:| -------:| -------:| -------:|
| File, lidar+radar    | 0.0972  | 0.0854  | 0.4533  | 0.4609  |
| File, lidar only     | 0.1222  | 0.0984  | 0.5620  | 0.4560  |
| File, radar only     | 0.1917  | 0.2796  | 0.4610  | 0.6950  |
| Simulator, Dataset 1 | 0.0973  | 0.0855  | 0.4537  | 0.4613  |
| Simulator, Dataset 2 | 0.0732  | 0.0962  | 0.3944  | 0.4745  |

## Other Important Dependencies

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


