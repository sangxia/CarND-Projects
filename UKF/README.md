# Unscented Kalman Filter Project

Use Unscented Kalman Filter to estimate the state of a moving object of interest with noisy lidar and radar measurements.

## Command line interface

The format is `./UnscentedKF [-l] [-r] <path/to/input> <path/to/output>`.
Use `-l` and `-r` to *exclude* lidar or radar measurements.

I also implemented `check_nis.py` to compare the NIS values from the output
file with the reference values. The interface is 
`check_nis.py <path/to/ukfoutputfile>`

## Results

The following uses `std_a_=3` and `std_yawdd_=0.4`.
The reference values are 
`L .05 = 0.103`,
`L .95 = 5.991`,
`R .05 = 0.352`,
`R .95 = 7.815`.

| Source         | RMSE px | RMSE py | RMSE vx | RMSE vy | L .05 | L .95 | R .05 | R .95 |
|:-------------- | -------:| -------:| -------:| -------:| -----:| -----:| -----:| -----:|
| Lidar+Radar    |  0.0750 |  0.0826 |  0.3462 |  0.2632 | 0.111 | 4.942 | 0.294 | 7.187 |
| Lidar only     |  0.1082 |  0.0958 |  0.5073 |  0.2701 | 0.088 | 5.790 |       |       |
| Radar only     |  0.1775 |  0.2856 |  0.3624 |  0.6798 |       |       | 0.295 | 8.847 |

Increasing the process noise standard deviations decreases the 
quantile values, and vice versa. 

Compared with results from 
[my EKF implementation](https://github.com/sangxia/CarND-Extended-Kalman-Filter-Project), 
results from UKF are in general much better, although the improvements seems larger
for x-axis than y-axis, and RMSE py for Radar only got slightly worse.

## Dependencies

* cmake >= v3.5
* make >= v4.1
* gcc/g++ >= v5.4

For `check_nis.py`:
* python >= 3.5
* pandas

