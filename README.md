# CarND-Controls-MPC

Using Model predictive control (MPC) to drive a car around a track 
in a simulator, using reference waypoint information as input.

## Running the Controller

The best way to use the controller is to provide it with a configuration 
file. For example, run build/pid params.conf in the root directory of 
the repository. The configuration file params.conf in the repository 
should allow the model to drive around the lake track at full speed.
However, this can be sensitive to the computational power of the machine
and excessive delay can cause problems. If the car does not drive around
the track smoothly, please change the `ref_v` parameter to a smaller value.

## The Model

Upon receiving the reference waypoints given by global coordinates, 
the controller first converts it into vehicle coordinates. Then, a polynomial
of degree 2 is fitted on these waypoints. This is the reference path used
by the controller.

The steering and throttle control settings are then found using a solver
via ipopt. The solver tries to find control settings for every `dt` seconds
for up to `N` steps in the future. Larger `N` and smaller `dt` resulted in
more accurate estimates of future states, but are computationally more expensive
for real time settings. I started with small `N` and large `dt` and quickly
settled on the current parameter setting that seems to work well. Setting `N`
to `5` and the car simply won't steer because it doesn't see far enough into
the future. Setting `dt` to large values such as `0.5` resulted in very unsmooth
drive.

The solver attempts to optimize according to the following criteria:

* Low CTE error
* Low vehicle bearing error
* Higher speed
* Smooth control, in the sense that change of control between consecutive `dt`
should not be too large, and in general the smaller the steering and throttle
the better.

The range of typical values of these three errors are very different (CTE and
speed differences take larger values, bearing errors take smaller values), so
to balance different criteria, I introduced weights for different errors, given
by `w_cte`, `w_v` and `w_epsi`. Similar weights are introduced for the smoothing
constraints (`w_delta`, `w_a`, `w_ddelta`, and `w_da`). Experiment shows that
having a larger value for `w_ddelta` is especially helpful because it gives
heavy penalty to abrupt steering change.

Under high speed, estimate of the vehicle trajectory further into the future can
be very inaccurate, resulting in very large error. To get a more stable drive, 
it makes sense to discount those errors using the parameter `time_discount`.

To account for the delay between the command being issued and it being actuated,
I simply set the first few actuator command to the current command given by
telemetry. If the delay is 0.1 seconds and `dt` is 0.05 seconds, then the first
`0.1/0.05=2` commands are set and the values of the third command in the
actual solution is taken as the command for the current step.
The delay is also a tunable parameter (`actuator_delay`). Experiment shows that
low delay setting results in more oscillation, which makes sense because when
the command is actually carried out, it is probably already too little too late
and in later steps larger corrections will be needed. Experiments also show that 
it is actually beneficial to slightly over-estimate the delay. Setting it to
`0.1` is fine but `0.15` gives a smoother drive. It could be that `0.15` is actually
a better estimate of latency when we take the running time of the optimizer into
account. Things become erratic when setting it to larger values.

## Results

The MPC with the given `params.conf` drives around the car safely at full throttle
around the whole track. It works equally well with lower speed limit (smaller `ref_v`
values). 

Here is a link to a recording of the drive.

[![Flat out around the lake track](https://img.youtube.com/vi/LHvNzXCDMc0/0.jpg)](https://www.youtube.com/watch?v=LHvNzXCDMc0)

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
* [uWebSockets](https://github.com/uWebSockets/uWebSockets) == 0.14, but the master branch will probably work just fine
  * Follow the instructions in the [uWebSockets README](https://github.com/uWebSockets/uWebSockets/blob/master/README.md) to get setup for your platform. You can download the zip of the appropriate version from the [releases page](https://github.com/uWebSockets/uWebSockets/releases). Here's a link to the [v0.14 zip](https://github.com/uWebSockets/uWebSockets/archive/v0.14.0.zip).
  * If you have MacOS and have [Homebrew](https://brew.sh/) installed you can just run the ./install-mac.sh script to install this.
* [Ipopt](https://projects.coin-or.org/Ipopt)
  * Mac: `brew install ipopt`
  * Linux
    * You will need a version of Ipopt 3.12.1 or higher. The version available through `apt-get` is 3.11.x. If you can get that version to work great but if not there's a script `install_ipopt.sh` that will install Ipopt. You just need to download the source from the Ipopt [releases page](https://www.coin-or.org/download/source/Ipopt/) or the [Github releases](https://github.com/coin-or/Ipopt/releases) page.
    * Then call `install_ipopt.sh` with the source directory as the first argument, ex: `bash install_ipopt.sh Ipopt-3.12.1`. 
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [CppAD](https://www.coin-or.org/CppAD/)
  * Mac: `brew install cppad`
  * Linux `sudo apt-get install cppad` or equivalent.
  * Windows: TODO. If you can use the Linux subsystem and follow the Linux instructions.
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). This is already part of the repo so you shouldn't have to worry about it.
* Simulator. You can download these from the [releases tab](https://github.com/udacity/CarND-MPC-Project/releases).
* Not a dependency but read the [DATA.md](./DATA.md) for a description of the data sent back from the simulator.

