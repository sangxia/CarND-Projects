# CarND Path Planning Project
   
The goal of this project is to safely navigate around a virtual highway 
with other traffic that is driving +/-10 MPH of the 50 MPH speed limit. 
The path planner gets from the simulator the car's localization and sensor 
fusion data, and there is also a sparse list of waypoints that can be
used to calculate the coordinates of points around the highway. 

The path planner needs to fulfill the following objectives:
* The car should stay on the road, and do not spend more than 3 
seconds driving on more than 1 lane.
* The car should avoid hitting other cars on the road.
* In addition to the speed limit, the total acceleration of the car should 
be less than 10 m/s^2 and the jerk should be less than 50 m/s^3.
* The target speed of the car is 50 MPH, and when the car is 
obstructed by slow traffic, it will try to change lanes when possible.

The path planner communicates with the simulator to control the car by
providing a list of points that the car should visit. The car has a perfect
controller and visits every point provided on the list. There is latency
in the process, and the simulator provides a list of points provided 
previously that has not yet been visited along with localization and sensor
fusion data.

*A note for running the path planner:* there is a part in `generateTrajectory`
that outputs fairly detailed trajectory information to `stderr`. 
Since we are not generating trajectory at very high rates, this should not
cause latency problems. Otherwise, redirecting `stderr` to `tmpfs` 
or `/dev/null`, or comment out this section entirely should be sufficient.

## Details of the Path Planner

The path planner I implemented can be divided into two parts: a higher level 
part that decides on speed and lane change, and a lower level part that 
generates the trajectory that implements those decisions.

Overall, the path planner aim to generate trajectories 5 seconds into the 
future, and regenerates a path about every 3.5 seconds. It is important to
*not* generate new trajectories too often, as this would lead to worse 
latency.

### High Level Planning

This part is implemented mostly in `main.cpp`. The algorithm keeps track of
the state of the car, its current lane, whether it is currently changing lanes,
and if so, which lane it is trying to change into.

The algorithm first checks if there are enough points from the previous 
trajectory that have not been processed. If so, then the same trajectory is
returned to the simulator. The algorithm allows more time to execute previous
trajectory if the car is in the middle of a lane change. This setup improves
stability of the trajectory during lane change.

If new trajectory is needed, the algorithm then evaluates different options.
In general, the criteria is to allow lane change if (1) it is safe, and 
(2) there is a clear advantage (speed difference of at least 1 m/s).
In addition to changing into the immediate next lane, the algorithm also 
evaluates if it is beneficial to change two lanes. The different proposals
are scored by `scoreProposal` in `spline_trajectory.h`. This function
returns an integer score indicating how good / safe the proposal is, and
also indicates what the speed might be if the proposal is followed.
Safety is determined by distance from the nearest vehicle ahead and behind, 
as well as the relative speed between the vehicles.
In order to minimize acceleration and jerk, the algorithm avoids lane change
if either of the following is true:
* There is a sharp curve ahead.
* The car changed lane not so long ago.

### Low Level Trajectory Generation

After making the high level decision regarding speed and lane, the algorithm
uses `generateTrajectory` in `spline_trajectory.h` to generate the actual 
trajectory. To ensure some continuity, the first few points (specified by
`prev_lim`) are always copied to the new trajectory as long as they are not 
too close to the current point (this is to lower the probability that the 
controller sees a path point behind the vehicle due to latency). The future
trajectory is then decided by fitting a spline through those points along
with a few waypoints ahead of the vehicle. If there is a decision to change
lane, or the car is in the middle of a lane change, the `d` value in 
Frenet coordinate is adjusted accordingly. The setting is such that most
lane change can be completed within 2 seconds.

## Results and Discussions

Here is [a link to the video recording of my run](https://youtu.be/3gok9qKyr64).
As one can see in the video, the vehicle safely finished 3 laps. It is also
able to change lanes when obstructed. 

There is a period in the video (from about mile 1.6) when the vehicle is
stuck in the right most lane. Although the left most lane is fairly empty,
it was not able to change because the middle lane is busy and cars are at
fairly close distance, and the algorithm decided that it is not safe to 
change lanes. The situation lasted until around mile 7.2 when the last
vehicle in the middle lane moved past and the algorithm immediately 
decided to gradually change into the left lane.

The lane change and speed control strategy of my current implementation is
fairly conservative and avoiding collision *naturally* takes higher 
priority. The other vehicles in the simulator exhibits some pretty bad
driving behavior, such as lane change at very short distance, not keeping
enough distance, rapid acceleration and decceleration, etc., some of these
are probably rather like many human drivers. One example would be around
mile 0.53 in the video.

This also means that in order to implement a more aggressive lane change 
strategy, we would certainly need an extra safety component that runs more 
frequently than every 5 seconds (the rate the path planner is running at),
and that intervenes when emergency brakes are needed or lane change must 
be aborted.

An alternative heuristics for high level planning would be to try to 
keep the vehicle in the center lane, instead of keeping the vehicle 
in the current lane as in my current implementation. This would perhaps
provide some more flexibility when it comes to choosing lanes.
It may also help with a more sophiscated high level proposal generation
algorithm. Being able to model the behavior of other vehicles better
(rather than just comparing relative speed) can also help with this.
However, this needs to be balanced with latency requirements.

The generated trajectory sometimes wiggles slightly. I think this is 
mainly due to the fact that the waypoints are sparse, and therefore does 
not follow the actual road very precisely. It would help if the waypoints 
are denser. It would also help if waypoints for each lane are provided.

Sometimes, such wiggling can lead the vehicle slightly off road for a 
brief moment if it is driving on the left or right lane, especially
during lane change. This is currently solved by moving the target lane 
center of the left and right lane slightly towards the center lane. 
This is not ideal in certain scenarios, as the vehicle may get too 
close to vehicles in other lanes. Adding more pathpoints when generating
the spline will probably give a better control of the generated trajectory.

Besides, the current trajectory is generated by fitting two splines, one
for `x` and one for `y`, separately and both as a function of time.
Here `x` and `y` are coordinates after transforming to ego coordinate.
An alternative would be to fit a spline where `y` is a function of `x`, 
and then generate evenly spaced path points. This might also give a 
trajectory that wiggles less in certain occasions.

The car currently accelerates rather slowly, especially from cold start.
This is mostly so that the car doesn't violate the limits on total 
acceleration and jerk, together with the fact that target speed is not
updated without new trajectory being generated, that is, every 3.5 
seconds or so. A more adaptive planner here could be useful.
