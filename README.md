# Particle Filter Localization Project

## Project Introduction
Your robot has been kidnapped and transported to a new location! 
Luckily it has a map of this location, a (noisy) GPS estimate of its initial location, 
and lots of (noisy) sensor and control data.

This project implements 2-dimensional localization with a particle filter.
The particle filter is given a map and some initial localization information 
(analogous to what a GPS would provide). 
At each time step it receives observation and control data. 

## Running the Code

In the repository's root directory, run the following commands from the command line:

```
> ./clean.sh
> ./build.sh
> ./run.sh [num_particles]
```

I made some changes to the interfaces, so `particle_filter.cpp` may not work with the
official grader. The file `particle_filter_sub.cpp` is the one that is compatible with
[the official version](https://github.com/udacity/CarND-Kidnapped-Vehicle-Project/commit/4829b0110f0eb91d4e677943007a806ba316e6d5).

| # Particles | x error | y error | yaw error | runtime (sec) |
|:----------- | -------:| -------:| ---------:| -------------:|
| 2           | 0.51303 | 0.43371 |   0.01618 |       0.09401 |
| 10          | 0.15583 | 0.14854 |   0.00506 |       0.23801 |
| 30          | 0.12587 | 0.11743 |   0.00415 |       0.61315 |
| 100         | 0.11525 | 0.10666 |   0.00364 |       1.96519 |

## Data

### The Map*
`map_data.txt` includes the position of landmarks (in meters) on an arbitrary Cartesian 
coordinate system. Each row has three columns
1. x position
2. y position
3. landmark id

> * Map data provided by 3D Mapping Solutions GmbH.

### Control Data
`control_data.txt` contains rows of control data. Each row corresponds to the control data 
for the corresponding time step. The two columns represent
1. vehicle speed (in meters per second)
2. vehicle yaw rate (in radians per second)

#### Observation Data
The `observation` directory includes around 2000 files. Each file is numbered according 
to the timestep in which that observation takes place. 

These files contain observation data for all "observable" landmarks. Here observable 
means the landmark is sufficiently close to the vehicle. Each row in these files corresponds 
to a single landmark. The two columns represent:
1. x distance to the landmark in meters (right is positive) RELATIVE TO THE VEHICLE. 
2. y distance to the landmark in meters (forward is positive) RELATIVE TO THE VEHICLE.

