/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[], int num_p) {
  num_particles = num_p;
  particles.clear();
  for (int i=0; i<num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = x + std[0]*std_gaussian(gen);
    p.y = y + std[1]*std_gaussian(gen);
    p.theta = theta + std[2]*std_gaussian(gen);
    p.weight = 1;
    particles.push_back(p);
  }
  uniform_dist = uniform_int_distribution<int>(0, num_particles-1);
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
    double velocity, double yaw_rate) {
  for (vector<Particle>::iterator it=particles.begin();
      it != particles.end(); it++) {
    double new_x, new_y, new_theta;
    if (fabs(yaw_rate) > 1e-3) {
      double yaw_delta = yaw_rate * delta_t;
      new_x = it->x + velocity/yaw_rate*(sin(it->theta+yaw_delta)-sin(it->theta));
      new_y = it->y + velocity/yaw_rate*(cos(it->theta)-cos(it->theta+yaw_delta));
      new_theta = it->theta + yaw_delta;
    } else {
      new_x = it->x + velocity*cos(it->theta);
      new_y = it->y + velocity*sin(it->theta);
      new_theta = it->theta;
    }
    it->x = new_x + std_pos[0]*std_gaussian(gen);
    it->y = new_y + std_pos[1]*std_gaussian(gen);
    it->theta = new_theta + std_pos[2]*std_gaussian(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. 
  // Your particles are located according to the MAP'S coordinate system. 
  // You will need to transform between the two systems.
	// Keep in mind that this transformation requires both rotation AND 
  // translation (but no scaling).
  //
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample() {
  vector<Particle> new_particles;
  double max_weight = 0.0;
  for (vector<Particle>::iterator it=particles.begin(); it!=particles.end(); it++) {
    if (it->weight > max_weight) {
      max_weight = it->weight;
    }
  }
  int index = uniform_dist(gen);
  double beta = 0.0;
  for (int i=0; i<num_particles; i++) {
    beta += uniform(gen) * 2.0 * max_weight;
    while (beta > particles[index].weight) {
      beta -= particles[index].weight;
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }
  particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}

void ParticleFilter::getBestParticle() {
  double best_weight = 0.0;
  Particle best_particle;
  for (vector<Particle>::iterator it=particles.begin(); 
      it != particles.end(); it++) {
    if (it->weight > best_weight) {
      best_weight = it->weight;
      best_particle = *it;
    }
  }
  return best_particle;
}

