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
  is_initialized = true;
  num_particles = num_p;
  gaussian = normal_distribution<double>(0.0, 1.0);
  uniform_r = uniform_real_distribution<double>(0.0, 1.0);
  uniform_index = uniform_int_distribution<int>(0, num_particles-1);
  particles.clear();
  for (int i=0; i<num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = x + std[0]*gaussian(gen);
    p.y = y + std[1]*gaussian(gen);
    p.theta = theta + std[2]*gaussian(gen);
    p.weight = 1;
    particles.push_back(p);
  }
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
      new_x = it->x + velocity*cos(it->theta)*delta_t;
      new_y = it->y + velocity*sin(it->theta)*delta_t;
      new_theta = it->theta;
    }
    // add noise to create some diversity in the particle set
    it->x = new_x + std_pos[0]*gaussian(gen);
    it->y = new_y + std_pos[1]*gaussian(gen);
    it->theta = new_theta + std_pos[2]*gaussian(gen);
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		vector<LandmarkObs> observations, Map map_landmarks) {
  double sq_sensor_range = sensor_range * sensor_range;
  for (vector<Particle>::iterator itp=particles.begin(); itp!=particles.end(); itp++) {
    double w = 0.0;
    for (vector<LandmarkObs>::iterator itobs=observations.begin();
        itobs!=observations.end(); itobs++) {
      // calculate observation in map coordinate
      double costh = cos(itp->theta);
      double sinth = sin(itp->theta);
      double est_x = itp->x + itobs->x * costh - itobs->y * sinth;
      double est_y = itp->y + itobs->x * sinth + itobs->y * costh;
      // TODO will there be two detections matched to the same landmark?
      double minsqdist = 1e9;
      int best_lmi = -1;
      double best_sq_xdiff = 1e9, best_sq_ydiff = 1e9;
      for (vector<Map::single_landmark_s>::iterator itlm=map_landmarks.landmark_list.begin();
          itlm!=map_landmarks.landmark_list.end(); itlm++) {
        double sq_xdiff = (est_x-itlm->x_f) * (est_x-itlm->x_f);
        double sq_ydiff = (est_y-itlm->y_f) * (est_y-itlm->y_f);
        if (sq_xdiff+sq_ydiff < minsqdist) {
          minsqdist = sq_xdiff+sq_ydiff;
          best_sq_xdiff = sq_xdiff;
          best_sq_ydiff = sq_ydiff;
          best_lmi = itlm->id_i;
        }
      }
      w += (-(best_sq_xdiff/(std_landmark[0]*std_landmark[0]) +
            best_sq_ydiff/(std_landmark[1]*std_landmark[1])));
    }
    itp->weight = w;
  }
  updateBestParticle();
}

void ParticleFilter::resample() {
  vector<Particle> new_particles;
  int index = uniform_index(gen);
  double beta = 0.0;
  for (int i=0; i<num_particles; i++) {
    beta += uniform_r(gen) * 2.0;
    while (beta > particles[index].weight) {
      beta -= particles[index].weight;
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }
  particles = new_particles;
}

void ParticleFilter::write(string filename) {
	// You don't need to modify this file.
	ofstream dataFile;
	dataFile.open(filename, ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}

void ParticleFilter::updateBestParticle() {
  double best_weight = -1e9;
  for (vector<Particle>::iterator it=particles.begin(); it!=particles.end(); it++) {
    if (it->weight > best_weight) {
      best_weight = it->weight;
      best_particle = *it;
    }
  }
  for (vector<Particle>::iterator it=particles.begin(); it!=particles.end(); it++) {
    it->weight = exp(it->weight - best_weight);
  }
}

