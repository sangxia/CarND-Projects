/*
 * particle_filter.h
 *
 * 2D particle filter class.
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include <random>
#include "helper_functions.h"

using namespace std;

struct Particle {
	int id;
	double x;
	double y;
	double theta;
	double weight;
};

class ParticleFilter {
	// Number of particles to draw
	int num_particles; 
	
	// Flag, if filter is initialized
	bool is_initialized;

	// Vector of weights of all particles
	// vector<double> weights;

	// Set of current particles
	vector<Particle> particles;

  default_random_engine gen;

  // standard gaussian, used to add noise
  normal_distribution<double> gaussian;

  // uniform distribution
  uniform_int_distribution<int> uniform_index;

  uniform_real_distribution<double> uniform_r;

  Particle best_particle;

  void updateBestParticle();

public:

	// Constructor
	ParticleFilter() : num_particles(0), is_initialized(false) {}

	// Destructor
	~ParticleFilter() {}

	/**
	 * init Initializes particle filter by initializing particles to Gaussian
	 *   distribution around first position and all the weights to 1.
	 * @param x Initial x position [m] (simulated estimate from GPS)
	 * @param y Initial y position [m]
	 * @param theta Initial orientation [rad]
	 * @param std[] Array of dimension 3 [standard deviation of x [m], 
   *   standard deviation of y [m], standard deviation of yaw [rad]]
   * @param num_p Number of particles to use
	 */
	void init(double x, double y, double theta, double std[], int num_p = 10);

	/**
	 * prediction Predicts the state for the next time step
	 *   using the process model.
	 * @param delta_t Time between time step t and t+1 in measurements [s]
	 * @param std_pos[] Array of dimension 3 [standard deviation of x [m], 
   *  standard deviation of y [m], standard deviation of yaw [rad]]
	 * @param velocity Velocity of car from t to t+1 [m/s]
	 * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
	 */
	void prediction(double delta_t, double std_pos[], double velocity, double yaw_rate);
	
	/**
	 * updateWeights Updates the weights for each particle based on the likelihood of the 
	 *   observed measurements. 
	 * @param sensor_range Range [m] of sensor
	 * @param std_landmark[] std dev of x and y
	 * @param observations Vector of landmark observations
	 * @param map Map class containing map landmarks
	 */
	void updateWeights(double sensor_range, double std_landmark[], vector<LandmarkObs> observations,
			Map map_landmarks);
	
	/**
	 * resample Resamples from the updated set of particles to form
	 *   the new set of particles.
	 */
	void resample();

  /**
   * return the best particle
   */
  Particle getBestParticle() { return best_particle; }
	
	/*
	 * write Writes particle positions to a file.
	 * @param filename File to write particle positions to.
	 */
	void write(string filename);
	
	/**
	 * initialized Returns whether particle filter is initialized yet or not.
	 */
	const bool initialized() const {
		return is_initialized;
	}
};

#endif /* PARTICLE_FILTER_H_ */
