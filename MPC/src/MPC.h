#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

using namespace std;

struct Parameters {
    bool verbose;
    // reference values
    double ref_v;
    double ref_cte;
    double ref_epsi;

    double actuator_delay;
    size_t delay_cycle;

    // importance of the various components for cost
    double w_v;
    double w_cte;
    double w_epsi;
    double w_delta;
    double w_a;
    double w_ddelta;
    double w_da;
    double time_discount;

    size_t N;
    double dt;

    // starting indices of state and actuator variables
    // N copies for each state variable
    // N-1 copies for each actuator variable
    size_t x_start;
    size_t y_start;
    size_t psi_start;
    size_t v_start;
    size_t cte_start;
    size_t epsi_start;
    size_t delta_start;
    size_t a_start;
};

class MPC {
  public:
    Parameters param;
  
    // This value assumes the model presented in the classroom is used.
    //
    // It was obtained by measuring the radius formed by running the vehicle in the
    // simulator around in a circle with a constant steering angle and velocity on a
    // flat terrain.
    //
    // Lf was tuned until the the radius formed by the simulating the model
    // presented in the classroom matched the previous radius.
    //
    // This is the length from front to CoG that has a similar radius.
    const double Lf = 2.67;

    MPC() {}
  
    ~MPC() {}

    void init(double ref_v, double ref_cte, double ref_epsi, double actuator_delay,
        double w_v, double w_cte, double w_epsi, double w_delta, double w_a,
        double w_ddelta, double w_da, double time_discount, size_t N, double dt,
        bool verbose);

    // Solve the model given an initial state and polynomial coefficients.
    // Return the first actuatotions.
    vector<double> Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs,
        Eigen::VectorXd coeffs_d, double prev_delta, double prev_a);
};

#endif /* MPC_H */
