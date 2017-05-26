#include "MPC.h"
#include <math.h>
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;


class FG_eval {
  public:
    // Coefficients of the fitted polynomial.
    Eigen::VectorXd coeffs;
    double Lf;
    double prev_a;
    double prev_delta;

    Parameters param;

    FG_eval(Eigen::VectorXd coeffs, Parameters p, double Lf, double prev_delta, 
        double prev_a) { 
      this->param = p;
      this->coeffs = coeffs;
      this->Lf = Lf;
      this->prev_delta = prev_delta;
      this->prev_a = prev_a;
    }
  
    typedef CPPAD_TESTVECTOR(AD<double>) ADvector;

    AD<double> evalpoly(AD<double> x) {
      AD<double> ret = 0.0;
      AD<double> xpow = 1.0;
      for (int i=0; i<coeffs.size(); i++) {
        ret += coeffs(i) * xpow;
        xpow *= x;
      }
      return ret;
    }

    // `fg` is a vector containing the cost and constraints.
    // `vars` is a vector containing the variable values (state & actuators).
    void operator()(ADvector& fg, const ADvector& vars) {
      // The cost is stored is the first element of `fg`.
      // Any additions to the cost should be added to `fg[0]`.
      fg[0] = 0;
      // MSE compared to the reference state.
      double time_weight = 1.0;
      for (int i = 0; i < param.N; i++) {
        fg[0] += time_weight * param.w_cte * 
          CppAD::pow(vars[param.cte_start + i] - param.ref_cte, 2);
        fg[0] += time_weight * param.w_epsi * 
          CppAD::pow(vars[param.epsi_start + i] - param.ref_epsi, 2);
        fg[0] += time_weight * param.w_v * 
          CppAD::pow(vars[param.v_start + i] - param.ref_v, 2);
        time_weight *= param.time_discount;
      }
      // Minimize the use of actuators.
      for (int i = param.delay_cycle; i < param.N - 1; i++) {
        fg[0] += param.w_delta * CppAD::pow(vars[param.delta_start + i], 2);
        fg[0] += param.w_a * CppAD::pow(vars[param.a_start + i], 2);
      }
      // Minimize the value gap between sequential actuations.
      for (int i = param.delay_cycle-1; i < param.N - 2; i++) {
        fg[0] += param.w_ddelta * 
          CppAD::pow(vars[param.delta_start + i + 1] - vars[param.delta_start + i], 2);
        fg[0] += param.w_da * 
          CppAD::pow(vars[param.a_start + i + 1] - vars[param.a_start + i], 2);
      }
  
      // relation between variables
      fg[1 + param.x_start] = vars[param.x_start];
      fg[1 + param.y_start] = vars[param.y_start];
      fg[1 + param.psi_start] = vars[param.psi_start];
      fg[1 + param.v_start] = vars[param.v_start];
      fg[1 + param.cte_start] = vars[param.cte_start];
      fg[1 + param.epsi_start] = vars[param.epsi_start];
      for (int i = 0; i < param.N - 1; i++) {
        // The state at time t.
        AD<double> x0 = vars[param.x_start + i];
        AD<double> y0 = vars[param.y_start + i];
        AD<double> psi0 = vars[param.psi_start + i];
        AD<double> v0 = vars[param.v_start + i];
        // AD<double> cte0 = vars[param.cte_start + i];
        // AD<double> epsi0 = vars[param.epsi_start + i];
        // The state at time t+1 .
        AD<double> x1 = vars[param.x_start + i + 1];
        AD<double> y1 = vars[param.y_start + i + 1];
        AD<double> psi1 = vars[param.psi_start + i + 1];
        AD<double> v1 = vars[param.v_start + i + 1];
        AD<double> cte1 = vars[param.cte_start + i + 1];
        AD<double> epsi1 = vars[param.epsi_start + i + 1];
        // reference position and orientation
        AD<double> f1 = evalpoly(x1);
        AD<double> psides0 = CppAD::atan(coeffs[1]);
        // the actuation at time t.
        fg[2 + param.x_start + i] = x1 - (x0 + v0 * CppAD::cos(psi0) * param.dt);
        fg[2 + param.y_start + i] = y1 - (y0 + v0 * CppAD::sin(psi0) * param.dt);
        fg[2 + param.cte_start + i] = cte1 - (f1 - y1);
        // TODO below is an approximation, psides0 ought to be some version of psides1
        fg[2 + param.epsi_start + i] = epsi1 - (psides0 - psi1);
        AD<double> delta0 = vars[param.delta_start + i];
        AD<double> a0 = vars[param.a_start + i];
        fg[2 + param.psi_start + i] = psi1 - (psi0 + v0 * delta0 / Lf * param.dt);
        fg[2 + param.v_start + i] = v1 - (v0 + a0 * param.dt);
      }
    }
};

//
// MPC class definition implementation.
//

void MPC::init(double ref_v, double ref_cte, double ref_epsi, double actuator_delay,
    double w_v, double w_cte, double w_epsi, double w_delta, double w_a,
    double w_ddelta, double w_da, double time_discount, size_t N, double dt, bool verbose){
  param.verbose = verbose;
  param.ref_v = ref_v;
  param.ref_cte = ref_cte;
  param.ref_epsi = ref_epsi;
  param.actuator_delay = actuator_delay;
  param.w_v = w_v;
  param.w_cte = w_cte;
  param.w_epsi = w_epsi;
  param.w_delta = w_delta;
  param.w_a = w_a;
  param.w_ddelta = w_ddelta;
  param.w_da = w_da;
  param.time_discount = time_discount;
  param.N = N;
  param.dt = dt;
  param.delay_cycle = (size_t)(ceil(actuator_delay/dt));
  param.x_start = 0;
  param.y_start = param.x_start + param.N;
  param.psi_start = param.y_start + param.N;
  param.v_start = param.psi_start + param.N;
  param.cte_start = param.v_start + param.N;
  param.epsi_start = param.cte_start + param.N;
  param.delta_start = param.epsi_start + param.N;
  param.a_start = param.delta_start + param.N - 1;
}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs,
    double prev_delta, double prev_a) {
  bool ok = true;
  size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;
  // current state
  double x = state[0];
  double y = state[1];
  double psi = state[2];
  double v = state[3];
  double cte = state[4];
  double epsi = state[5];
  // number of independent variables
  // N timesteps == N - 1 actuations
  size_t n_vars = param.N * 6 + (param.N - 1) * 2;
  // Number of constraints
  size_t n_constraints = param.N * 6;

  // Initial value of the independent variables.
  // Should be 0 except for the initial values.
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars[i] = 0.0;
  }
  // Set the initial variable values
  vars[param.x_start] = x;
  vars[param.y_start] = y;
  vars[param.psi_start] = psi;
  vars[param.v_start] = v;
  vars[param.cte_start] = cte;
  vars[param.epsi_start] = epsi;

  // Lower and upper limits for x
  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  // Set all non-actuators upper and lowerlimits
  // to large negative and positive values.
  for (int i = 0; i < param.delta_start; i++) {
    vars_lowerbound[i] = -1.0e19;
    vars_upperbound[i] = 1.0e19;
  }
  // The upper and lower limits of delta are set to -25 and 25
  // degrees (values in radians).
  for (int i = param.delta_start; i < param.a_start; i++) {
    if (i-param.delta_start < param.delay_cycle) {
      vars_lowerbound[i] = prev_delta;
      vars_upperbound[i] = prev_delta;
    } else {
      vars_lowerbound[i] = -25.0/180.0*M_PI;
      vars_upperbound[i] = 25.0/180.0*M_PI;
    }
  }
  // Acceleration/decceleration upper and lower limits.
  for (int i = param.a_start; i < n_vars; i++) {
    if (i-param.a_start < param.delay_cycle) {
      vars_lowerbound[i] = prev_a;
      vars_upperbound[i] = prev_a;
    } else {
      vars_lowerbound[i] = -1.0;
      vars_upperbound[i] = 1.0;
    }
  }

  // Lower and upper limits for constraints
  // All of these should be 0 except the initial
  // state indices.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (int i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }
  constraints_lowerbound[param.x_start] = x;
  constraints_lowerbound[param.y_start] = y;
  constraints_lowerbound[param.psi_start] = psi;
  constraints_lowerbound[param.v_start] = v;
  constraints_lowerbound[param.cte_start] = cte;
  constraints_lowerbound[param.epsi_start] = epsi;

  constraints_upperbound[param.x_start] = x;
  constraints_upperbound[param.y_start] = y;
  constraints_upperbound[param.psi_start] = psi;
  constraints_upperbound[param.v_start] = v;
  constraints_upperbound[param.cte_start] = cte;
  constraints_upperbound[param.epsi_start] = epsi;

  // Object that computes objective and constraints
  FG_eval fg_eval(coeffs, param, Lf, prev_delta, prev_a);

  // options
  std::string options;
  options += "Integer print_level  0\n";
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  options += "Numeric max_cpu_time          0.05\n";

  CppAD::ipopt::solve_result<Dvector> solution;
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  ok &= (solution.status == CppAD::ipopt::solve_result<Dvector>::success);

  auto cost = solution.obj_value;
  if (param.verbose) {
    std::cout << ok << " Cost " << cost << std::endl;
  }
  return {solution.x[param.delta_start+param.delay_cycle], 
    solution.x[param.a_start+param.delay_cycle]};
}
