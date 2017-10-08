#ifndef POLYUTILS_H
#define POLYUTILS_H

#include "Eigen-3.3/Eigen/Core"

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  double xpow = 1.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs(i) * xpow;
    xpow *= x;
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int degree) {
  assert(xvals.size() == yvals.size());
  assert(degree >= 1 && degree <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), degree + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < degree; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

Eigen::VectorXd get_poly_derivative(Eigen::VectorXd coeffs) {
  assert(coeffs.size()>0);
  Eigen::VectorXd ret(coeffs.size()-1);
  for (int i=1; i<coeffs.size(); i++) {
    ret(i-1) = i*coeffs(i);
  }
  return ret;
}

#endif
