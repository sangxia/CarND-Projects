#include <math.h>
#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth, 
                              int start) {
  int size = estimations.size();
  if (size == 0) {
    return VectorXd::Constant(4, 0.0);
  }
  VectorXd res = VectorXd::Constant(estimations[0].size(), 0.0);
  for (int i=start; i<size; i++) {
    VectorXd err = estimations[i]-ground_truth[i];
    err = err.array()*err.array();
    res += err;
  }
  res = res/size;
  return res.array().sqrt();
}

