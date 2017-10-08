#ifndef GEOUTILS_H
#define GEOUTILS_H

#include <math.h>
#include <vector>

using std::vector;

void global_to_vehicle(double global_x, double global_y,
    double vehicle_x, double vehicle_y, double vehicle_psi,
    double &result_x, double &result_y) {
  double temp_x = global_x - vehicle_x;
  double temp_y = global_y - vehicle_y;
  double cos_psi = cos(vehicle_psi);
  double sin_psi = sin(vehicle_psi);
  result_x = temp_x * cos_psi + temp_y * sin_psi;
  result_y = -temp_x * sin_psi + temp_y * cos_psi;
}

void global_to_vehicle(vector<double> &global_x, vector<double> &global_y,
    double vehicle_x, double vehicle_y, double vehicle_psi,
    vector<double> &result_x, vector<double> &result_y) {
  int n = global_x.size();
  result_x.resize(n);
  result_y.resize(n);
  for (int i=0; i<n; i++) {
    global_to_vehicle(global_x[i], global_y[i], 
        vehicle_x, vehicle_y, vehicle_psi, 
        result_x[i], result_y[i]);
  }
}

#endif
