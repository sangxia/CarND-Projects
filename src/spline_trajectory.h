#ifndef _SPLINE_TRAJECTORY_H_
#define _SPLINE_TRAJECTORY_H_

#include <math.h>
#include <vector>
#include "spline.h"
#include "road_geometry.h"

using namespace std;

void generateTrajectory(
    vector<double> &maps_s,
    vector<double> &maps_x, vector<double> &maps_y, 
    vector<double> &maps_dx, vector<double> &maps_dy, 
    vector<double> const &prev_path_x, vector<double> const &prev_path_y, 
    double start_s, double start_d,
    double start_x, double start_y, double start_theta, double start_speed,
    double target_time, double target_speed, 
    vector<double> &trajectory_x, vector<double> &trajectory_y) {

  trajectory_x.clear();
  trajectory_y.clear();

  int curr_d = 6.0; // TODO make this adaptive

  // if has previous trajectory, preserve the first few points
  // then generate new trajectory from the last point
  vector<double> target_x, target_y, target_t;
  vector<double> curr_p;
  curr_p.push_back(0.0);
  curr_p.push_back(0.0);
  double tmp_dist = 0.0;
  double curr_x = start_x;
  double curr_y = start_y;
  double dist = 0.0;
  int prev_lim = 20;
  if (prev_path_x.size() > 0) {
    for (int i=0; i<prev_lim; i++) {
      // trajectory_x.push_back(prev_path_x[i]);
      // trajectory_y.push_back(prev_path_y[i]);

      curr_p = transformToEgo(start_x,start_y,start_theta,
          prev_path_x[i], prev_path_y[i]);
      tmp_dist += distance(prev_path_x[i], prev_path_y[i], curr_x, curr_y);
      // ignore points that are too close, so that latency does not
      // cause the generated trajectory to go backwards
      if (target_speed*1e-2 < tmp_dist) {
        target_x.push_back(curr_p[0]);
        target_y.push_back(curr_p[1]);
        target_t.push_back(tmp_dist/target_speed);
      }
      curr_x = prev_path_x[i];
      curr_y = prev_path_y[i];
    }
  } else {
    target_x.push_back(curr_p[0]);
    target_y.push_back(curr_p[1]);
    target_t.push_back(0.0);
  }

  // collect at least two more waypoints
  int min_target_size = target_x.size() + 2;
  // collect the next few waypoints until distance target is filled
  int wp = NextWaypoint(curr_x, curr_y, start_theta, maps_x, maps_y);
  //std::cout << "next wp id " << wp << std::endl;
  while (target_x.size() < min_target_size || 
      tmp_dist < target_time*target_speed) {
    //std::cout << "next wp id " << wp << " s " << maps_s[wp] << std::endl;
    vector<double> p = getXY(maps_s[wp], curr_d, maps_s, maps_x, maps_y);
    //std::cout << p[0] << " " << p[1] << std::endl;
    double dst = distance(p[0], p[1], curr_x, curr_y);
    curr_p = transformToEgo(start_x,start_y,start_theta,p[0],p[1]);
    target_x.push_back(curr_p[0]);
    target_y.push_back(curr_p[1]);
    tmp_dist += dst;
    target_t.push_back(tmp_dist/target_speed);
    curr_x = p[0];
    curr_y = p[1];
    if (wp < maps_s.size()-1) {
      wp++;
    } else {
      wp = 0;
    }
  }
  tk::spline sx;
  sx.set_points(target_t, target_x);
  tk::spline sy;
  sy.set_points(target_t, target_y);
  for (double t=0.0; t<target_time; t+=0.02) {
  //for (double t=0.02*(min(int(prev_path_x.size()), prev_lim)); t<target_time; t+=0.02) {
    curr_p = transformFromEgo(start_x,start_y,start_theta,sx(t),sy(t));
    //std::cout << " " << curr_p[0] << " " << curr_p[1];
    trajectory_x.push_back(curr_p[0]);
    trajectory_y.push_back(curr_p[1]);
  }
  //std::cout << std::endl;
  //
  /*
  std::cout << "start " << start_x << " " << start_y << std::endl;
  for (int i=0; i<40; i++) {
    std::cout << trajectory_x[i] << " " << trajectory_y[i] << std::endl;
  }
  std::cout << std::endl;
  */
}

bool detectCollision(vector<vector<double>> const &obstacle_x, vector<vector<double>> const &obstacle_y,
    vector<double> const &trajectory_x, vector<double> const &trajectory_y, double time_limit, double tolerance=4) {
  int idx_t=0, idx_c=0;
  // std::cout << "obstacle size " << obstacle_x[0].size() << " traj size " << trajectory_x.size() << std::endl;
  // std::cout << "next point " << trajectory_x[0] << " " << trajectory_y[0] << std::endl;
  while (idx_t<trajectory_x.size() && idx_c<obstacle_x.size()) {
    double curr_x=trajectory_x[idx_t], curr_y=trajectory_y[idx_t];
    for (int i=0; i<obstacle_x[idx_c].size(); i++) {
      // std::cout << distance(curr_x,curr_y,obstacle_x[idx_c][i],obstacle_y[idx_c][i]) << std::endl;
      if (distance(curr_x,curr_y,obstacle_x[idx_c][i],obstacle_y[idx_c][i])<tolerance) {
        return true;
      }
    }
    idx_t += 10;
    idx_c += 1;
  }
  return false;
}

void generateObstacle(vector<vector<double>> const &sensor_fusion, double time_limit, 
    vector<vector<double>> &obstacle_x, vector<vector<double>> &obstacle_y) {
  for (int i=0; i<int(time_limit/0.2); i++) {
    obstacle_x[i].clear();
    obstacle_y[i].clear();
  }
  for (auto itr=sensor_fusion.begin(); itr != sensor_fusion.end(); itr++) {
    for (int i=0; i<int(time_limit/0.2); i++) {
      obstacle_x[i].push_back((*itr)[1]+i*0.2*(*itr)[3]);
      obstacle_y[i].push_back((*itr)[2]+i*0.2*(*itr)[4]);
      // std::cout << i << " " << (*itr)[1]+i*0.2*(*itr)[3] << " " << (*itr)[2]+i*0.2*(*itr)[4] << std::endl;
    }
  }
}

#endif
