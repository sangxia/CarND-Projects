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
    vector<double> const &prev_path_x, vector<double> const &prev_path_y, 
    double start_s, double start_d,
    double start_x, double start_y, double start_theta, double start_speed,
    double target_time, double target_speed, double target_d, bool change_lane,
    vector<double> &trajectory_x, vector<double> &trajectory_y) {

  std::cout << "start_d " << start_d << " target_d " << target_d << std::endl;
  trajectory_x.clear();
  trajectory_y.clear();
  // if has previous trajectory, preserve the first few points
  // then generate new trajectory from the last point
  vector<double> target_x, target_y, target_t;
  vector<double> curr_p;
  curr_p.push_back(0.0);
  curr_p.push_back(0.0);
  double tmp_dist = 0.0; // accumulate trj. dist. to estimate time & know when to stop
  double curr_x = start_x;
  double curr_y = start_y;
  int prev_lim = min(30, int(prev_path_x.size())); // number of points to keep from previous trajectory
  double curr_time = 0.0;
  double prev_d = start_d;
  if (prev_path_x.size() > 0) {
    for (int i=0; i<prev_lim; i++) {
      curr_p = transformToEgo(start_x,start_y,start_theta,
          prev_path_x[i], prev_path_y[i]);
      tmp_dist += distance(prev_path_x[i], prev_path_y[i], curr_x, curr_y);
      if (i==0) {
        curr_time += tmp_dist / start_speed;
      } else {
        curr_time += 0.02;
      }
      // ignore points that are too close, so that latency does not
      // cause the generated trajectory to go backwards
      if (target_speed*1e-2 < tmp_dist) {
        target_x.push_back(curr_p[0]);
        target_y.push_back(curr_p[1]);
        target_t.push_back(curr_time);
      }
      curr_x = prev_path_x[i];
      curr_y = prev_path_y[i];
    }
    // this is useful if regenerating in the middle of lane change
    // perhaps not necessary for the current approach
    // std::cout << "prev_yaw " << atan2(prev_path_y[prev_lim-1]-prev_path_y[prev_lim-2],
    //       prev_path_x[prev_lim-1]-prev_path_x[prev_lim-2]) << std::endl;
    vector<double> tmp = getFrenet(curr_x,curr_y,
        atan2(prev_path_y[prev_lim-1]-prev_path_y[prev_lim-2],
          prev_path_x[prev_lim-1]-prev_path_x[prev_lim-2]),
        maps_x, maps_y);
    prev_d = tmp[1];
  } else {
    target_x.push_back(curr_p[0]);
    target_y.push_back(curr_p[1]);
    target_t.push_back(0.0);
  }

  // collect at least two more waypoints
  int min_target_size = target_x.size() + 2;
  // collect the next few waypoints until distance target is filled
  int wp = NextWaypoint(curr_x, curr_y, start_theta, maps_x, maps_y);
  vector<double> p;
  bool lane_changed = false; // decide which d to use when putting down trj. waypoints
  bool no_wait = change_lane && (fabs(prev_d-getLaneCenterByD(prev_d))>1.0);
  while (target_x.size() < min_target_size || tmp_dist < target_time*target_speed) {
    if (lane_changed) {
      p = getXY(maps_s[wp], getLaneCenterByD(target_d), maps_s, maps_x, maps_y);
    } else if (change_lane) {
      // the calculation of this part is not finished yet, see below
      std::cout << prev_d << std::endl;
      p = getXY(maps_s[wp], prev_d, maps_s, maps_x, maps_y);
    } else {
      p = getXY(maps_s[wp], getLaneCenterByD(start_d), maps_s, maps_x, maps_y);
    }
    double dst = distance(p[0], p[1], curr_x, curr_y);
    if (change_lane && (!lane_changed) && (dst>25 || no_wait)) {
      // only switch d if there has been enough distance
      p = getXY(maps_s[wp], getLaneCenterByD(target_d), maps_s, maps_x, maps_y);
      dst = distance(p[0], p[1], curr_x, curr_y);
      lane_changed = true;
    } 
    curr_p = transformToEgo(start_x,start_y,start_theta,p[0],p[1]);
    target_x.push_back(curr_p[0]);
    target_y.push_back(curr_p[1]);
    tmp_dist += dst;
    curr_time += dst/target_speed;
    target_t.push_back(curr_time);
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
  for (double t=target_t[0]; t<target_time; t+=0.02) {
    // start from the end of the previous trajectory
    curr_p = transformFromEgo(start_x,start_y,start_theta,sx(t),sy(t));
    trajectory_x.push_back(curr_p[0]);
    trajectory_y.push_back(curr_p[1]); 
  }
  std::cout << std::endl;

  std::cerr << "start point " << start_x << " " << start_y << " speed " << start_speed << std::endl;
  for (int i=0; i<min(200, int(trajectory_x.size())); i++) {
    std::cerr << trajectory_x[i] << " " << trajectory_y[i] << std::endl;
  }
  std::cerr << std::endl;
}

int scoreProposal(double car_s, double car_d, double car_speed,
    double track_len, int target_lane, bool check_behind,
    vector<vector<double>> const &sensor_fusion, double &proposed_speed) {
  double s_ahead, v_ahead, s_behind, v_behind;
  v_ahead = 21.5;
  s_ahead = 1e8;
  v_behind = 0.0;
  s_behind = 1e8;
  for (int i=0; i<sensor_fusion.size(); i++) {
    if (isInLane(sensor_fusion[i][6], target_lane, 0)) {
      double s=sensor_fusion[i][5];

      double ahead_dist=(s-car_s>=0)?(s-car_s):(s+track_len-car_s);
      if (ahead_dist<100 && ahead_dist<s_ahead) {
        s_ahead = ahead_dist;
        v_ahead = distance(sensor_fusion[i][3],sensor_fusion[i][4],0.0,0.0);
      }
      double behind_dist=(car_s-s>=0)?(car_s-s):(car_s+track_len-s);
      if (behind_dist<s_behind) {
        s_behind = behind_dist;
        v_behind = distance(sensor_fusion[i][3],sensor_fusion[i][4],0.0,0.0);
      }
    }
  }
  // -2 means reject, 2 means can accelerate, 1 means maintain speed, 0 and -1 means decc
  std::cout << "lane " << target_lane 
    << " s ahead " << s_ahead << " v ahead " << v_ahead
    << " s behind " << s_behind << " v behind " << v_behind << std::endl;
  proposed_speed = v_ahead;
  if (check_behind && ( s_ahead < 20 || s_behind < 5 || (v_behind-car_speed)*4.0>s_behind-5)) {
    // make sure we have space of 3 seconds
    return -2;
  }
  if ((car_speed-v_ahead)*4.0 > s_ahead-10) {
    return -1;
  }
  if (s_ahead >= 100 || (v_ahead >= car_speed && s_ahead>=40)) {
    return 2;
  }
  if (s_ahead >= 70) {
    return 1;
  }
  if (s_ahead >= 40) {
    return 0;
  } else {
    return -1;
  }
}

#endif
