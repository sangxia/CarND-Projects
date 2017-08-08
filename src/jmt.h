#ifndef _JMT_H_
#define _JMT_H_

#include <math.h>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
// NOTE including QR as in the given template doesn't work
#include "Eigen-3.3/Eigen/LU"
#include "road_geometry.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

vector<double> JMT(
    double start, double dstart, double ddstart, 
    double end, double dend, double ddend, double T) {
    /*
    Calculate the Jerk Minimizing Trajectory that connects the initial state
    to the final state in time T.

    INPUTS

    start - the vehicles start location given as a length three array
        corresponding to initial values of [s, s_dot, s_double_dot]

    end   - the desired end state for vehicle. Like "start" this is a
        length three array.

    T     - The duration, in seconds, over which this maneuver should occur.

    OUTPUT 
    an array of length 6, each value corresponding to a coefficent in the polynomial 
    s(t) = a_0 + a_1 * t + a_2 * t**2 + a_3 * t**3 + a_4 * t**4 + a_5 * t**5

    */
    
  MatrixXd A = MatrixXd(3, 3);
	A << T*T*T, T*T*T*T, T*T*T*T*T, 
    3*T*T, 4*T*T*T,5*T*T*T*T, 
    6*T, 12*T*T, 20*T*T*T;
	MatrixXd B = MatrixXd(3,1);	    
	B << end-(start+dstart*T+.5*ddstart*T*T), 
    dend-(dstart+ddstart*T), 
    ddend-ddstart;
	MatrixXd Ai = A.inverse();
	MatrixXd C = Ai*B;
	vector <double> result = {start, dstart, .5*ddstart};
	for(int i = 0; i < C.size(); i++) { 
    result.push_back(C.data()[i]);
	}
  return result;
}

double evalpoly(vector<double> p, double x) {
  double result = 0.0;
  for (int i=p.size()-1; i>=0; i--) {
    result = result*x + p[i];
  }
  return result;
}

void generateTrajectory(
    vector<double> &maps_s,
    vector<double> &maps_x, vector<double> &maps_y, 
    vector<double> const &prev_path_x, vector<double> const &prev_path_y, 
    double start_s, double start_d,
    double start_x, double start_y, double start_theta, double start_speed,
    double target_time, double target_speed, 
    vector<double> &trajectory_x, vector<double> &trajectory_y) {

  trajectory_x.clear();
  trajectory_y.clear();

  double start_speed_x = start_speed*cos(start_theta);
  double start_speed_y = start_speed*sin(start_theta);
  std::cout << "start " << start_x << " " << start_y << " " << start_speed_x << " " << start_speed_y << " " << start_theta << std::endl;

  int curr_d = 6.0; // TODO make this adaptive

  // if has previous trajectory, preserve the first few points
  // then generate new trajectory from the last point
  vector<double> target_x, target_y, target_t;
  double curr_x = start_x;
  double curr_y = start_y;
  double tmp_dist = 0.0;
  if (prev_path_x.size() > 0) {
    int lim = 8;
    for (int i=1; i<lim-1; i++) {
      trajectory_x.push_back(prev_path_x[i]);
      trajectory_y.push_back(prev_path_y[i]);
      tmp_dist += distance(prev_path_x[i], prev_path_y[i], curr_x, curr_y);
      curr_x = prev_path_x[i];
      curr_y = prev_path_y[i];
      std::cout << "trj " << prev_path_x[i] << " " << prev_path_y[i] << std::endl;
    }
    target_x.push_back(prev_path_x[lim-1]);
    target_y.push_back(prev_path_y[lim-1]);
    target_t.push_back((lim-1)*0.02);
    double last_speed = distance(prev_path_x[lim-2],prev_path_y[lim-2],prev_path_x[lim-1],prev_path_y[lim-1])/0.02;
    double last_yaw = atan2(prev_path_y[lim-1]-prev_path_y[lim-2],prev_path_x[lim-1]-prev_path_x[lim-2]);
    start_speed_x = last_speed*cos(last_yaw);
    start_speed_y = last_speed*sin(last_yaw);
  } else {
    target_x.push_back(curr_x);
    target_y.push_back(curr_y);
    target_t.push_back(0.0);
  }

  // collect the next few waypoints until distance target is filled
  int wp = NextWaypoint(curr_x, curr_y, start_theta, maps_x, maps_y);
  std::cout << "next wp id " << wp << std::endl;
  while (tmp_dist < target_time*target_speed) {
    vector<double> p = getXY(maps_s[wp], curr_d, maps_s, maps_x, maps_y);
    double dst = distance(p[0], p[1], curr_x, curr_y);
    if (tmp_dist+dst <= target_time*target_speed) {
      target_x.push_back(p[0]);
      target_y.push_back(p[1]);
      tmp_dist += dst;
      target_t.push_back(tmp_dist/target_speed);
      curr_x = p[0];
      curr_y = p[1];
    } else {
      double r = (target_time*target_speed-tmp_dist) / dst;
      double x = curr_x + (p[0]-curr_x)*r;
      double y = curr_y + (p[1]-curr_y)*r;
      target_x.push_back(x);
      target_y.push_back(y);
      target_t.push_back(target_time);
      break;
    }
    if (wp < maps_s.size()-1) {
      wp++;
    } else {
      wp = 0;
    }
  }
  std::cout << "targets: ";
  for (int i=0; i<target_x.size(); i++) {
    std::cout << "(" << target_x[i] << "," << target_y[i] << ")@" << target_t[i] << " ";
  }
  std::cout << std::endl;

  double ts = target_t[0];
  double ts_shift = 0.0;
  int tp = 0;
  double x=target_x[0]; 
  double dx=start_speed_x; 
  double ddx = 0.0;
  double y=target_y[0]; 
  double dy=start_speed_y; 
  double ddy = 0.0;
  double curr_yaw;
  if (start_speed > 1e-2) { 
    curr_yaw = atan2(dy,dx);
  } else {
    curr_yaw = start_theta;
  }
  std::cout << "curr yaw " << curr_yaw << std::endl;
  double nxt_yaw;
  vector<double> px, py;
  while (ts < target_time) {
    if (ts >= target_t[tp]) {
      ts_shift = ts - target_t[tp];
      std::cout << "ts " << ts << " " << target_t[tp] << std::endl;
      tp++;
      if (tp < target_x.size()-1) {
        nxt_yaw = atan2(target_y[tp+1]-target_y[tp], target_x[tp+1]-target_x[tp]);
      } else {
        std::cout << " reached target end " << std::endl;
        nxt_yaw = curr_yaw;
      }
      std::cout << "next yaw " << nxt_yaw << std::endl;
      double tx=target_x[tp];
      double dtx=target_speed*cos(nxt_yaw);
      double ddtx = 0.0;
      double ty=target_y[tp];
      double dty=target_speed*sin(nxt_yaw);
      double ddty = 0.0;
      std::cout << " gen " << tp << " " << x << " " << dx << " " << y << " " << dy << " " << tx << " " << dtx << " " << ty << " " << dty << std::endl;
      px = JMT(x,dx,ddx,tx,dtx,ddtx,target_t[tp]-ts);
      py = JMT(y,dy,ddy,ty,dty,ddty,target_t[tp]-ts);
      std::cout << "---" << std::endl;
      for (int i=0; i<px.size(); i++) {
        std::cout << " " << px[i];
      }
      std::cout << std::endl;
      for (int i=0; i<py.size(); i++) {
        std::cout << " " << py[i];
      }
      std::cout << std::endl;
      std::cout << "---" << std::endl;
      x=tx;
      dx=dtx;
      ddx=ddtx;
      y=ty;
      dy=dty;
      ddy=ddty;
      curr_yaw = nxt_yaw;
    }
    double traj_x=evalpoly(px, ts_shift);
    double traj_y=evalpoly(py, ts_shift);
    std::cout << "time " << ts << " " << traj_x << " " << traj_y << std::endl;
    trajectory_x.push_back(traj_x);
    trajectory_y.push_back(traj_y);
    ts_shift += 0.02;
    ts += 0.02;
  }
  std::cout << std::endl;
}

#endif
