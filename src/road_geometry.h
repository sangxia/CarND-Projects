#ifndef _ROAD_GEOMETRY_H_
#define _ROAD_GEOMETRY_H_

#include <math.h>
#include <vector>
#include <iostream>

using namespace std;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

double distance(double x1, double y1, double x2, double y2) {
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}

int ClosestWaypoint(double x, double y, vector<double> &maps_x, vector<double> &maps_y) {
	double closestLen = 100000; //large number
	int closestWaypoint = 0;
	for(int i = 0; i < maps_x.size(); i++) {
		double map_x = maps_x[i];
		double map_y = maps_y[i];
		double dist = distance(x,y,map_x,map_y);
		if(dist < closestLen) {
			closestLen = dist;
			closestWaypoint = i;
		}
	}
	return closestWaypoint;
}

int NextWaypoint(double x, double y, double theta, vector<double> &maps_x, vector<double> &maps_y) {
	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);
	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];
	double heading = atan2( (map_y-y),(map_x-x) );
	double angle = abs(theta-heading);
  if (angle > pi()) {
    // i think not including this is a bug in the official template
    angle = abs(2*pi()-angle);
  }
	if(angle > pi()/2) {
    std::cout << '+' << closestWaypoint << " " << heading << " " << theta << std::endl;
		closestWaypoint++;
	}
	return closestWaypoint;
}

int NextWaypoint(double s, vector<double> &maps_s) {
  // TODO deduct total length
  if (s >= maps_s[maps_s.size()-1]) {
    return 0;
  }
  int ret = 0;
  while (s >= maps_s[ret]) {
    ret++;
  }
  return ret;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, vector<double> &maps_x, vector<double> &maps_y) {
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);
	int prev_wp;
	if(next_wp == 0) {
		prev_wp  = maps_x.size()-1;
	} else {
	  prev_wp = next_wp-1;
  }
  // n: prev->next
	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
  // x: prev->current
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];
	// find the projection of x onto n
	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;
	double frenet_d = distance(x_x,x_y,proj_x,proj_y);
	//see if d value is positive or negative by comparing it to a center point
	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);
	if(centerToPos <= centerToRef) {
		frenet_d *= -1;
	}
	// calculate s value
	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++) {
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}
	frenet_s += distance(0,0,proj_x,proj_y);
	return {frenet_s,frenet_d};
}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, vector<double> &maps_s, vector<double> &maps_x, vector<double> &maps_y) {
	int prev_wp = -1;
	while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) )) {
		prev_wp++;
	}
	int wp2 = (prev_wp+1)%maps_x.size();
	double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
	// the x,y,s along the segment
	double seg_s = (s-maps_s[prev_wp]);
	double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
	double seg_y = maps_y[prev_wp]+seg_s*sin(heading);
	double perp_heading = heading-pi()/2;
	double x = seg_x + d*cos(perp_heading);
	double y = seg_y + d*sin(perp_heading);
	return {x,y};
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenetWithSpeed(double x, double y, double theta, double speed,
    vector<double> &maps_x, vector<double> &maps_y) {
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);
  std::cout << "next_wp" << next_wp << " dist " << distance(x,y,maps_x[next_wp],maps_y[next_wp]) << std::endl;
	int prev_wp;
	if(next_wp == 0) {
		prev_wp  = maps_x.size()-1;
	} else {
	  prev_wp = next_wp-1;
  }
  // n: prev->next
	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
  // x: prev->current
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];
  // next: prev->current + 0.01s
  double next_x = x_x + speed * cos(deg2rad(theta)) * 0.01;
  double next_y = x_y + speed * sin(deg2rad(theta)) * 0.01;
	// find the projection of x onto n
	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;
	double frenet_d = distance(x_x,x_y,proj_x,proj_y);
  std::cout << "proj " << proj_x << " " << proj_y << std::endl;
  //
	double proj_norm_next = (next_x*n_x+next_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x_next = proj_norm_next*n_x;
	double proj_y_next = proj_norm_next*n_y;
	double frenet_d_next = distance(next_x,next_y,proj_x_next,proj_y_next);
  double delta_s = distance(proj_x,proj_y,proj_x_next,proj_y_next) * 100;
  double delta_d = (frenet_d_next - frenet_d) * 100;
	//see if d value is positive or negative by comparing it to a center point
	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);
	if(centerToPos <= centerToRef) {
		frenet_d *= -1;
		delta_d *= -1;
	}
	// calculate s value
	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++) {
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}
	frenet_s += distance(0,0,proj_x,proj_y);
	return {frenet_s,frenet_d, delta_s, delta_d};
}

vector<double> getFrenetWithXYSpeed(double x, double y, double speed_x, double speed_y,
    vector<double> &maps_x, vector<double> &maps_y) {
  double speed = distance(speed_x, speed_y, 0, 0);
  double theta = atan2(speed_y, speed_x);
  return getFrenetWithSpeed(x, y, theta, speed, maps_x, maps_y);
}

#endif

