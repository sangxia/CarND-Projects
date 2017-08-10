#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"

#include "road_geometry.h"
#include "spline_trajectory.h"
//#include "jmt.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// for convenience
using json = nlohmann::json;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double get_current_time() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock().now().time_since_epoch()).count();
}

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
  	istringstream iss(line);
  	double x;
  	double y;
  	float s;
  	float d_x;
  	float d_y;
  	iss >> x;
  	iss >> y;
  	iss >> s;
  	iss >> d_x;
  	iss >> d_y;
  	map_waypoints_x.push_back(x);
  	map_waypoints_y.push_back(y);
  	map_waypoints_s.push_back(s);
  	map_waypoints_dx.push_back(d_x);
  	map_waypoints_dy.push_back(d_y);
  }

  int map_waypoints_size = map_waypoints_s.size();
  double track_len = map_waypoints_s[map_waypoints_size-1] + 
    distance(map_waypoints_x[0],map_waypoints_y[0],map_waypoints_x[map_waypoints_size-1],map_waypoints_y[map_waypoints_size-1]);

  double target_speed = 0.0; 
  bool changing_lane = false;
  double last_lane_change = 0.0;
  int current_lane = -1, target_lane = 1;

  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy,&track_len,map_waypoints_size,
      &target_speed, &changing_lane, &last_lane_change, &current_lane, &target_lane](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
        	// Main car's localization Data
          	double car_x = j[1]["x"];
          	double car_y = j[1]["y"];
          	double car_s = j[1]["s"];
          	double car_d = j[1]["d"];
          	double car_yaw = j[1]["yaw"];
            car_yaw = deg2rad(car_yaw);
          	double car_speed = j[1]["speed"];
            car_speed = car_speed *1.61 / 3.6;
            if (current_lane<0) {
              current_lane = getLane(car_d);
            }


          	// Previous path data given to the Planner
          	auto previous_path_x = j[1]["previous_path_x"];
          	auto previous_path_y = j[1]["previous_path_y"];
          	// Previous path's end s and d values 
          	double end_path_s = j[1]["end_path_s"];
          	double end_path_d = j[1]["end_path_d"];

          	// Sensor Fusion Data, a list of all other cars on the same side of the road.
          	auto sensor_fusion = j[1]["sensor_fusion"];
            
          	json msgJson;
            
            if (previous_path_x.size() > 100) {
              msgJson["next_x"] = previous_path_x;
              msgJson["next_y"] = previous_path_y;
            } else {
              int nxt_wp = NextWaypoint(car_x, car_y, car_yaw, map_waypoints_x, map_waypoints_y);
              int prev_wp = (nxt_wp-1+map_waypoints_size) % map_waypoints_size;
              int nxt_wp2 = (nxt_wp+1) % map_waypoints_size;
              double dp = dotproduct(map_waypoints_x[prev_wp], map_waypoints_y[prev_wp], 
                  map_waypoints_x[nxt_wp], map_waypoints_y[nxt_wp], 
                  map_waypoints_x[nxt_wp2], map_waypoints_y[nxt_wp2]); 
              std::cout << "curve " << dp << std::endl;

              vector<double> next_x_vals;
              vector<double> next_y_vals;
              double current_time = get_current_time();

              // check if lane change has finished
              if (changing_lane) {
                if (fabs(car_d-getLaneCenterById(target_lane)) < 0.4) {
                  changing_lane = false;
                  current_lane = target_lane;
                  last_lane_change = current_time;
                }
              }

              double best_target_lane_speed, tmp_target_lane_speed;
              int best_lane = current_lane;
              int best_score, tmp_score;
              
              // calculate the score and best speed for the plan
              if (changing_lane) {
                best_score = scoreProposal(car_s,car_d,car_speed,track_len,target_lane,true,sensor_fusion,best_target_lane_speed);
              } else {
                best_score = scoreProposal(car_s,car_d,car_speed,track_len,current_lane,false,sensor_fusion,best_target_lane_speed);
              }

              double current_lane_speed = best_target_lane_speed;

              // only consider lane change if not in the middle of it and have not done so for a while and no sharp curve
              if (!changing_lane && current_time - last_lane_change > 10.0 && dp>0.992) {
                if (current_lane > 0) {
                  tmp_score = scoreProposal(car_s,car_d,car_speed,track_len,current_lane-1,true,sensor_fusion,tmp_target_lane_speed);
                  if (tmp_score>=0 && tmp_target_lane_speed>best_target_lane_speed && tmp_target_lane_speed>current_lane_speed+1) {
                    best_score = tmp_score;
                    best_target_lane_speed = tmp_target_lane_speed;
                    best_lane = current_lane-1;
                    target_lane = best_lane;
                    changing_lane = true;
                  }
                }
                if (current_lane < 2) {
                  tmp_score = scoreProposal(car_s,car_d,car_speed,track_len,current_lane+1,true,sensor_fusion,tmp_target_lane_speed);
                  if (tmp_score>=0 && tmp_target_lane_speed>best_target_lane_speed && tmp_target_lane_speed>current_lane_speed+1) {
                    best_score = tmp_score;
                    best_target_lane_speed = tmp_target_lane_speed;
                    best_lane = current_lane+1;
                    target_lane = best_lane;
                    changing_lane = true;
                  }
                }
              } else {
                std::cout << "not considering lane change because " << changing_lane << " " << current_time << " " << last_lane_change << std::endl;
              }

              if (best_score == 2) {
                target_speed = min(21.0, car_speed+1.0+0.8*(car_speed<5));
              } else if (best_score == 0) {
                target_speed = max(car_speed-0.8, 4.0);
              } else if (best_score == -1) {
                target_speed = max(car_speed-1.2, 4.0);
              } else if (best_score == -2) {
                target_speed = max(car_speed-2.5, 4.0);
              }

              std::cout << "car speed " << car_speed << " target speed " << target_speed << std::endl;
              std::cout << "current lane " << current_lane << " target lane " << target_lane << std::endl;
              std::cout << "best score " << best_score << " lane change " << changing_lane << std::endl;
              if (changing_lane) {
                generateTrajectory(map_waypoints_s, map_waypoints_x, map_waypoints_y, previous_path_x, previous_path_y,
                    car_s, car_d, car_x, car_y, car_yaw, car_speed, 4, target_speed, getLaneCenterById(target_lane), true, next_x_vals, next_y_vals);
              } else {
                generateTrajectory(map_waypoints_s, map_waypoints_x, map_waypoints_y, previous_path_x, previous_path_y,
                    car_s, car_d, car_x, car_y, car_yaw, car_speed, 4, target_speed, getLaneCenterById(current_lane), false, next_x_vals, next_y_vals);
              }

              /*
              if (car_speed<15) {
                generateTrajectory(map_waypoints_s, map_waypoints_x, map_waypoints_y, previous_path_x, previous_path_y,
                    car_s, car_d, car_x, car_y, car_yaw, car_speed, 4, target_speed, 6., fabs(car_d-6)>1.0, next_x_vals, next_y_vals);
              } else {
                generateTrajectory(map_waypoints_s, map_waypoints_x, map_waypoints_y, previous_path_x, previous_path_y,
                    car_s, car_d, car_x, car_y, car_yaw, car_speed, 4, target_speed, 2.2, fabs(car_d-2.2)>1.0, next_x_vals, next_y_vals);
              }
              */
              msgJson["next_x"] = next_x_vals;
              msgJson["next_y"] = next_y_vals;
            }

          	auto msg = "42[\"control\","+ msgJson.dump()+"]";

          	//this_thread::sleep_for(chrono::milliseconds(1000));
          	ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
