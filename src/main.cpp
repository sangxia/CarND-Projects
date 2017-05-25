#include <math.h>
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <uWS/uWS.h>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "polyutils.h"
#include "geoutils.h"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

int main() {
  uWS::Hub h;

  double ref_v = 30.0;
  double ref_cte = 0.0;
  double ref_epsi = 0.0;
  double actuator_delay = 0.1;
  double w_v = 1.0;
  double w_cte = 0.2;
  double w_epsi = 1.0;
  double w_delta = 1.0;
  double w_a = 1.0;
  double w_ddelta = 1.0;
  double w_da = 1.0;
  double time_discount = 0.9;
  size_t N = 20;
  double dt = 0.05;

  // MPC is initialized here!
  MPC mpc;
  mpc.init(ref_v, ref_cte, ref_epsi, actuator_delay, w_v, w_cte, w_epsi, w_delta, w_a, w_ddelta, w_da,
      time_discount, N, dt);

  int degree = 3;

  h.onMessage(static_cast<std::function<void(uWS::WebSocket<uWS::SERVER>*, char*, size_t, uWS::OpCode)>>(
      [&mpc, &degree](uWS::WebSocket<uWS::SERVER> *ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"];

          // convert reference waypoints to vehicle viewpoint
          vector<double> veh_ref_x;
          vector<double> veh_ref_y;
          global_to_vehicle(ptsx, ptsy, px, py, psi, veh_ref_x, veh_ref_y);
          Eigen::VectorXd coeffs = polyfit(
              Eigen::VectorXd::Map(veh_ref_x.data(), veh_ref_x.size()), 
              Eigen::VectorXd::Map(veh_ref_y.data(), veh_ref_y.size()), 
              degree);
          std::cout << coeffs << std::endl;
          // Eigen::VectorXd coeffs_d = get_poly_derivative(coeffs);
          Eigen::VectorXd state(6);
          state << 0., 0., 0., v, polyeval(coeffs, 0.), atan(coeffs(1));
          std::cout << state << std::endl;
          vector<double> sol = mpc.Solve(state, coeffs);
          std::cout << sol[0] << " " << sol[1] << std::endl;

          /*
          * TODO: Calculate steeering angle and throttle using MPC.
          *
          * Both are in between [-1, 1].
          *
          */
          double steer_value;
          double throttle_value;

          json msgJson;
          // TODO I don't understand why this needs to be negated
          msgJson["steering_angle"] = -sol[0];
          msgJson["throttle"] = sol[1];
//          msgJson["steering_angle"] = steer_value;
//          msgJson["throttle"] = throttle_value;

          //Display the MPC predicted trajectory 
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;
          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          vector<double> next_x_vals;
          vector<double> next_y_vals;

          next_x_vals = {-20, -10, -5, 0, 5, 10, 20};
          next_y_vals = {0, 0, 0, 0, 0, 0, 0};
          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line
//           for (int i=0; i<ptsx.size(); i++) {
//             double tempx, tempy;
//             global_to_vehicle(ptsx[i], ptsy[i], px, py, psi, tempx, tempy);
//             next_x_vals.push_back(tempx);
//             next_y_vals.push_back(tempy);
//           }

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;


          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws->send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws->send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  }));

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

  h.onConnection(static_cast<std::function<void(uWS::WebSocket<uWS::SERVER>*, uWS::HttpRequest)>>(
        [&h](uWS::WebSocket<uWS::SERVER> *ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  }));

  h.onDisconnection(static_cast<std::function<void(uWS::WebSocket<uWS::SERVER>*, int, char*, size_t)>>(
      [&h](uWS::WebSocket<uWS::SERVER> *ws, int code,
                         char *message, size_t length) {
    ws->close();
    std::cout << "Disconnected" << std::endl;
  }));

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
