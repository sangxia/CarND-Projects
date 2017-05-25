#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
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

int readParams(char* fname, double &ref_v, double &ref_cte, double &ref_epsi,
    double &actuator_delay, double &w_v, double &w_cte, double &w_epsi,
    double &w_delta, double &w_a, double &w_ddelta, double &w_da,
    double &time_discount, size_t &N, double &dt, int &degree) {

  std::ifstream in_file(fname, std::ifstream::in);
  if (!in_file) {
    return -1;
  }
  std::string line;
  std::string pname;
  while (getline(in_file, line)) {
    std::istringstream iss(line);
    iss >> pname;
    if (pname == "#") {
      continue;
    } else if (pname == "ref_v") {
      iss >> ref_v;
      std::cout << pname << " " << ref_v << std::endl;
    } else if (pname == "ref_cte") {
      iss >> ref_cte;
      std::cout << pname << " " << ref_cte << std::endl;
    } else if (pname == "ref_epsi") {
      iss >> ref_epsi;
      std::cout << pname << " " << ref_epsi << std::endl;
    } else if (pname == "actuator_delay") {
      iss >> actuator_delay;
      std::cout << pname << " " << actuator_delay << std::endl;
    } else if (pname == "w_v") {
      iss >> w_v;
      std::cout << pname << " " << w_v << std::endl;
    } else if (pname == "w_cte") {
      iss >> w_cte;
      std::cout << pname << " " << w_cte << std::endl;
    } else if (pname == "w_epsi") {
      iss >> w_epsi;
      std::cout << pname << " " << w_epsi << std::endl;
    } else if (pname == "w_delta") {
      iss >> w_delta;
      std::cout << pname << " " << w_delta << std::endl;
    } else if (pname == "w_a") {
      iss >> w_a;
      std::cout << pname << " " << w_a << std::endl;
    } else if (pname == "w_ddelta") {
      iss >> w_ddelta;
      std::cout << pname << " " << w_ddelta << std::endl;
    } else if (pname == "w_da") {
      iss >> w_da;
      std::cout << pname << " " << w_da << std::endl;
    } else if (pname == "time_discount") {
      iss >> time_discount;
      std::cout << pname << " " << time_discount << std::endl;
    } else if (pname == "N") {
      iss >> N;
      std::cout << pname << " " << N << std::endl;
    } else if (pname == "dt") {
      iss >> dt;
      std::cout << pname << " " << dt << std::endl;
    } else if (pname == "degree") {
      iss >> degree;
      std::cout << pname << " " << degree << std::endl;
    } else {
      return -1;
    }
  }
  return 0;
}

int main(int argc, char* argv[]) {
  uWS::Hub h;

  double ref_v = 15.0;
  double ref_cte = 0.0;
  double ref_epsi = 0.0;
  double actuator_delay = 0.1;
  double w_v = 1.0;
  double w_cte = 0.3;
  double w_epsi = 1.0;
  double w_delta = 1.0;
  double w_a = 1.0;
  double w_ddelta = 1.0;
  double w_da = 1.0;
  double time_discount = 0.9;
  size_t N = 20;
  double dt = 0.05;
  int degree = 2;

  if (argc > 1) {
    int ret = readParams(argv[1], ref_v, ref_cte, ref_epsi, actuator_delay, w_v, w_cte,
        w_epsi, w_delta, w_a, w_ddelta, w_da, time_discount, N, dt, degree);
    if (ret != 0) {
      std::cout << "wrong parameters" << std::endl;
      return -1;
    }
  }

  MPC mpc;
  mpc.init(ref_v, ref_cte, ref_epsi, actuator_delay, w_v, w_cte, w_epsi, w_delta, w_a, w_ddelta, w_da,
      time_discount, N, dt);

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
          // Eigen::VectorXd coeffs_d = get_poly_derivative(coeffs);
          Eigen::VectorXd state(6);
          state << 0., 0., 0., v, polyeval(coeffs, 0.), atan(coeffs(1));
          double current_steer = j[1]["steering_angle"];
          double current_throttle = j[1]["throttle"];
          vector<double> sol = mpc.Solve(state, coeffs, -current_steer * (25.0/180.0*M_PI), 
              current_throttle);

          json msgJson;
          
          msgJson["steering_angle"] = -sol[0] / (25.0/180.0*M_PI);
          msgJson["throttle"] = sol[1];

          //Display the MPC predicted trajectory 
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;
          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          vector<double> next_x_vals;
          vector<double> next_y_vals;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line

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
