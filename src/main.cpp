#include <math.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <uWS/uWS.h>
#include "json.hpp"
#include "PID.h"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_last_of("]");
  if (found_null != std::string::npos) {
    return "";
  }
  else if (b1 != std::string::npos && b2 != std::string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

int readParams(char* fname, double &Kp, double &Ki, double &Kd, double &tol,
    double &speed_r, double &speed_max, double &const_throttle, double &brake, double &cruise_throttle,
    int &steer_hist_max, bool &verb) {
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
    } else if (pname == "Kp") {
      iss >> Kp;
      std::cout << pname << " " << Kp << std::endl;
    } else if (pname == "Ki") {
      iss >> Ki;
      std::cout << pname << " " << Ki << std::endl;
    } else if (pname == "Kd") {
      iss >> Kd;
      std::cout << pname << " " << Kd << std::endl;
    } else if (pname == "speed_r") {
      iss >> speed_r;
      std::cout << pname << " " << speed_r << std::endl;
    } else if (pname == "speed_max") {
      iss >> speed_max;
      std::cout << pname << " " << speed_max << std::endl;
    } else if (pname == "cruise") {
      iss >> cruise_throttle;
      std::cout << pname << " " << cruise_throttle << std::endl;
    } else if (pname == "const_throttle") {
      iss >> const_throttle;
      std::cout << pname << " " << const_throttle << std::endl;
    } else if (pname == "brake") {
      iss >> brake;
      std::cout << pname << " " << brake << std::endl;
    } else if (pname == "cte_tol") {
      iss >> tol;
      std::cout << pname << " " << tol << std::endl;
    } else if (pname == "verbose") {
      iss >> verb;
      std::cout << pname << " " << verb << std::endl;
    } else if (pname == "steer_hist") {
      iss >> steer_hist_max;
      std::cout << pname << " " << steer_hist_max << std::endl;
    } else {
      return -1;
    }
  }
  return 0;
}

int main(int argc, char* argv[])
{
  uWS::Hub h;

  PID pid;
  double Kp = 0.1;
  double Ki = 0.00001;
  double Kd = 2;
  double speed_r = 20;
  double speed_max = 60;
  double const_throttle = 1;
  double brake = -0.5;
  double cruise_throttle = 0.6;
  double tol = 0.3;
  int steer_hist_max = 1;
  bool verb = false;

  if (argc > 1) {
    int ret = readParams(argv[1], Kp, Ki, Kd, tol, 
        speed_r, speed_max, const_throttle, brake, cruise_throttle, steer_hist_max, verb);
    if (ret != 0) {
      std::cout << "wrong parameters" << std::endl;
      return -1;
    }
  }
  pid.init(Kp, Ki, Kd, tol, speed_r, speed_max, 
      const_throttle, brake, cruise_throttle, steer_hist_max, verb);

  double topspeed = 0.0;
  h.onMessage(static_cast<std::function<void(uWS::WebSocket<uWS::SERVER>*, char*, size_t, uWS::OpCode)>>(
      [&pid, &topspeed](uWS::WebSocket<uWS::SERVER> *ws, char *data, size_t length, uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2')
    {
      auto s = hasData(std::string(data).substr(0, length));
      if (s != "") {
        auto j = json::parse(s);
        std::string event = j[0].get<std::string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          double cte = std::stod(j[1]["cte"].get<std::string>());
          double speed = std::stod(j[1]["speed"].get<std::string>());
          double angle = std::stod(j[1]["steering_angle"].get<std::string>());
          double steer_value;
          double throttle;
          pid.updateError(cte, speed);
          steer_value = pid.steer_ctrl;
          throttle = pid.throttle_ctrl;

          if (speed > topspeed) {
            topspeed = speed;
            std::cout << "NEW TOP SPEED " << speed << std::endl;
          }
          
          // DEBUG
          if (pid.verbose) {
            std::cout << "CTE:\t" << cte << "\tD:\t" << pid.d_error << "\tI:\t" << pid.i_error
              << "\tSteer:\t" << steer_value << std::endl;
          }

          json msgJson;
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle;
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          // std::cout << msg << std::endl;
          ws->send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws->send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  }));

  // We don't need this since we're not using HTTP but if it's removed the program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1)
    {
      res->end(s.data(), s.length());
    }
    else
    {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection(static_cast<std::function<void(uWS::WebSocket<uWS::SERVER>*, uWS::HttpRequest)>>(
        [&h](uWS::WebSocket<uWS::SERVER> *ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  }));

  h.onDisconnection(static_cast<std::function<void(uWS::WebSocket<uWS::SERVER>*, int, char*, size_t)>>(
        [&h](uWS::WebSocket<uWS::SERVER> *ws, int code, char *message, size_t length) {
    ws->close();
    std::cout << "Disconnected" << std::endl;
  }));

  int port = 4567;
  if (h.listen(port))
  {
    std::cout << "Listening to port " << port << std::endl;
  }
  else
  {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
