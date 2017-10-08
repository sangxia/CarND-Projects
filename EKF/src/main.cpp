#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <uWS/uWS.h>
#include "json.hpp"
#include "Eigen/Dense"
#include "FusionEKF.h"
#include "ground_truth_package.h"
#include "measurement_package.h"
#include "tools.h"

enum class Modes { File, Socket };

struct Param {
  Modes mode;
  bool Radar;
  bool Lidar;
  string infile;
  string outfile;
};

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

// for convenience
using json = nlohmann::json;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("]");
  if (found_null != std::string::npos) {
    return "";
  } else if (b1 != std::string::npos && b2 != std::string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

Param check_arguments(int argc, char* argv[]) {
  string usage_instructions = "Usage instructions: \n";
  usage_instructions += argv[0];
  usage_instructions += " file [-l] [-r] path/to/input.txt path/to/output.txt\n";
  usage_instructions += argv[0];
  usage_instructions += " socket\n";

  bool has_valid_args = false;
  Param ret;
  ret.Radar = true;
  ret.Lidar = true;

  // make sure the user has provided input and output files
  if (argc == 1) {
    cerr << usage_instructions << endl;
  } else {
    if (strcmp(argv[1], "file")==0) {
      if (argc < 4) {
        cerr << "Missing parameters.\n" << usage_instructions << endl;
      } else {
        if (strcmp(argv[2], "-l")==0 || strcmp(argv[3], "-l")==0) {
          cout << "Lidar off" << endl;
          ret.Lidar = false;
        }
        if (strcmp(argv[2], "-r")==0 || strcmp(argv[3], "-r")==0) {
          cout << "Radar off" << endl;
          ret.Radar = false;
        }
        if (!(ret.Lidar || ret.Radar)) {
          cerr << "Lidar and Radar cannot both be off.\n" << endl;
        }
        else {
          int n_switches = 2 - (ret.Lidar + ret.Radar);
          if (argc < 4+n_switches) {
            cerr << "Missing parameters.\n" << usage_instructions << endl;
          }
          else if (argc > 4+n_switches) {
            cerr << "Too many arguments.\n" << usage_instructions << endl;
          } else {
            has_valid_args = true;
            ret.mode = Modes::File;
            ret.infile = argv[2+n_switches];
            ret.outfile = argv[3+n_switches];
          }
        }
      }
    } else if (strcmp(argv[1], "socket") == 0) {
      if (argc == 2) {
        has_valid_args = true;
        ret.mode = Modes::Socket;
      } else {
        cerr << "Too many arguments.\n" << usage_instructions << endl;
      }
    }
    else {
      cerr << "Please specify either file or socket.\n" << usage_instructions <<endl;
    }
  }
  if (!has_valid_args) {
    exit(EXIT_FAILURE);
  }
  return ret;
}

void check_files(ifstream& in_file, string& in_name,
                 ofstream& out_file, string& out_name) {
  if (!in_file.is_open()) {
    cerr << "Cannot open input file: " << in_name << endl;
    exit(EXIT_FAILURE);
  }
  if (!out_file.is_open()) {
    cerr << "Cannot open output file: " << out_name << endl;
    exit(EXIT_FAILURE);
  }
}

void processFile(string in_file_name_, string out_file_name_, 
                 bool useLidar, bool useRadar) {
  ifstream in_file_(in_file_name_.c_str(), ifstream::in);
  ofstream out_file_(out_file_name_.c_str(), ofstream::out);
  check_files(in_file_, in_file_name_, out_file_, out_file_name_);

  vector<MeasurementPackage> measurement_pack_list;
  vector<GroundTruthPackage> gt_pack_list;
  string line;

  // prep the measurement packages (each line represents a measurement at a
  // timestamp)
  while (getline(in_file_, line)) {
    string sensor_type;
    MeasurementPackage meas_package;
    GroundTruthPackage gt_package;
    istringstream iss(line);
    long long timestamp;
    bool push = false;

    // reads first element from the current line
    iss >> sensor_type;
    if (sensor_type.compare("L") == 0) {
      // LASER MEASUREMENT
      // read measurements at this timestamp
      meas_package.sensor_type_ = MeasurementPackage::LASER;
      meas_package.raw_measurements_ = VectorXd(2);
      float x;
      float y;
      iss >> x;
      iss >> y;
      meas_package.raw_measurements_ << x, y;
      iss >> timestamp;
      meas_package.timestamp_ = timestamp;
      if (useLidar) {
        measurement_pack_list.push_back(meas_package);
        push = true;
      }
    } else if (sensor_type.compare("R") == 0) {
      // RADAR MEASUREMENT
      // read measurements at this timestamp
      meas_package.sensor_type_ = MeasurementPackage::RADAR;
      meas_package.raw_measurements_ = VectorXd(3);
      float ro;
      float phi;
      float ro_dot;
      iss >> ro;
      iss >> phi;
      iss >> ro_dot;
      meas_package.raw_measurements_ << ro, phi, ro_dot;
      iss >> timestamp;
      meas_package.timestamp_ = timestamp;
      if (useRadar) {
        measurement_pack_list.push_back(meas_package);
        push = true;
      }
    }
    // read ground truth data to compare later
    float x_gt;
    float y_gt;
    float vx_gt;
    float vy_gt;
    iss >> x_gt;
    iss >> y_gt;
    iss >> vx_gt;
    iss >> vy_gt;
    gt_package.gt_values_ = VectorXd(4);
    gt_package.gt_values_ << x_gt, y_gt, vx_gt, vy_gt;
    if (push) {
      gt_pack_list.push_back(gt_package);
    }
  }

  // Create a Fusion EKF instance
  FusionEKF fusionEKF;
  // used to compute the RMSE later
  vector<VectorXd> estimations;
  vector<VectorXd> ground_truth;
  //Call the EKF-based fusion
  size_t N = measurement_pack_list.size();
  for (size_t k = 0; k < N; ++k) {
    // start filtering from the second frame (the speed is unknown in the first
    // frame)
    fusionEKF.ProcessMeasurement(measurement_pack_list[k]);
    // output the estimation
    out_file_ << fusionEKF.ekf_.x_(0) << "\t";
    out_file_ << fusionEKF.ekf_.x_(1) << "\t";
    out_file_ << fusionEKF.ekf_.x_(2) << "\t";
    out_file_ << fusionEKF.ekf_.x_(3) << "\t";
    // output the measurements
    if (measurement_pack_list[k].sensor_type_ == MeasurementPackage::LASER) {
      // output the estimation
      out_file_ << measurement_pack_list[k].raw_measurements_(0) << "\t";
      out_file_ << measurement_pack_list[k].raw_measurements_(1) << "\t";
    } else if (measurement_pack_list[k].sensor_type_ == MeasurementPackage::RADAR) {
      // output the estimation in the cartesian coordinates
      float ro = measurement_pack_list[k].raw_measurements_(0);
      float phi = measurement_pack_list[k].raw_measurements_(1);
      out_file_ << ro * cos(phi) << "\t"; // p1_meas
      out_file_ << ro * sin(phi) << "\t"; // ps_meas
    }
    // output the ground truth packages
    out_file_ << gt_pack_list[k].gt_values_(0) << "\t";
    out_file_ << gt_pack_list[k].gt_values_(1) << "\t";
    out_file_ << gt_pack_list[k].gt_values_(2) << "\t";
    out_file_ << gt_pack_list[k].gt_values_(3) << "\n";
    estimations.push_back(fusionEKF.ekf_.x_);
    ground_truth.push_back(gt_pack_list[k].gt_values_);
  }
  // compute the accuracy (RMSE)
  Tools tools;
  cout << "Accuracy - RMSE:" << endl << tools.CalculateRMSE(estimations, ground_truth) << endl;
  cout << "Accuracy 2 - RMSE:" << endl << tools.CalculateRMSE(estimations, ground_truth, 20) << endl;
  // close files
  if (out_file_.is_open()) {
    out_file_.close();
  }
  if (in_file_.is_open()) {
    in_file_.close();
  }
  return;
}

void processSocket() {
  uWS::Hub h;
  // Create a Kalman Filter instance
  FusionEKF fusionEKF;
  // used to compute the RMSE later
  Tools tools;
  vector<VectorXd> estimations;
  vector<VectorXd> ground_truth;

  h.onMessage(static_cast<std::function<void(uWS::WebSocket<uWS::SERVER>*, char*, long unsigned int, uWS::OpCode)>>(
      [&fusionEKF,&tools,&estimations,&ground_truth](uWS::WebSocket<uWS::SERVER> *ws, char *data, size_t length, uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event

    if (length && length > 2 && data[0] == '4' && data[1] == '2') {
      auto s = hasData(std::string(data));
      if (s != "") {
        auto j = json::parse(s);
        std::string event = j[0].get<std::string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          string sensor_measurment = j[1]["sensor_measurement"];
          MeasurementPackage meas_package;
          istringstream iss(sensor_measurment);
          long long timestamp;
          // reads first element from the current line
          string sensor_type;
          iss >> sensor_type;
          if (sensor_type.compare("L") == 0) {
            meas_package.sensor_type_ = MeasurementPackage::LASER;
            meas_package.raw_measurements_ = VectorXd(2);
            float px;
            float py;
            iss >> px;
            iss >> py;
            meas_package.raw_measurements_ << px, py;
            iss >> timestamp;
            meas_package.timestamp_ = timestamp;
          } else if (sensor_type.compare("R") == 0) {
            meas_package.sensor_type_ = MeasurementPackage::RADAR;
            meas_package.raw_measurements_ = VectorXd(3);
            float ro;
            float theta;
            float ro_dot;
            iss >> ro;
            iss >> theta;
            iss >> ro_dot;
            meas_package.raw_measurements_ << ro,theta, ro_dot;
            iss >> timestamp;
            meas_package.timestamp_ = timestamp;
          }
          // reset the EKF after pause, otherwise time delta will be very large
          // and would result in a large initial error
          if (abs(meas_package.timestamp_ - fusionEKF.getPreviousTimestamp()) > 1e6) {
            fusionEKF.Reset();
            estimations.clear();
            ground_truth.clear();
          }
          float x_gt;
          float y_gt;
          float vx_gt;
          float vy_gt;
          iss >> x_gt;
          iss >> y_gt;
          iss >> vx_gt;
          iss >> vy_gt;
          VectorXd gt_values(4);
          gt_values(0) = x_gt;
          gt_values(1) = y_gt; 
          gt_values(2) = vx_gt;
          gt_values(3) = vy_gt;
          ground_truth.push_back(gt_values);
          // Call ProcessMeasurment(meas_package) for Kalman filter
          fusionEKF.ProcessMeasurement(meas_package);          

          //Push the current estimated x,y positon from the Klaman filter's state vector
          VectorXd estimate(4);
          double p_x = fusionEKF.ekf_.x_(0);
          double p_y = fusionEKF.ekf_.x_(1);
          double v1  = fusionEKF.ekf_.x_(2);
          double v2 = fusionEKF.ekf_.x_(3);
          estimate(0) = p_x;
          estimate(1) = p_y;
          estimate(2) = v1;
          estimate(3) = v2;
          estimations.push_back(estimate);

          VectorXd RMSE = tools.CalculateRMSE(estimations, ground_truth);
          json msgJson;
          msgJson["estimate_x"] = p_x;
          msgJson["estimate_y"] = p_y;
          msgJson["rmse_x"] =  RMSE(0);
          msgJson["rmse_y"] =  RMSE(1);
          msgJson["rmse_vx"] = RMSE(2);
          msgJson["rmse_vy"] = RMSE(3);
          auto msg = "42[\"estimate_marker\"," + msgJson.dump() + "]";
          // std::cout << msg << std::endl;
          ws->send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
        else {
          cout << event << endl;
        }
      } else {
        std::string msg = "42[\"manual\",{}]";
        ws->send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  }));

  // We don't need this since we're not using HTTP but if it's removed the program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) {
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
      [&h](uWS::WebSocket<uWS::SERVER> *ws, int code, char *message, size_t length) {
    ws->close();
    std::cout << "Disconnected" << std::endl;
  }));

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return;
  }
  h.run();
}

int main(int argc, char* argv[]) {
  Param param = check_arguments(argc, argv);
  if (param.mode == Modes::File) {
    processFile(param.infile, param.outfile, param.Lidar, param.Radar);
  } else if (param.mode == Modes::Socket) {
    processSocket();
  }
  return 0;
}
