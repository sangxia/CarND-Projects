
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include "Eigen/Dense"
#include "ukf.h"
#include "ground_truth_package.h"
#include "measurement_package.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

struct Param {
  bool Radar;
  bool Lidar;
  string infile;
  string outfile;
};

Param check_arguments(int argc, char* argv[]) {
  string usage_instructions = "Usage instructions: \n";
  usage_instructions += argv[0];
  usage_instructions += " [-l] [-r] path/to/input.txt path/to/output.txt";

  bool has_valid_args = false;
  Param ret;
  ret.Radar = true;
  ret.Lidar = true;

  // make sure the user has provided input and output files
  if (argc < 3) {
    cerr << "Missing arguments.\n" << usage_instructions << endl;
  } else {
    if (strcmp(argv[1], "-l")==0 || strcmp(argv[2], "-l")==0) {
      cout << "Lidar off" << endl;
      ret.Lidar = false;
    }
    if (strcmp(argv[1], "-r")==0 || strcmp(argv[2], "-r")==0) {
      cout << "Radar off" << endl;
      ret.Radar = false;
    }
    if (!(ret.Lidar || ret.Radar)) {
      cerr << "Lidar and Radar cannot both be off.\n" << endl;
    } else {
      int n_switches = 2 - (ret.Lidar + ret.Radar);
      if (argc < 3+n_switches) {
        cerr << "Missing arguments.\n" << usage_instructions << endl;
      } else if (argc > 3+n_switches) {
        cerr << "Too many arguments.\n" << usage_instructions << endl;
      } else {
        has_valid_args = true;
        ret.infile = argv[1+n_switches];
        ret.outfile = argv[2+n_switches];
      }
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

int main(int argc, char* argv[]) {

  Param param = check_arguments(argc, argv);

  string in_file_name_ = param.infile;
  ifstream in_file_(in_file_name_.c_str(), ifstream::in);

  string out_file_name_ = param.outfile;
  ofstream out_file_(out_file_name_.c_str(), ofstream::out);

  check_files(in_file_, in_file_name_, out_file_, out_file_name_);

  /**********************************************
   *  Set Measurements                          *
   **********************************************/

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

    // reads first element from the current line
    iss >> sensor_type;

    bool pushgt = false;

    if (sensor_type.compare("L") == 0) {
      // laser measurement

      // read measurements at this timestamp
      meas_package.sensor_type_ = MeasurementPackage::LASER;
      meas_package.raw_measurements_ = VectorXd(2);
      float px;
      float py;
      iss >> px;
      iss >> py;
      meas_package.raw_measurements_ << px, py;
      iss >> timestamp;
      meas_package.timestamp_ = timestamp;
      if (param.Lidar) {
        measurement_pack_list.push_back(meas_package);
        pushgt = true;
      }
    } else if (sensor_type.compare("R") == 0) {
      // radar measurement

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
      if (param.Radar) {
        measurement_pack_list.push_back(meas_package);
        pushgt = true;
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
      if (pushgt) {
        gt_pack_list.push_back(gt_package);
      }
  }

  // Create a UKF instance
  UKF ukf;

  // used to compute the RMSE later
  vector<VectorXd> estimations;
  vector<VectorXd> ground_truth;

  // start filtering from the second frame (the speed is unknown in the first
  // frame)

  size_t number_of_measurements = measurement_pack_list.size();

  // column names for output file
  out_file_ << "time_stamp" << "\t";  
  out_file_ << "px_state" << "\t";
  out_file_ << "py_state" << "\t";
  out_file_ << "v_state" << "\t";
  out_file_ << "yaw_angle_state" << "\t";
  out_file_ << "yaw_rate_state" << "\t";
  out_file_ << "sensor_type" << "\t";
  out_file_ << "NIS" << "\t";  
  out_file_ << "px_measured" << "\t";
  out_file_ << "py_measured" << "\t";
  out_file_ << "px_ground_truth" << "\t";
  out_file_ << "py_ground_truth" << "\t";
  out_file_ << "vx_ground_truth" << "\t";
  out_file_ << "vy_ground_truth" << "\n";


  for (size_t k = 0; k < number_of_measurements; ++k) {
    // Call the UKF-based fusion
    ukf.ProcessMeasurement(measurement_pack_list[k]);

    // timestamp
    out_file_ << measurement_pack_list[k].timestamp_ << "\t"; // pos1 - est

    // output the state vector
    out_file_ << ukf.x_(0) << "\t"; // pos1 - est
    out_file_ << ukf.x_(1) << "\t"; // pos2 - est
    out_file_ << ukf.x_(2) << "\t"; // vel_abs -est
    out_file_ << ukf.x_(3) << "\t"; // yaw_angle -est
    out_file_ << ukf.x_(4) << "\t"; // yaw_rate -est

    // output lidar and radar specific data
    if (measurement_pack_list[k].sensor_type_ == MeasurementPackage::LASER) {
      // sensor type
      out_file_ << "lidar" << "\t";

      // NIS value
      out_file_ << ukf.NIS_laser_ << "\t";

      // output the lidar sensor measurement px and py
      out_file_ << measurement_pack_list[k].raw_measurements_(0) << "\t";
      out_file_ << measurement_pack_list[k].raw_measurements_(1) << "\t";

    } else if (measurement_pack_list[k].sensor_type_ == MeasurementPackage::RADAR) {
      // sensor type
      out_file_ << "radar" << "\t";

      // NIS value
      out_file_ << ukf.NIS_radar_ << "\t";

      // output radar measurement in cartesian coordinates
      float ro = measurement_pack_list[k].raw_measurements_(0);
      float phi = measurement_pack_list[k].raw_measurements_(1);
      out_file_ << ro * cos(phi) << "\t"; // px measurement
      out_file_ << ro * sin(phi) << "\t"; // py measurement
    }

    // output the ground truth
    out_file_ << gt_pack_list[k].gt_values_(0) << "\t";
    out_file_ << gt_pack_list[k].gt_values_(1) << "\t";
    out_file_ << gt_pack_list[k].gt_values_(2) << "\t";
    out_file_ << gt_pack_list[k].gt_values_(3) << "\n";

    // convert ukf x vector to cartesian to compare to ground truth
    VectorXd ukf_x_cartesian_ = VectorXd(4);

    float x_estimate_ = ukf.x_(0);
    float y_estimate_ = ukf.x_(1);
    float vx_estimate_ = ukf.x_(2) * cos(ukf.x_(3));
    float vy_estimate_ = ukf.x_(2) * sin(ukf.x_(3));
    
    ukf_x_cartesian_ << x_estimate_, y_estimate_, vx_estimate_, vy_estimate_;
    
    estimations.push_back(ukf_x_cartesian_);
    ground_truth.push_back(gt_pack_list[k].gt_values_);

  }

  // compute the accuracy (RMSE)
  Tools tools;
  cout << "RMSE" << endl << tools.CalculateRMSE(estimations, ground_truth) << endl;

  // close files
  if (out_file_.is_open()) {
    out_file_.close();
  }

  if (in_file_.is_open()) {
    in_file_.close();
  }

  cout << "Done!" << endl;
  return 0;
}
