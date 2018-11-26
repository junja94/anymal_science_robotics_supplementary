
// Eigen
#include <Eigen/Dense>
#include <Eigen/StdVector>

// commons
#include "raiCommon/enumeration.hpp"
#include <rai/Core>

// task
#include "quadrupedLocomotion/ANYmal_minimal.hpp"

// noise model
#include "rai/noiseModel/NormalDistributionNoise.hpp"

// algorithm
//#include <rai/common/Tensor.hpp>
#include "rai/function/tensorflow/StochasticPolicy_TensorFlow.hpp"

using namespace std;
using namespace Eigen;

/// learning states
using Dtype = float;

/// shortcuts
using rai::Task::ActionDim;
using rai::Task::StateDim;
using rai::Task::CommandDim;
using Task_ = rai::Task::ANYmal_minimal<Dtype>;

using State = Task_::State;
using Action = Task_::Action;
using Command =  Task_::Command;
using VectorXD = Task_::VectorXD;
using MatrixXD = Task_::MatrixXD;
typedef Eigen::Matrix<Dtype, ActionDim, ActionDim> NoiseCovariance;
using Noise = rai::Noise::Noise<Dtype, ActionDim>;
using NormNoise = rai::Noise::NormalDistributionNoise<Dtype, ActionDim>;
using NoiseCov = rai::Noise::NormalDistributionNoise<Dtype, ActionDim>::CovarianceMatrix;
//using RNNPolicy = rai::FuncApprox::RecurrentStochasticPolicyValue_Tensorflow<Dtype, StateDim, ActionDim>;
using Policy_TensorFlow = rai::FuncApprox::StochasticPolicy_TensorFlow<Dtype, StateDim, ActionDim>;

int main(int argc, char *argv[]) {

  rai::init();

  ////////////////////////// Define task ////////////////////////////
  std::string urdfPath;
  std::string actuatorPath;

  urdfPath +=
      "/home/joonho/workspace/oldrai/src/anymal_raisim/task/include/quadrupedLocomotion/model/robot_minimal.urdf";
  actuatorPath = "/home/joonho/workspace/oldrai/src/anymal_raisim/data/seaModel_10000.txt";

  Task_ task(true, 3, actuatorPath, urdfPath);

  task.setDiscountFactor(0.99);
//  task.setControlUpdate_dt(0.0025);
  task.setControlUpdate_dt(0.05);

  task.setTimeLimitPerEpisode(7.0);
  task.setRealTimeFactor(1.0);
  task.setNoiseFtr(1.0);

  Task_::GeneralizedCoordinate initialPos;
  Task_::GeneralizedVelocities initialVel;
  Task_::Vector12d jointTorque;

  initialPos << 0.0, 0.0, 0.14, 1.0, 0.0, 0.0, 0.0,
      0, M_PI / 2.0, -M_PI,
      0, M_PI / 2.0, -M_PI,
      0.0, -M_PI / 2.0, M_PI,
      0.0, -M_PI / 2.0, M_PI;
//
//    initialPos << 0.0, 0.0, 0.14,
//      0.5401, -0.8416, 0.0044, 0.0038,
//        0, 0.7, -1.0,
//        0, 0.7, -1.0,
//        0.0, -0.7, 1.0,
//        0.0, -0.7, 1.0;
//
//  initialPos << 0.0, 0.0, 0.14,
//      0.5401, 0.8353, 0.0, 0.0121,
//      0, 0.7, -1.0,
//      0, 0.7, -1.0,
//      0.0, -0.7, 1.0,
//      0.0, -0.7, 1.0;
//
  initialPos << 0.0, 0.0, 0.14,
      0.5401, -0.8353, 0.0, 0.0121,
      0, 0.4, -0.8,
      0, 0.4, -0.8,
      0.0, -0.4, 0.8,
      0.0, -0.4, 0.8;

  initialPos << 0.0, 0.0, 0.25,
      0.5184, 0.7746, 0.3107, 0.1867,
      -0.8009, 0.5985, -0.4031,
      -0.8960, -1.2083, 0.2099,
      0.0377, 0.7822, -0.8328,
      -0.0671, -0.3675, 0.7284;

  initialVel << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

  task.setInitialState(initialPos, initialVel);

  rai_sim::World_RG *env_ = task.getEnv();
  //noise
  NoiseCov covariance = NoiseCov::Identity();
  covariance *= 1.0;
  NormNoise actionNoise(covariance);

  /// Tensors
  State state_e;
  Task_::Action action_e;
  rai::Tensor<Dtype, 3> State3D({StateDim, 1, 1}, "state");
  rai::Tensor<Dtype, 3> Action3D({ActionDim, 1, 1}, "action");
  rai::Tensor<Dtype, 2> State2D({StateDim, 1}, "state");
  rai::Tensor<Dtype, 2> Action2D({ActionDim, 1}, "action");

  rai::Utils::logger->addVariableToLog(2, "Ncontacts", "");
  rai::Utils::logger->addVariableToLog(4, "speed", "");
  rai::Utils::logger->addVariableToLog(13, "command", "");

  rai::Utils::Graph::FigProp2D figprop;

  task.turnOnVisualization("");
  task.init0();

  task.setActionLimit(M_PI);

//  Policy_TensorFlow policy_("gpu,0", "MLP", "tanh 1e-3 96 128 128 12", 0.001);
//  policy_.loadParam("/home/joonho/workspace/oldrai/src/anymal_raisim/data/controller/recovery_policy_0.txt"); //0

  bool video = true;
  bool plot = false;
  bool stand = true;

  task.setFriction(1.5);
  int stableCounter = 0;
  int max = 50000;
  Eigen::Matrix<Dtype, -1, 1> stateSaveBuffer;
  Eigen::Matrix<Dtype, -1, 1> actionSaveBuffer;
  stateSaveBuffer.resize(96 * max);
  actionSaveBuffer.resize(12 * max);
  task.getState(state_e);

  Eigen::Matrix<double, 19, 1> defaultConfig;
  defaultConfig
      << 0.0, 0.0, 0.44, 1.0, 0.0, 0.0, 0.0, -0.15, 0.4, -0.8, 0.15, 0.4, -0.8, -0.15, -0.4, 0.8, 0.15, -0.4, 0.8;
  Eigen::Matrix<double, 18, 1> defaultVel;
  defaultVel.setZero();

  Eigen::Matrix<Dtype, 12, 1> jointNominalConfig;

  jointNominalConfig << 0.0, 0.4, -0.8, 0.0, 0.4, -0.8, 0.0, -0.4, 0.8, 0.0, -0.4, 0.8;

  Eigen::Matrix<Dtype, 12, 1> flipJointConfig;
  Eigen::Matrix<Dtype, 12, 1> flipJointConfig2;
  Eigen::Matrix<Dtype, 12, 1> sitJointConfig;
  Eigen::Matrix<Dtype, 12, 1> dampedJointPos_;
  dampedJointPos_ << 0.0, 0.4, -0.8, 0.0, 0.4, -0.8, 0.0, -0.4, 0.8, 0.0, -0.4, 0.8;
  sitJointConfig <<
                 -0.15, M_PI / 3.0, -2.,
      0.15, M_PI / 3.0, -2.,
      -0.15, -M_PI / 3.0, 2.,
      0.15, -M_PI / 3.0, 2.;

  flipJointConfig <<
                  0.0, 0.2, -0.3,
      0.5, 1.0, -2.45,
      0.0, -0.2, 0.3,
      0.5, -1.0, 2.45;

  flipJointConfig2 <<
                   0.0, 0.2, -0.3,
      0.5, 0.4, -0.8,
      0.0, -0.2, 0.3,
      0.5, -0.4, 0.8;

  task.setInitialState(defaultConfig, defaultVel);
  task.init0();

  for (int iterationNumber = 0; iterationNumber < max; iterationNumber++) {

    if (iterationNumber == 2 && video) task.startRecordingVideo(RAI_LOG_PATH, "locotest");

    rai::TerminationType type = rai::TerminationType::not_terminated;
    Dtype cost;

    //// forward
    State2D = state_e;

    stateSaveBuffer.segment(96 * iterationNumber, 96) = state_e;
    actionSaveBuffer.segment(12 * iterationNumber, 12) = action_e;

//    policy_.forward(State2D, Action2D);
//    action_e = Action2D.eMat();

    action_e.setZero();

    task.step(action_e, state_e, type, cost);

    if (env_->getEvent(rai_graphics::KeyboardEvent::O)) {
      task.noisifyDynamics();

    }
    /// Keyboard Inputs
    if (env_->getEvent(rai_graphics::KeyboardEvent::I)||type!=rai::TerminationType::not_terminated) {
      task.init();
      logger->clearData("speed");
      logger->clearData("Ncontacts");
      logger->clearData("command");
//      task.noisifyDynamics();

      RAIINFO("init");
      stableCounter = 0;
    }
    if (env_->getEvent(rai_graphics::KeyboardEvent::P)) {
//      task.noisifyDynamics();
      task.init0();
      logger->clearData("speed");
      logger->clearData("command");
      RAIINFO("init0");
      stableCounter = 0;

    }

    if (env_->getEvent(rai_graphics::KeyboardEvent::K)) {
      break;
    }

    if (plot && iterationNumber % 1 == 0) {
      graph->figure(2, figprop);
      graph->appendData(2, logger->getData("Ncontacts", 0),
                        logger->getData("Ncontacts", 1),
                        logger->getDataSize("Ncontacts"),
                        rai::Utils::Graph::PlotMethods2D::lines,
                        "Ncontacts",
                        "");
      graph->drawFigure(2, rai::Utils::Graph::OutputFormat::pdf);

//        graph->figure(0, figprop);
//        graph->figure(1, figprop);
//
//        graph->appendData(0, logger->getData("command", 0),
//                        logger->getData("command", 1),
//                       logger->getDataSize("command"),
//                       rai::Utils::Graph::PlotMethods2D::lines,
//                        "command0",
//                       "");
//      graph->appendData(0, logger->getData("command", 0),
//                        logger->getData("command", 2),
//                        logger->getDataSize("command"),
//                        rai::Utils::Graph::PlotMethods2D::lines,
//                        "command1",
//                        "");
//        graph->appendData(0, logger->getData("command", 0),
//                        logger->getData("command", 3),
//                        logger->getDataSize("command"),
//                        rai::Utils::Graph::PlotMethods2D::lines, "command2",
//                       "");
//        graph->appendData(0, logger->getData("command", 0),
//                          logger->getData("command", 4),
//                          logger->getDataSize("command"),
//                          rai::Utils::Graph::PlotMethods2D::lines,
//                          "command1",
//                          "");
//        graph->appendData(0, logger->getData("command", 0),
//                          logger->getData("command", 5),
//                          logger->getDataSize("command"),
//                          rai::Utils::Graph::PlotMethods2D::lines, "command2",
//                          "");
//        graph->appendData(0, logger->getData("command", 0),
//                          logger->getData("command", 6),
//                          logger->getDataSize("command"),
//                          rai::Utils::Graph::PlotMethods2D::lines, "command2",
//                          "");
//        graph->appendData(0, logger->getData("command", 0),
//                          logger->getData("command", 7),
//                          logger->getDataSize("command"),
//                          rai::Utils::Graph::PlotMethods2D::lines, "command2",
//                          "");
//        graph->appendData(0, logger->getData("command", 0),
//                          logger->getData("command", 8),
//                          logger->getDataSize("command"),
//                          rai::Utils::Graph::PlotMethods2D::lines, "command2",
//                          "");
//        graph->appendData(0, logger->getData("command", 0),
//                          logger->getData("command", 9),
//                          logger->getDataSize("command"),
//                          rai::Utils::Graph::PlotMethods2D::lines, "command2",
//                          "");
//        graph->appendData(0, logger->getData("command", 0),
//                          logger->getData("command", 10),
//                          logger->getDataSize("command"),
//                          rai::Utils::Graph::PlotMethods2D::lines, "command2",
//                          "");
//        graph->appendData(0, logger->getData("command", 0),
//                          logger->getData("command", 11),
//                          logger->getDataSize("command"),
//                          rai::Utils::Graph::PlotMethods2D::lines, "command2",
//                          "");
//
//        graph->appendData(1, logger->getData("speed", 0),
//                          logger->getData("speed", 1),
//                          logger->getDataSize("speed"),
//                          rai::Utils::Graph::PlotMethods2D::lines,
//                          "HAA0",
//                          "");
//        graph->appendData(1, logger->getData("speed", 0),
//                          logger->getData("speed", 2),
//                          logger->getDataSize("speed"),
//                          rai::Utils::Graph::PlotMethods2D::lines,
//                          "HFE0",
//                          "");
//        graph->appendData(1, logger->getData("speed", 0),
//                          logger->getData("speed", 3),
//                          logger->getDataSize("speed"),
//                          rai::Utils::Graph::PlotMethods2D::lines,
//                          "KFE0",
//                          "");
//        graph->drawFigure(0);
//        graph->drawFigure(1);
    }
//      task.init();
//      logger->clearData("cost");
//    }

//    usleep(0.0025 * 1e6);
  }

  rai::Utils::Graph::FigPropPieChart propChart;
  graph->drawPieChartWith_RAI_Timer(5, timer->getTimedItems(), propChart);
  graph->drawFigure(5, rai::Utils::Graph::OutputFormat::pdf);
  if (video) {
    task.endRecordingVideo();
    std::cout << "wait for video saving message and press enter" << std::endl;
    do {
    } while (cin.get() != '\n');
  } else {
    graph->waitForEnter();
  }

  return 0;

}