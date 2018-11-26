/*
 * master.cpp
 *
 *  Created on: Mar 7, 2016
 *      Author: jemin
 *
 *
 *											   generalized coordinates
 *  Note"
 *	1. Visualize with a single CPU only
 *
 */

// Eigen
#include <Eigen/Dense>
#include <Eigen/StdVector>

// commons
#include "rai/common/enumeration.hpp"
#include <rai/Core>
// task
#include "quadrupedLocomotion/ANYmal_full_collision.hpp"
// noise model
#include "rai/noiseModel/NormalDistributionNoise.hpp"

// Neural network
#include "rai/function/tensorflow/StochasticPolicy_TensorFlow.hpp"
#include "rai/function/tensorflow/ValueFunction_TensorFlow.hpp"

// algorithm
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_Sequential.hpp>
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_Parallel.hpp>
#include <rai/algorithm/TRPO_gae.hpp>

// filesystem
#include <experimental/filesystem>

using namespace std;
using namespace Eigen;

/// learning states
using Dtype = float;

/// shortcuts
using rai::Task::ActionDim;
using rai::Task::StateDim;
using rai::Task::CommandDim;
using Task_ = rai::Task::ANYmal_full_collision<Dtype>;

using State = Task_::State;
using Action = Task_::Action;
using Command =  Task_::Command;
using VectorXD = Task_::VectorXD;
using MatrixXD = Task_::MatrixXD;
typedef Eigen::Matrix<Dtype, ActionDim, ActionDim> NoiseCovariance;
using Policy_TensorFlow = rai::FuncApprox::StochasticPolicy_TensorFlow<Dtype, StateDim, ActionDim>;
using Vfunction_TensorFlow = rai::FuncApprox::ValueFunction_TensorFlow<Dtype, StateDim>;
using ReplayMemorySARS = rai::Memory::ReplayMemorySARS<Dtype, StateDim, ActionDim>;
using Acquisitor_ = rai::ExpAcq::TrajectoryAcquisitor_Parallel<Dtype, StateDim, ActionDim>;
using Noise = rai::Noise::Noise<Dtype, ActionDim>;
using NormNoise = rai::Noise::NormalDistributionNoise<Dtype, ActionDim>;
using NoiseCov = rai::Noise::NormalDistributionNoise<Dtype, ActionDim>::CovarianceMatrix;

#define nThread 40

int main(int argc, char *argv[]) {

  rai::init();
  omp_set_dynamic(0);
  omp_set_num_threads(nThread);

  RAIWARN(omp_get_max_threads())
  ////////////////////////// Define task ////////////////////////////
  std::string urdfPath;
  std::string actuatorPath;

  urdfPath += "/home/joonho/workspace/oldrai/src/anymal_raisim/task/include/quadrupedLocomotion/model/robot_nofan_nolimit.urdf";
  actuatorPath = "/home/joonho/workspace/oldrai/src/anymal_raisim/data/seaModel_10000.txt";

  std::vector<std::unique_ptr<Task_>> taskVec;

  std::unique_ptr<Task_> temp(new Task_(0, true, 0, actuatorPath, urdfPath));
  taskVec.emplace_back(std::move(temp)); /// turn on visualization only for the first thread

  for (int i = 1; i < nThread; i++) {
    std::unique_ptr<Task_> temp(new Task_(0, false, i % 10, actuatorPath, urdfPath));
    taskVec.emplace_back(std::move(temp));
  }


  std::vector<rai::Task::Task<Dtype, StateDim, ActionDim, 0> *> taskVector;
  for (auto &task : taskVec) {
    task->setDiscountFactor(0.993);
    task->setValueAtTerminalState(0);
    task->setControlUpdate_dt(0.05);
    task->setTimeLimitPerEpisode(6.0);
    task->setRealTimeFactor(1.0);

    taskVector.push_back(task.get());
  }

  std::experimental::filesystem::copy(taskVec[0]->getFilePath(),
                                      std::experimental::filesystem::path(RAI_LOG_PATH+"/TASKFILESAVE"));

  std::experimental::filesystem::copy(std::string(__FILE__),
                                      std::experimental::filesystem::path(RAI_LOG_PATH+"/APPFILESAVE"));

  //////////////////////////// Define Noise /////////////////////////////
  NoiseCov covariance = NoiseCov::Identity();
  std::vector<NormNoise> noiseVec(nThread, NormNoise(covariance));
  std::vector<NormNoise *> noiseVector;
  for (auto &noise : noiseVec)
    noiseVector.push_back(&noise);

  ////////////////////////// Define Function approximations //////////
  Vfunction_TensorFlow vfunction("gpu,0", "MLP", "tanh 1e-3 96 128 128 1", 0.001);
  Policy_TensorFlow policy("gpu,0", "MLP", "tanh 1e-3 96 128 128 12", 0.001);

// policy.loadParam("/home/joonho/Documents/ANYparam/stand_new/flip1/policy_3000.txt");
//   policy.loadParam("/home/joonho/Documents/ANYparam/stand_new/standing_policy_5000.txt"); /// retrain

  ////////////////////////// Acquisitor
  Acquisitor_ acquisitor;
  
  ////////////////////////// Algorithm ////////////////////////////////
  rai::Algorithm::TRPO_gae<Dtype, StateDim, ActionDim>
      algorithm(taskVector, &vfunction, &policy, noiseVector, &acquisitor, 0.97, 0, 0, 10, 0.6, 0.02, false);

  //algorithm.scalePolicyVar(2.0); ///retrain flip

  /////////////////////// Plotting properties ////////////////////////
  rai::Utils::Graph::FigProp2D figurePropertiesEVP;
  figurePropertiesEVP.title = "Number of Episodes vs Performance";
  figurePropertiesEVP.xlabel = "N. Episodes";
  figurePropertiesEVP.ylabel = "Performance";

  constexpr int loggingInterval = 100;
  constexpr int iterlimit = 10000;
  ////////////////////////// Learning /////////////////////////////////

  double noisefactor = 0.2;

  for (auto &task : taskVec) {
    task->setTaskLabel(0);
    task->setcostScale1(0.01);
    task->setcostScale2(0.01);
  }

  double actionLimit = M_PI;
  int nEpisode = 20000 / (taskVec[0]->timeLimit() / taskVec[0]->dt());

  for (int iterationNumber = 0; iterationNumber < iterlimit + 1; iterationNumber++) {

    LOG(INFO) << "iter :" << iterationNumber;
    LOG(INFO) << "Learning Rate: " << vfunction.getLearningRate();
    LOG(INFO) << "Label: " << taskVec[0]->taskLabel_;
    LOG(INFO) << "actionLimit: " << actionLimit;

    LOG(INFO) << "costScale1:" << taskVec[0]->getcostScale1();
    LOG(INFO) << "costScale2:" << taskVec[0]->getcostScale2();

    if (iterationNumber % loggingInterval == 0 || iterationNumber == iterlimit) {
      algorithm.setVisualizationLevel(1);
      taskVector[0]->enableVideoRecording();
    }


    if(iterationNumber == 1000){
      for (auto &task : taskVec) {
        task->setControlUpdate_dt(0.01);
        task->setDiscountFactor(0.9985);
      }
    }

    //
//    if(iterationNumber == 5000) {
//      for (auto &task : taskVec) {
//        task->setControlUpdate_dt(0.005);
//      }
//    }

    for (auto &task : taskVec) {
      task->increaseCostScale1(0.995);
      task->increaseCostScale2(0.998);
      task->setActionLimit(actionLimit);
    }
    algorithm.runOneLoopForNEpisode(nEpisode);

    if (iterationNumber % loggingInterval == 0 || iterationNumber == iterlimit) {
      taskVector[0]->disableRecording();
      algorithm.setVisualizationLevel(0);
      graph->figure(0, figurePropertiesEVP);
      graph->appendData(0, logger->getData("PerformanceTester/performance", 0),
                        logger->getData("PerformanceTester/performance", 1),
                        logger->getDataSize("PerformanceTester/performance"),
                        rai::Utils::Graph::PlotMethods2D::linespoints,
                        "performance",
                        "lw 2 lc 4 pi 1 pt 5 ps 1");
      graph->drawFigure(0);

      policy.dumpParam(
          RAI_LOG_PATH + "/recovery_policy_" + std::to_string(iterationNumber) + ".txt");
      vfunction.dumpParam(
          RAI_LOG_PATH + "/value_" + std::to_string(iterationNumber) + ".txt");
    }
  }
  rai::Utils::Graph::FigPropPieChart propChart;
  graph->drawPieChartWith_RAI_Timer(5, timer->getTimedItems(), propChart);
  graph->drawFigure(5, rai::Utils::Graph::OutputFormat::pdf);
  graph->waitForEnter();

  return 0;

}