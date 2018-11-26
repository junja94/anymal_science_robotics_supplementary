/**
 *
 *		19 dimension, q = [body_position (3 numbers IDX 0-2)
 *	 	 	 	 	 	   body_quaternion   (4 numbers IDX 3-6),
 * 	 	 	 	 	 	   leg1- HAA, HFE, KFE (3 numbers, IDX 7-9)
 *	 	 	 	 	 	   leg2- ... leg3- ... leg4- ...]
 *
 *	 	generalized velocities
 *		18 dimension, u = [body_linear (3 numbers IDX 0-2)
 *	 	 	 	 	 	   body_ang vel   (3 numbers IDX 3-5),
 *	 	 	 	 	 	   leg1- HAA, HFE, KFE vel (3 numbers, IDX 6-8)
 *	 	 	 	 	 	   leg2- ... leg3- ... leg4- ...]
 *
*       learning state =
       *      [command (horizontal velocity, yawrate)                      n =  3, si =   0
       *       z-axis in world frame expressed in body frame (R_b.row(2)), n =  3, si =   3
       *       body Linear velocities,                                     n =  3, si =   6
       *       body Angular velocities,                                    n =  3, si =   9
       *       joint position history(t0, t-2, t-4),                       n = 36, si =  12
       *       joint velocities(t0, t-2, t-4)                              n = 36, si =  48
       *       previous action                                             n = 12, si =  84
       *       ]
        task label =
        0 : flip
        1 : stand
        2 : walk

 */


#pragma once

// system inclusion
#include <cmath>
#include <iostream>
#include <string>
#include <fstream>
#include "rai/tasks/common/Task.hpp"
#include "rai/common/enumeration.hpp"
#include <unistd.h>

#include "rai/Core"
#include "raiCommon/TypeDef.hpp"
#include "raiCommon/math/RAI_math.hpp"
#include <raiSim/World_RG.hpp>
#include "raiCommon/utils/StopWatch.hpp"
#include <rai/function/common/SimpleMLPLayer.hpp>
#include "jhUtil.hpp"

namespace rai {
namespace Task {

constexpr int ActionDim = 12;
constexpr int StateDim = 96;
constexpr int CommandDim = 0;
constexpr int HistoryLength = 15;
constexpr int SmoothingWindow = 30;
constexpr int shapeDim = 68;

template<typename Dtype>
class ANYmal_full_collision : public Task<Dtype, StateDim, ActionDim, CommandDim> {

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using TaskBase = Task<Dtype, StateDim, ActionDim, CommandDim>;
  typedef typename TaskBase::Action Action;
  typedef typename TaskBase::State State;
  typedef typename TaskBase::StateBatch StateBatch;
  typedef typename TaskBase::Command Command;
  typedef typename TaskBase::VectorXD VectorXD;
  typedef typename TaskBase::MatrixXD MatrixXD;
  typedef Eigen::Matrix<double, 19, 1> GeneralizedCoordinate;
  typedef Eigen::Matrix<double, 18, 1> GeneralizedVelocities;

  typedef Eigen::MatrixXd MatrixXd;
  typedef Eigen::VectorXd VectorXd;
  typedef Eigen::Vector2d Vector2d;
  typedef Eigen::Vector3d Vector3d;
  typedef Eigen::Vector4d Vector4d;
  typedef Eigen::Matrix3d Matrix3d;
  typedef Eigen::Matrix4d Matrix4d;

  typedef Eigen::Matrix<double, 12, 1> Vector12d;
  typedef Eigen::Matrix<double, 18, 1> Vector18d;

  ANYmal_full_collision() = delete;

  explicit ANYmal_full_collision(int task_label = 0,
                                 bool visualize = false,
                                 int instance = 0,
                                 std::string actuatorParamPath = "",
                                 std::string urdfpath = "task/include/quadrupedLocomotion/model/robot_nofan_nolimit.urdf",
                                 std::string statePath = "/home/joonho/workspace/oldrai/src/anymal_raisim/data/initialStates_flip/"
  ) :
      taskLabel_(task_label),
      vis_on_(visualize),
      vis_ready_(false),
      vid_on_(false),
      actuator_(actuatorParamPath, {32, 32}),
      instance_(instance),
      test_(false) {

    //// set default parameters
    this->valueAtTermination_ = 10.0;
    this->discountFactor_ = 0.995;
    this->timeLimit_ = 5.0;
    this->controlUpdate_dt_ = 0.01;

    /// parameters for dynamics
    jointNominalConfig_ << 0.0, 0.4, -0.8, 0.0, 0.4, -0.8, 0.0, -0.4, 0.8, 0.0, -0.4, 0.8;
//    jointNominalConfig_ << 0.0, 0.7, -1.0, 0.0, 0.7, -1.0, 0.0, -0.7, 1.0, 0.0, -0.7, 1.0;
    q_.setZero(19);
    u_.setZero(18);
    q0.resize(19);

    q0 << 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.4, -0.8, -0.0, 0.4, -0.8, 0.0, -0.4, 0.8, -0.0, -0.4, 0.8;
    u0.setZero(18);

    q_initialNoiseScale.setZero(19);
    u_initialNoiseScale.setZero(18);

    tauMax_.setConstant(40.0);
    tauMin_.setConstant(-40.0);
    tau_.setZero();

    actionMax_.setConstant(M_PI);
    actionMin_.setConstant(-M_PI);

    /// state params
    stateOffset_ << VectorXD::Constant(2, 0.0), 0.0, /// command
        0.0, 0.0, 0.0, /// gravity axis
        VectorXD::Constant(6, 0.0), /// body lin/ang vel
        jointNominalConfig_.template cast<Dtype>(), /// joint position
        VectorXD::Constant(24, 0.0), /// position error
        VectorXD::Constant(36, 0.0), /// joint velocity history
        VectorXD::Constant(12, 0.0); /// prev. action

    stateScale_ << 1.0, 1.0 / 0.3, 1.0, /// command
        VectorXD::Constant(3, 1.0), /// gravity axis
        1.0 / 1.5, 1.0 / 0.5, 1.0 / 0.5, 1.0 / 2.5, 1.0 / 2.5, 1.0 / 2.5, /// linear and angular velocities
        VectorXD::Constant(36, 1 / 1.0), /// joint angles
        VectorXD::Constant(36, 1 / 10.0), /// joint velocities
        VectorXD::Constant(12, 1.0 / 1.0); /// prev. action

    /// action params
    actionScale_.setConstant(0.5);
    actionOffset_.setZero();
    previousAction_.setZero();

    /// env setup & visualization
    realTimeRatio_ = 1.0;
    watch_.start();
    viswatch_.start();
    if (vis_on_) {
      env_.reset(new rai_sim::World_RG(600, 600, 0.12f, rai_sim::NO_BACKGROUND));
      env_->setBackgroundColor(1.0f, 1.0f, 1.0f);
      env_->setLightPosition(10.0f, 10.0f, 5.0f);
    } else {
      env_.reset(new rai_sim::World_RG());
    }
    env_->setTimeStep(simulation_dt_);
    anymal_ = env_->addArticulatedSystem(urdfpath);
    terrain_ = env_->addCheckerboard(2.0, 100, 100, 0.2, -1);

    gravity_ << 0, 0, -9.81;
    env_->setGravity(gravity_);

    // additional actuator config
    //    Eigen::VectorXd damping(19);
    //    damping.setConstant(0.5);
    //    damping.head(7).setZero();
    //    anymal_->setJointDamping(damping);

    /// env visual
    if (vis_on_) {
      env_->cameraFollowObject(anymal_, {1.2, 1.2, 1.0});
      env_->visStart();
      rai_graphics::object::CheckerBoard
          *board = dynamic_cast<rai_graphics::object::CheckerBoard *>(terrain_.visual()[0]);
      board->setBoardColor({0.0, 0.0, 0.0}, {0.75, 0.75, 1.0});
      board->setBoardColor({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});

      std::vector<float> spec = {0.0, 0.0, 0.0}, amb = {5.0, 5.0, 5.0}, diff = {0.0, 0.0, 0.0};
      board->setLightProp(amb, diff, spec, 0.2);
      amb = {3.0, 3.0, 3.0};
      diff = {1.0, 1.0, 1.0};
      for (int i = 0; i < anymal_.visual().size(); i++) {
        anymal_.visual()[i]->setLightProp(amb, diff, spec, 0.1);
      }
      anymal_.visual()[0]->setColor({0.4f, 0.4f, 0.4f});
      for (int i = 0; i < 4; i++) {
        anymal_.visual()[2 + 5 * i]->setColor({0.4f, 0.4f, 0.4f});
        anymal_.visual()[3 + 5 * i]->setColor({0.4f, 0.4f, 0.4f});
        anymal_.visual()[4 + 5 * i]->setColor({0.4f, 0.4f, 0.4f});
        anymal_.visual()[5 + 5 * i]->setColor({0.0f, 0.0f, 0.0f});
      }
      env_->getRaiGraphics()->changeMenuText(0, false, "");

      vis_ready_ = true;
    }

    /// task options
    steps_ = 0;
    maxSteps_ = (int) (this->timeLimit_ / this->controlUpdate_dt_);
    initCounter_ = 0;
    costScale_ = 0.1;
    costScale2_ = 0.05;

    desiredHeight_ = 0.5;
    command_ << 0.0, 0.0, 0.0;

    initAngle_ = M_PI / 4;
    termHeight_ = 0.25;
    costMax_ = this->valueAtTermination_;
    shapes_.setZero(shapeDim);
    uPrev_.setZero(18);
    acc_.setZero(18);

    if (instance == 0)noisify_ = false;
    else noisify_ = true;
    /// cost options
    contactMultiplier_ = 0.0001;
    slipMultiplier_ = 2.0;
    torqueMultiplier_ = 0.005;
    velMultiplier_ = 0.02;

    /// noise peoperty
    q_initialNoiseScale.setConstant(0.6);
    q_initialNoiseScale.segment(0, 3) << 0.00, 0.00, 0.15;

    q_initialNoiseScale(7) = 0.0;
    q_initialNoiseScale(10) = 0.0;
    q_initialNoiseScale(13) = 0.0;
    q_initialNoiseScale(16) = 0.0;

    u_initialNoiseScale.setConstant(.0);
    u_initialNoiseScale.segment(3, 3).setConstant(0.5);
    u_initialNoiseScale(0) = 0.5;
    u_initialNoiseScale(1) = 0.5;
    u_initialNoiseScale(2) = 0.05;

    /// inner states
    for (int i = 0; i < 4; i++) {
      footContactState_[i] = false;
      shankContactState_[i] = false;
      thighContactState_[i] = false;
    }

    numContact_ = 0;
    previousAction_.setZero();
    jointVelHist_.setZero();
    jointPosHist_.setZero();
    torqueHist_.setZero();

    for (int i = 0; i < 4; i++) {
      footPos_.push_back(std::get<1>(anymal_->getCollisionObj()[9 * i + 13]));
      footR_[i] = anymal_->getVisColProps()[9 * i + 13].second[0];
    }

    footPos_W.resize(4);
    footVel_W.resize(4);
    footContactVel_.resize(4);
    footNormal_.resize(4);
    badlyConditioned_ = false;

    /// initialize ANYmal
    anymal_->setGeneralizedCoordinate(q0);
    anymal_->setGeneralizedVelocity(u0);
    anymal_->setGeneralizedForce(tau_);

    for (int i = 0; i < 4; i++) {
      int idx1 = 9 * i + 9;
      int idx2 = 9 * i + 11;
      anymal_->ignoreCollisionBetween2(idx1, idx2);

      idx1 = 9 * i + 7;
      idx2 = 9 * i + 13;
      anymal_->ignoreCollisionBetween2(idx1, idx2);

      idx1 = 9 * i + 5;
      idx2 = 9 * i + 7;
      anymal_->ignoreCollisionBetween2(idx1, idx2);

      idx2 = 9 * i + 8;
      anymal_->ignoreCollisionBetween2(idx1, idx2);

      idx1 = 9 * i + 6;
      anymal_->ignoreCollisionBetween2(idx1, idx2);
      anymal_->ignoreCollisionBetween2(idx1, 0);

      idx1 = 9 * i + 10;
      idx2 = 9 * i + 11;
      anymal_->ignoreCollisionBetween2(idx1, idx2);
    }

    /// initialize Material property
    materials_.setMaterialNames({"terrain", "robot"});
    materials_.setMaterialPairProp("terrain", "robot", 1.2, 0.0, 0.0);
    materials_.setMaterialPairProp("robot", "robot", 1.2, 0.0, 0.0);

    env_->setERP(0.0, 0.0);

    env_->updateMaterialProp(materials_);
    terrainKey_ = env_->getMaterialKey("terrain");
    robotKey_ = env_->getMaterialKey("robot");
    terrain_->setMaterial(terrainKey_);
    anymal_->setMaterial(robotKey_);


    /// collect joint positions, collision geometry
    defaultJointPositions_.resize(13);
    defaultBodyMasses_.resize(13);

    for (int i = 0; i < 13; i++) {
      defaultJointPositions_[i] = anymal_->getJointPos_P()[i].e();
      defaultBodyMasses_[i] = anymal_->getMass(i);
    }
    int cnt = 0;

    for (auto &obj : anymal_->getCollisionObj()) {
      Position props;
      props << anymal_->getVisColProps()[cnt].second[0],
          anymal_->getVisColProps()[cnt].second[1],
          anymal_->getVisColProps()[cnt].second[2];
      defaultCollisionBodyPositions_.push_back(std::get<1>(obj).e());
      defaultCollisionBodyProps_.push_back(props);
      cnt++;
    }

    COMPosition_ = anymal_->getLinkCOM()[0].e();

    if (instance != -1) {
      /// load initial states
      initialStateCapacity_ = 42000;
      noisifyDynamicsInterval_ = 300;
      changeShape_ = false;
      {
        std::string dataPath;
        dataPath = statePath + "initialStates_" + std::to_string(instance_) + ".txt";
        const static Eigen::IOFormat CSVFormat(10, Eigen::DontAlignCols, ", ", "\n");

        q0s.resize(initialStateCapacity_ * 19);

        std::ifstream indata;
        indata.open(dataPath);
        LOG_IF(FATAL, !indata.is_open()) << "Parameter file " << dataPath << " could not be opened";
        std::string line;
        getline(indata, line);
        std::stringstream lineStream(line);
        std::string cell;
        int paramSize = 0;
        while (std::getline(lineStream, cell, ',')) {
          q0s(paramSize++) = std::stof(cell);
        }
        LOG_IF(FATAL, paramSize % 19 != 0) << "wrong param size";
      }
      {
        std::string dataPath;
        dataPath = statePath + "initialShapes_" + std::to_string(instance_) + ".txt";

        const static Eigen::IOFormat CSVFormat(10, Eigen::DontAlignCols, ", ", "\n");

        shape0s_.resize(initialStateCapacity_ * shapeDim);

        std::ifstream indata;
        indata.open(dataPath);
        LOG_IF(FATAL, !indata.is_open()) << "Parameter file " << dataPath << " could not be opened";
        std::string line;
        getline(indata, line);
        std::stringstream lineStream(line);
        std::string cell;
        int paramSize = 0;
        while (std::getline(lineStream, cell, ',')) {
          shape0s_(paramSize++) = std::stof(cell);
        }
        LOG_IF(FATAL, paramSize % shapeDim != 0) << "wrong param size";
      }
    }

  }

  ~ANYmal_full_collision() {
    if (vis_ready_) env_->visEnd();
  }

  std::string getFilePath() {
    return __FILE__;
  }

  rai_sim::World_RG *getEnv() {
    return env_.get();
  }

  void setcostScale1(double in) {
    costScale_ = in;
  }
  void setcostScale2(double in) {
    costScale2_ = in;
  }

  double &getcostScale1() {
    return costScale_;
  }
  double &getcostScale2() {
    return costScale2_;
  }

  void setInitAngle(double in) {
    initAngle_ = in;
  }

  void setActionLimit(double in) {
    actionMax_.setConstant(in);
    actionMin_.setConstant(-in);
  }

  void increaseInitAngle(double in, double max = M_PI / 2) {
    initAngle_ = std::min(max, initAngle_ * in);
  }

  void increaseCostScale(double in) {
    costScale_ = std::pow(costScale_, in);
    costScale2_ = std::pow(costScale2_, in);
  }

  void increaseCostScale1(double in) {
    costScale_ = std::pow(costScale_, in);
  }

  void increaseCostScale2(double in) {
    costScale2_ = std::pow(costScale2_, in);
  }

  void setTaskLabel(int n) {
    taskLabel_ = n;
  }

  int getTaskLabel() {
    return taskLabel_;
  }

  void getContactState(Eigen::Matrix<Dtype, 4, 1> &contactState) {
    contactState << footContactState_[0], footContactState_[1], footContactState_[2], footContactState_[3];
  }

  void setInitialState(const GeneralizedCoordinate &q, const GeneralizedVelocities &u) {
    q0 = q;
    u0 = u;
    conversion_GeneralizedState2LearningState(state0_, q0, u0);
  }

  // using learning state
  void setInitialState(const State &in) {
    state0_ = in;
    conversion_LearningState2GeneralizedState(in, q0, u0);
  }

  inline void setCommand(const Vector3d &commandIN) {
    command_ = commandIN;
  }

  inline void updateVisual() {
    if (this->visualization_ON_ && vis_on_) {
      /// visualize contact
      for (int i = 0; i < 4; i++) {
        if (footContactState_[i]) {
          anymal_.visual()[5 + 5 * i]->setColor({0.3f, 0.3f, 1.0f});
        } else {
          anymal_.visual()[5 + 5 * i]->setColor({0.0f, 0.0f, 0.0f});
        }
      }
    }
  }

  inline void comprehendContacts() {
    numContact_ = anymal_->getContacts().size();

    numFootContact_ = 0;
    numThighContact_ = 0;
    numShankContact_ = 0;
    numHipContact_ = 0;
    numInternalContact_ = 0;
    numBodyContact_ = 0;

    sumBodyImpulse_ = 0;
    sumBodyContactVel_ = 0;

    for (int k = 0; k < 4; k++) {
      footContactState_[k] = false;
      shankContactState_[k] = false;
      thighContactState_[k] = false;
    }

    rai_sim::Vec<3> vec3;

    //position of the feet
    anymal_->getPosition_W(3, footPos_[0], footPos_W[0]);
    anymal_->getVelocity_W(3, footPos_[0], footVel_W[0]);
    footPos_W[0][2] -= footR_[0];
    anymal_->getPosition_W(6, footPos_[1], footPos_W[1]);
    anymal_->getVelocity_W(6, footPos_[1], footVel_W[1]);
    footPos_W[1][2] -= footR_[1];
    anymal_->getPosition_W(9, footPos_[2], footPos_W[2]);
    anymal_->getVelocity_W(9, footPos_[2], footVel_W[2]);
    footPos_W[2][2] -= footR_[2];
    anymal_->getPosition_W(12, footPos_[3], footPos_W[3]);
    anymal_->getVelocity_W(12, footPos_[3], footVel_W[3]);
    footPos_W[3][2] -= footR_[3];

    //Classify foot contact
    if (numContact_ > 0) {
      for (int k = 0; k < numContact_; k++) {
        if (!anymal_->getContacts()[k].skip()) {

          int idx = anymal_->getContacts()[k].getlocalBodyIndex();

          // check foot height to distinguish shank contact
          // TODO: this only works for flat terrain

          if (idx == 3 && footPos_W[0][2] < 1e-6 && !footContactState_[0]) {
            footContactState_[0] = true;
            footNormal_[0] = anymal_->getContacts()[k].getNormal().e();
            anymal_->getContactPointVel(k, vec3);
            footContactVel_[0] = vec3.e();
            numFootContact_++;
          } else if (idx == 6 && footPos_W[1][2] < 1e-6 && !footContactState_[1]) {
            footContactState_[1] = true;
            footNormal_[1] = anymal_->getContacts()[k].getNormal().e();
            anymal_->getContactPointVel(k, vec3);
            footContactVel_[1] = vec3.e();
            numFootContact_++;
          } else if (idx == 9 && footPos_W[2][2] < 1e-6 && !footContactState_[2]) {
            footContactState_[2] = true;
            footNormal_[2] = anymal_->getContacts()[k].getNormal().e();
            anymal_->getContactPointVel(k, vec3);
            footContactVel_[2] = vec3.e();
            numFootContact_++;
          } else if (idx == 12 && footPos_W[3][2] < 1e-6 && !footContactState_[3]) {
            footContactState_[3] = true;
            footNormal_[3] = anymal_->getContacts()[k].getNormal().e();
            anymal_->getContactPointVel(k, vec3);
            footContactVel_[3] = vec3.e();
            numFootContact_++;
          } else {
            numBodyContact_++;
            anymal_->getContactPointVel(k, vec3);
            sumBodyContactVel_ += vec3.e().head(2).squaredNorm();

            if (!anymal_->getContacts()[k].isSelfCollision()) {

              if (idx == 1 || idx == 4 || idx == 7 || idx == 10) {
                numHipContact_++;
              } else if (idx == 2) {
                thighContactState_[0] = true;
                numThighContact_++;
              } else if (idx == 5) {
                thighContactState_[1] = true;
                numThighContact_++;
              } else if (idx == 8) {
                thighContactState_[2] = true;
                numThighContact_++;
              } else if (idx == 11) {
                thighContactState_[3] = true;
                numThighContact_++;
              } else if (idx == 3) {
                shankContactState_[0] = true;
                numShankContact_++;
              } else if (idx == 6) {
                shankContactState_[1] = true;
                numShankContact_++;
              } else if (idx == 9) {
                shankContactState_[2] = true;
                numShankContact_++;
              } else if (idx == 12) {
                shankContactState_[3] = true;
                numShankContact_++;
              }
              sumBodyImpulse_ += anymal_->getContacts()[k].getImpulse()->norm();
            }
          }
        } else {
          numInternalContact_++;
        }
      }
    }
  }

  void step(const Action &action_t,
            State &state_tp1,
            TerminationType &termType,
            Dtype &costOUT) {
    costOUT = 0.0;

    RAIWARN_IF(isinf(action_t.norm()), "action is inf" << std::endl
                                                       << action_t.transpose());
    RAIWARN_IF(isnan(action_t.norm()), "action is nan" << std::endl
                                                       << action_t.transpose());
    if (isnan(action_t.norm())) badlyConditioned_ = true;
    if (isinf(action_t.norm())) badlyConditioned_ = true;

    Dtype intermediateCost;

    /// PD controller
    scaledAction_ = action_t.cwiseProduct(actionScale_) + actionOffset_;
    targetPosition_ = scaledAction_.template cast<double>().cwiseMin(actionMax_);
    targetPosition_ = targetPosition_.cwiseMax(actionMin_);

    if (false) {
      targetPosition_ += jointNominalConfig_;
    } else {
      targetPosition_ += q_.tail(12);
    }

    bool terminate = false;

    for (int i = 0; i < (int) (this->controlUpdate_dt_ / simulation_dt_); i++) {

      Eigen::Matrix<Dtype, HistoryLength * 12 - 12, 1> temp;
      temp = jointVelHist_.tail(HistoryLength * 12 - 12);
      jointVelHist_.head(HistoryLength * 12 - 12) = temp;
      jointVelHist_.tail(12) = u_.tail(12).template cast<Dtype>();

      temp = jointPosHist_.tail(HistoryLength * 12 - 12);
      jointPosHist_.head(HistoryLength * 12 - 12) = temp;
      jointPosHist_.tail(12) = (targetPosition_ - q_.tail(12)).template cast<Dtype>();

      temp = torqueHist_.tail(HistoryLength * 12 - 12);
      torqueHist_.head(HistoryLength * 12 - 12) = temp;
      torqueHist_.tail(12) = tau_.tail(12).template cast<Dtype>();

      Eigen::Matrix<double, 6, 1> seaInput;
      for (int actId = 0; actId < 12; actId++) {
        seaInput[0] = (jointVelHist_(actId + (HistoryLength - 7) * 12) + 0.003) * 0.474;
        seaInput[1] = (jointVelHist_(actId + (HistoryLength - 4) * 12) + 0.003) * 0.473;
        seaInput[2] = (jointVelHist_(actId + (HistoryLength - 1) * 12) + 0.003) * 0.473;

        seaInput[3] = (jointPosHist_(actId + (HistoryLength - 7) * 12) + 0.005) * 7.629;
        seaInput[4] = (jointPosHist_(actId + (HistoryLength - 4) * 12) + 0.005) * 7.629;
        seaInput[5] = (jointPosHist_(actId + (HistoryLength - 1) * 12) + 0.005) * 7.628;

//
//        seaInput[0] = jointVelHist_(actId + HistoryLength*12-84) * 0.2;
//        seaInput[1] = jointVelHist_(actId + HistoryLength*12-45) * 0.2;
//        seaInput[2] = jointVelHist_(actId + HistoryLength*12-12) * 0.2;
//
//        seaInput[3] = jointPosHist_(actId + HistoryLength*12-84) * 5.0;
//        seaInput[4] = jointPosHist_(actId + HistoryLength*12-48) * 5.0;
//        seaInput[5] = jointPosHist_(actId + HistoryLength*12-12) * 5.0;

        tau_(6 + actId) = actuator_.forward(seaInput)[0] * 20.0;
      }

      Eigen::Matrix<double, 12, 1> tauTemp;
      tauTemp = tau_.tail(12).cwiseMin(tauMax_);
      for (size_t k = 0; k < 12; k++) {
        tau_[6 + k] = std::max(tauMin_[k], tauTemp[k]);
      }

      tau_.head(6).setZero();

      integrateOneTimeStep();
      if (test_) {
        Utils::logger->appendData("Ncontacts",
                                  loggingStep_++ * simulation_dt_,
                                  anymal_->getContacts().size());

//        Utils::logger->appendData("speed",
//                                  loggingStep_++ * simulation_dt_,
//                                  u_[6], u_[7], u_[8]);
//
//        Utils::logger->appendData("command", loggingStep_++ * simulation_dt_,
//                                  tau_.tail(12)[0], tau_.tail(12)[1], tau_.tail(12)[2],
//                                  tau_.tail(12)[3], tau_.tail(12)[4], tau_.tail(12)[5],
//                                  tau_.tail(12)[6], tau_.tail(12)[7], tau_.tail(12)[8],
//                                  tau_.tail(12)[9], tau_.tail(12)[10], tau_.tail(12)[11]);

//      velMean_.setZero();
//      for (int j = 0; j < SmoothingWindow; j++){
//        velMean_ += jointVelHist_.template segment<12>(12 * j);
//      }
//     velMean_ /= (SmoothingWindow);
//      Utils::logger->appendData("actuator", loggingStep_++,
//                                scaledAction_[2],
//                                targetPosition_(2) - q_(9),
//                                q_(9),
//                                u_(8),
//                                tau_(8));
      } else {
        noisifyTerrain();
      }

      for (int j = 0; j < 18; j++) {
        acc_[j] = (u_[j] - uPrev_[j]) / simulation_dt_;
      }
      uPrev_ = u_;

      if (!badlyConditioned_) calculateCost(intermediateCost);

      costOUT += intermediateCost;

      if (isTerminalState(q_, u_)) {
        termType = TerminationType::terminalState;
        break;
      }
    }

    updateVisual();
    getState(state_tp1);

    RAIWARN_IF(isinf(state_tp1.norm()), "state_tp1 is inf" << std::endl
                                                           << state_tp1.transpose());
    RAIWARN_IF(isnan(state_tp1.norm()), "state_tp1 is nan" << std::endl
                                                           << state_tp1.transpose());
    if (isnan(state_tp1.norm())) badlyConditioned_ = true;
    if (isinf(state_tp1.norm())) badlyConditioned_ = true;

    if (badlyConditioned_) {
      termType = TerminationType::terminalState;
      costOUT = costMax_;
    }

    Dtype NoiseFtr = 1.0 * costScale2_;

    /// noisify body orientation
    for (int i = 3; i < 6; i++)
      state_tp1[i] += NoiseFtr * rn_.sampleUniform() * 0.2;

    if (numFootContact_ < 2) {
      /// noisify body vel
      for (int i = 6; i < 12; i++)
        state_tp1[i] += NoiseFtr * rn_.sampleUniform() * 5.0;
    } else {
      /// noisify body vel
      for (int i = 6; i < 12; i++)
        state_tp1[i] += NoiseFtr * rn_.sampleUniform() * 0.2;
    }

    /// noisify joint vel
    for (int i = 48; i < 84; i++)
      state_tp1[i] += NoiseFtr * rn_.sampleUniform() * 0.05;

    previousAction_ = scaledAction_;
    steps_++;
  }

  void sampleInitialState(Eigen::VectorXd &qSample,
                          Eigen::VectorXd &shapeSample,
                          double noiseFactor,
                          double dt = 0.0025) {
    int stepLimit = 500;
    double angle;
    steps_ = 0; //just to debug
    if (dt != simulation_dt_) env_->setTimeStep(dt);
    do {
      q_ = q0;
      u_ = u0;
      initTasks();

      noiseFactor = 0.5;

      for (int i = 0; i < 4; i++) {
        q_(3 * i + 8) += rn_.sampleNormal() * noiseFactor; //
        q_(3 * i + 9) += rn_.sampleNormal() * noiseFactor; //
      }
      q_(7) = q_(10) = q_(13) = q_(16) = 0;

      /// Random orientation using angle axis

      Eigen::Vector3d heading;
      if (initCounter_ % 2 == 0) heading(0) = -1.0;
      else heading(0) = 1.0;
      heading(1) = 1.0 * rn_.sampleUniform();
      heading(2) = 1.0 * rn_.sampleUniform();

      heading.normalize();

      angle = initAngle_ + rn_.sampleNormal() * 1.0;
      angle = std::min(angle, initAngle_ * 1.2);
      angle = std::max(angle, 0.0);

      angle = rn_.sampleUniform() * 0.3;

      double sin = std::sin(angle / 2.0);
      q_.template segment<3>(4) = heading * sin;
      q_(3) = std::cos(angle / 2.0);
      q_(2) = 0.35;

      u_.setZero();
      for (int i = 0; i < 6; i++)
        u_(i) = u_initialNoiseScale(i) * rn_.sampleUniform() * 1.0;

      anymal_->setGeneralizedCoordinate(q_);
      anymal_->setGeneralizedVelocity(u_);
      Eigen::VectorXd damping(19);
      damping.setConstant(1.0);
      damping.setConstant(0.3);
      damping[7] = damping[10] = damping[13] = damping[16] = 5.0;
      damping.head(7).setZero();
      anymal_->setJointDamping(damping);
      int cnt = 0;
      do {
        cnt++;
        this->passiveStep(dt, 0.0);
        if (badlyConditioned_)break;
      } while (u_.norm() > 0.0001 && cnt < stepLimit);
    } while (isTerminalState(q_, u_));

    qSample = q_;
    for (int i = 0; i < 12; i++) {
      qSample[7 + i] = anglemod(qSample[7 + i]); // within [-180, 180]
    }

    shapeSample = shapes_;
    if (dt != simulation_dt_) env_->setTimeStep(simulation_dt_);
    initCounter_++;
  }

  void passiveStep(double dt, double damping) {

    Eigen::VectorXd damping_(19);
    damping_.setConstant(damping);
    damping_.head(7).setZero();

    anymal_->setJointDamping(damping_);

    tau_.setZero();
    env_->integrate1();
    anymal_->setGeneralizedForce(tau_);
    env_->integrate2();
    steps_++;
    q_ = anymal_->getGeneralizedCoordinate();
    u_ = anymal_->getGeneralizedVelocity();

    if (isnan(u_.norm()) || isinf(u_.norm())) {
      RAIWARN("error in simulation, " << steps_);
      badlyConditioned_ = true;
    }

    if (this->visualization_ON_ && vis_on_) {
      if (!vis_ready_) {
        env_->visStart();
      }

      double waitTime = std::max(0.0, dt / 2.0 - watch_.measure()); // 2x realtime
      usleep(waitTime * 1e6);
      watch_.start();
      if (viswatch_.measure() > 1.0 / 80.0) {
        env_->updateFrame();
        viswatch_.start();
      }
    }

    damping_.setZero();

    anymal_->setJointDamping(damping_);
  }

  void noisifyDynamics() {
    int shapeIdx = 0;
    shapes_.setZero(shapeDim);

    /// link length & shape randomization
    for (int i = 0; i < 4; i++) {

      double x_, y_, z_;
      if (i < 2) x_ = rn_.sampleUniform01() * 0.01;
      else x_ = -rn_.sampleUniform01() * 0.01;

      y_ = rn_.sampleUniform() * 0.02;
      z_ = rn_.sampleUniform() * 0.02;

      shapes_[shapeIdx++] = x_;//0
      shapes_[shapeIdx++] = y_;//1
      shapes_[shapeIdx++] = z_;//2

      int hipIdx = 3 * i + 1;
      int shankIdx = 3 * i + 3;

      ///hip
      anymal_->getJointPos_P()[hipIdx][0] = defaultJointPositions_[hipIdx][0] + x_;
      anymal_->getJointPos_P()[hipIdx][1] = defaultJointPositions_[hipIdx][1] + y_;
      anymal_->getJointPos_P()[hipIdx][2] = defaultJointPositions_[hipIdx][2] + z_; ///1

      ///hip actuator
      int protectorIdx = 9 * i + 5;
      double dr = rn_.sampleUniform() * 0.005;
      double temp = rn_.sampleUniform() * 0.01;
      shapes_[shapeIdx++] = dr;//3
      shapes_[shapeIdx++] = temp;//4

      anymal_->getVisColProps()[protectorIdx].second[0] =
          defaultCollisionBodyProps_[protectorIdx][0] + dr;
      anymal_->getVisColProps()[protectorIdx].second[1] =
          defaultCollisionBodyProps_[protectorIdx][1] + temp;
      protectorIdx++;
      anymal_->getVisColProps()[protectorIdx].second[0] =
          defaultCollisionBodyProps_[protectorIdx][0] + dr;

      double dy_ = rn_.sampleUniform() * 0.01;
      shapes_[shapeIdx++] = dy_;//5

      /// shank
      //  dy>0 -> move outwards
      if (i % 2 == 1) {
        y_ = -dy_;
      } else {
        y_ = dy_;
      }

      x_ = rn_.sampleUniform() * 0.01;
      z_ = rn_.sampleUniform() * 0.01; // 너무 늘리면 다리끼리 충돌

      shapes_[shapeIdx++] = x_;//6
      shapes_[shapeIdx++] = z_;//7

      anymal_->getJointPos_P()[shankIdx].v[0] = defaultJointPositions_[shankIdx][0] + x_;
      anymal_->getJointPos_P()[shankIdx].v[1] = defaultJointPositions_[shankIdx][1] + y_;
      anymal_->getJointPos_P()[shankIdx].v[2] = defaultJointPositions_[shankIdx][2] + z_;

      shankIdx = 9 * i + 10;

      /// KFE actuator (note: attached to thigh)
      std::get<1>(anymal_->getVisColOb()[shankIdx])[0] =
      std::get<1>(anymal_->getCollisionObj()[shankIdx])[0] =
          defaultCollisionBodyPositions_[shankIdx][0] + x_; // position
      std::get<1>(anymal_->getVisColOb()[shankIdx])[1] =
      std::get<1>(anymal_->getCollisionObj()[shankIdx])[1] =
          defaultCollisionBodyPositions_[shankIdx][1] + y_; // position
      std::get<1>(anymal_->getVisColOb()[shankIdx])[2] =
      std::get<1>(anymal_->getCollisionObj()[shankIdx])[2] =
          defaultCollisionBodyPositions_[shankIdx][2] + z_; // position

      dr = rn_.sampleUniform() * 0.01;
      double dl = rn_.sampleUniform() * 0.01;
      shapes_[shapeIdx++] = dr;//8
      shapes_[shapeIdx++] = dl;//9

      anymal_->getVisColProps()[shankIdx].second[0] =
          defaultCollisionBodyProps_[shankIdx][0] + dr;
      anymal_->getVisColProps()[shankIdx].second[1] =
          defaultCollisionBodyProps_[shankIdx][1] + dl;

      shankIdx++;

      if (i < 2) {
        x_ = rn_.sampleUniform01() * dr;
      } else {
        x_ = -rn_.sampleUniform01() * dr;
      }
      shapes_[shapeIdx++] = x_;//10

      /// shank box
      std::get<1>(anymal_->getVisColOb()[shankIdx])[0] =
      std::get<1>(anymal_->getCollisionObj()[shankIdx])[0] =
          defaultCollisionBodyPositions_[shankIdx][0] + x_; // position

      /// Foot
      int footIdx = 9 * i + 13;
      double dz_ = rn_.sampleUniform() * 0.02;
      shapes_[shapeIdx++] = dz_;//11

      std::get<1>(anymal_->getVisColOb()[footIdx])[0] =
      std::get<1>(anymal_->getCollisionObj()[footIdx])[0] = defaultCollisionBodyPositions_[footIdx][0] + x_; // position
      std::get<1>(anymal_->getVisColOb()[footIdx])[2] =
      std::get<1>(anymal_->getCollisionObj()[footIdx])[2] =
          defaultCollisionBodyPositions_[footIdx][2] + dz_;

      footPos_[i] = std::get<1>(anymal_->getCollisionObj()[footIdx]);

      /// shank cylinder
      shankIdx++;
      std::get<1>(anymal_->getVisColOb()[shankIdx])[0] =
      std::get<1>(anymal_->getCollisionObj()[shankIdx])[0] =
          defaultCollisionBodyPositions_[shankIdx][0] + x_; // position
      std::get<1>(anymal_->getVisColOb()[shankIdx])[2] =
      std::get<1>(anymal_->getCollisionObj()[shankIdx])[2] =
          defaultCollisionBodyPositions_[shankIdx][2] + dz_ * 0.5; // position
      anymal_->getVisColProps()[shankIdx].second[1] =
          defaultCollisionBodyProps_[shankIdx][1] - dz_;

      /// Foot radius
      dr = rn_.sampleUniform() * 0.005;
      shapes_[shapeIdx++] = dr;//12

      footR_[i] = defaultCollisionBodyProps_[footIdx][0] + dr;
      anymal_->getVisColProps()[footIdx].second[0] = footR_[i];

      /// modify shoulder protector
      protectorIdx = 9 * i + 7;

      double dr2 = rn_.sampleUniform() * 0.005;
      double margin = (dy_ - dr);

      dl = rn_.sampleUniform() * std::min(margin, 0.02);

      shapes_[shapeIdx++] = dr2;//13
      shapes_[shapeIdx++] = dl;//14

      anymal_->getVisColProps()[protectorIdx].second[0] =
          defaultCollisionBodyProps_[protectorIdx][0] + dr2;
      anymal_->getVisColProps()[protectorIdx].second[1] =
          defaultCollisionBodyProps_[protectorIdx][1] + dl; // cylinder length

      protectorIdx++;
      anymal_->getVisColProps()[protectorIdx].second[0] =
          defaultCollisionBodyProps_[protectorIdx][0] + dr2;
      anymal_->getVisColProps()[protectorIdx].second[1] =
          defaultCollisionBodyProps_[protectorIdx][1] + dl;


      /// shank
      if (i % 2 == 1) {
        dl *= -1;
      }
      std::get<1>(anymal_->getVisColOb()[protectorIdx])[1] =
      std::get<1>(anymal_->getCollisionObj()[protectorIdx])[1] =
          defaultCollisionBodyPositions_[protectorIdx][1] + 0.5 * dl; // position

      env_->renewCollisionObjectShape(anymal_, 9 * i + 5);
      env_->renewCollisionObjectShape(anymal_, 9 * i + 6);
      env_->renewCollisionObjectShape(anymal_, 9 * i + 7);
      env_->renewCollisionObjectShape(anymal_, 9 * i + 8);
      env_->renewCollisionObjectShape(anymal_, 9 * i + 10);
      env_->renewCollisionObjectShape(anymal_, 9 * i + 12);
      env_->renewCollisionObjectShape(anymal_, 9 * i + 13);
    }

    /// base attachments
    for (int i = 1; i < 3; i++) {
      double dy = rn_.sampleUniform() * 0.02;
      double dz = rn_.sampleUniform() * 0.02;
      if (i == 2) dy = 0; // temp fix
      shapes_[shapeIdx++] = dy;
      shapes_[shapeIdx++] = dz;

      std::get<1>(anymal_->getVisColOb()[i])[1] =
      std::get<1>(anymal_->getCollisionObj()[i])[1] =
          defaultCollisionBodyPositions_[i][1] + dy;

      std::get<1>(anymal_->getVisColOb()[i])[2] =
      std::get<1>(anymal_->getCollisionObj()[i])[2] =
          defaultCollisionBodyPositions_[i][2] + dz;
    }
    {
      double dy = rn_.sampleUniform() * 0.02;
      double dz = rn_.sampleUniform() * 0.02;

      shapes_[shapeIdx++] = dy;
      shapes_[shapeIdx++] = dz;

      std::get<1>(anymal_->getVisColOb()[3])[1] =
      std::get<1>(anymal_->getCollisionObj()[3])[1] =
          defaultCollisionBodyPositions_[3][1] + dy;

      std::get<1>(anymal_->getVisColOb()[4])[1] =
      std::get<1>(anymal_->getCollisionObj()[3])[1] =
          defaultCollisionBodyPositions_[4][1] + dy;

      std::get<1>(anymal_->getVisColOb()[3])[2] =
      std::get<1>(anymal_->getCollisionObj()[3])[2] =
          defaultCollisionBodyPositions_[3][2] + dz * 0.5;

      std::get<1>(anymal_->getVisColOb()[4])[2] =
      std::get<1>(anymal_->getCollisionObj()[4])[2] =
          defaultCollisionBodyPositions_[4][2] + dz;
    }
    double dz = rn_.sampleUniform() * 0.02;
    double dwidth = rn_.sampleUniform() * 0.02;

    shapes_[shapeIdx++] = dz;

    std::get<1>(anymal_->getVisColOb()[0])[2] =
    std::get<1>(anymal_->getCollisionObj()[0])[2] =
        defaultCollisionBodyPositions_[0][2] + dz;

    shapes_[shapeIdx++] = dwidth;
    anymal_->getVisColProps()[0].second[1] = defaultCollisionBodyProps_[0][1] + dwidth;
    env_->renewCollisionObjectShape(anymal_, 0);
    noisifyMass();
    DRAIWARN(shapeIdx)

    anymal_->ignoreCollisionClear();
    for (int i = 0; i < 4; i++) {
      int idx1 = 9 * i + 9;
      int idx2 = 9 * i + 11;
      anymal_->ignoreCollisionBetween2(idx1, idx2);

      idx1 = 9 * i + 7;
      idx2 = 9 * i + 13;
      anymal_->ignoreCollisionBetween2(idx1, idx2);

      idx1 = 9 * i + 5;
      idx2 = 9 * i + 7;
      anymal_->ignoreCollisionBetween2(idx1, idx2);

      idx2 = 9 * i + 8;
      anymal_->ignoreCollisionBetween2(idx1, idx2);

      idx1 = 9 * i + 6;
      anymal_->ignoreCollisionBetween2(idx1, 0);

      idx1 = 9 * i + 10;
      idx2 = 9 * i + 11;
      anymal_->ignoreCollisionBetween2(idx1, idx2);
    }
  }

  void noisifyMass() {
    anymal_->getMass()[0] = defaultBodyMasses_[0] + rn_.sampleUniform01() * 3;

    /// hip
    for (int i = 1; i < 13; i += 3) {
      anymal_->getMass()[i] = defaultBodyMasses_[i] + rn_.sampleUniform() * 0.22;
    }

    /// thigh
    for (int i = 2; i < 13; i += 3) {
      anymal_->getMass()[i] = defaultBodyMasses_[i] + rn_.sampleUniform() * 0.22;
    }

    /// shank
    for (int i = 3; i < 13; i += 3) {
      anymal_->getMass()[i] = defaultBodyMasses_[i] + rn_.sampleUniform() * 0.06;
    }

    for (int i = 0; i < 3; i++) {
      anymal_->getLinkCOM()[0].v[i] = COMPosition_[i] + rn_.sampleUniform() * 0.03;
    }
    anymal_->updateMassInfo();
  }

  void noisifyTerrain() {
    double newcf = 0.8 + 1.2 * rn_.sampleUniform01();
    materials_.getMaterialPairProp(terrainKey_, robotKey_).c_f = newcf;
    newcf = 0.8 + 1.2 * rn_.sampleUniform01();
    materials_.getMaterialPairProp(robotKey_, robotKey_).c_f = newcf;

    env_->updateMaterialProp(materials_);
  }

  void setFriction(Dtype in) {
    double newcf = in;
    materials_.getMaterialPairProp(terrainKey_, robotKey_).c_f = newcf;
    materials_.getMaterialPairProp(robotKey_, robotKey_).c_f = newcf;

    env_->updateMaterialProp(materials_);
  }

  void initTasks() {
    numContact_ = 0;
    numFootContact_ = 0;
    numThighContact_ = 0;
    q_ = q0;
    u_ = u0;

    Vector3d command_temp;

    if (taskLabel_ < 2) {
      command_temp << 0.0, 0.0, 0.0;
      setCommand(command_temp);
    } else {
      double mag = rn_.sampleUniform01();
      double angle = M_PI * rn_.sampleUniform();
      command_temp << mag * std::cos(angle), mag * std::sin(angle), 0.5 * rn_.sampleUniform01();
//      command_temp << 0.5, 0.0, 0.0;
      setCommand(command_temp);
    }

    for (int i = 0; i < 4; i++) footContactState_[i] = false;

    if (noisify_) {
      noisifyTerrain();
      noisifyMass();
    }
  }

  void initFromSample() {
    /// read
    q_ = q0s.segment<19>(19 * initCounter_);
    Eigen::VectorXd shape;
    shape = shape0s_.segment<shapeDim>(shapeDim * initCounter_);

//    shape.setZero(shapeDim);

    if (initCounter_ % noisifyDynamicsInterval_ == 0 || changeShape_) {
      this->setShape(shape);
      changeShape_ = false;
    }
    initCounter_++;
    if (initCounter_ == initialStateCapacity_) initCounter_ = 0;
  }

  void setShape(Eigen::VectorXd &shapesIn) {
    shapes_ = shapesIn;
    /// apply shapes
    int shapeIdx = 0;
    for (int i = 0; i < 4; i++) {

      double x_, y_, z_;

      x_ = shapes_[shapeIdx++]; //0
      y_ = shapes_[shapeIdx++]; //1
      z_ = shapes_[shapeIdx++]; //2

      int hipIdx = 3 * i + 1;
      int shankIdx = 3 * i + 3;

      ///hip
      anymal_->getJointPos_P()[hipIdx][0] = defaultJointPositions_[hipIdx][0] + x_;
      anymal_->getJointPos_P()[hipIdx][1] = defaultJointPositions_[hipIdx][1] + y_;
      anymal_->getJointPos_P()[hipIdx][2] = defaultJointPositions_[hipIdx][2] + z_;

      ///protector
      int protectorIdx = 9 * i + 5;
      double dr = shapes_[shapeIdx++]; //3
      double temp = shapes_[shapeIdx++]; //4

      anymal_->getVisColProps()[protectorIdx].second[0] =
          defaultCollisionBodyProps_[protectorIdx][0] + dr;
      anymal_->getVisColProps()[protectorIdx].second[1] =
          defaultCollisionBodyProps_[protectorIdx][1] + temp;

      if (i > 1) dr *= -1.0; // to avoid internal col
      std::get<1>(anymal_->getVisColOb()[protectorIdx])[0] =
      std::get<1>(anymal_->getCollisionObj()[protectorIdx])[0] =
          defaultCollisionBodyPositions_[protectorIdx][0] + dr; // position

      protectorIdx++;
      anymal_->getVisColProps()[protectorIdx].second[0] =
          defaultCollisionBodyProps_[protectorIdx][0] + dr;

      double dy_ = shapes_[shapeIdx++]; //5

      /// shank
      if (i % 2 == 1) {
        y_ = -dy_;
      } else {
        y_ = dy_;
      }

      x_ = shapes_[shapeIdx++]; //6
      z_ = shapes_[shapeIdx++]; //7

      anymal_->getJointPos_P()[shankIdx].v[0] = defaultJointPositions_[shankIdx][0] + x_;
      anymal_->getJointPos_P()[shankIdx].v[1] = defaultJointPositions_[shankIdx][1] + y_;
      anymal_->getJointPos_P()[shankIdx].v[2] = defaultJointPositions_[shankIdx][2] + z_;

      shankIdx = 9 * i + 10;

      /// KFE actuator (note: attached to thigh)
      std::get<1>(anymal_->getVisColOb()[shankIdx])[0] =
      std::get<1>(anymal_->getCollisionObj()[shankIdx])[0] =
          defaultCollisionBodyPositions_[shankIdx][0] + x_; // position
      std::get<1>(anymal_->getVisColOb()[shankIdx])[1] =
      std::get<1>(anymal_->getCollisionObj()[shankIdx])[1] =
          defaultCollisionBodyPositions_[shankIdx][1] + y_; // position
      std::get<1>(anymal_->getVisColOb()[shankIdx])[2] =
      std::get<1>(anymal_->getCollisionObj()[shankIdx])[2] =
          defaultCollisionBodyPositions_[shankIdx][2] + z_; // position

      dr = shapes_[shapeIdx++]; //8
      double dl = shapes_[shapeIdx++]; //9

      anymal_->getVisColProps()[shankIdx].second[0] =
          defaultCollisionBodyProps_[shankIdx][0] + dr;
      anymal_->getVisColProps()[shankIdx].second[1] =
          defaultCollisionBodyProps_[shankIdx][1] + dl;

      shankIdx++;

      x_ = shapes_[shapeIdx++]; //10

      /// shank box
      std::get<1>(anymal_->getVisColOb()[shankIdx])[0] =
      std::get<1>(anymal_->getCollisionObj()[shankIdx])[0] =
          defaultCollisionBodyPositions_[shankIdx][0] + x_; // position

      /// Foot
      int footIdx = 9 * i + 13;
      double dz_ = shapes_[shapeIdx++]; //11
      std::get<1>(anymal_->getVisColOb()[footIdx])[0] =
      std::get<1>(anymal_->getCollisionObj()[footIdx])[0] = defaultCollisionBodyPositions_[footIdx][0] + x_; // position
      std::get<1>(anymal_->getVisColOb()[footIdx])[2] =
      std::get<1>(anymal_->getCollisionObj()[footIdx])[2] =
          defaultCollisionBodyPositions_[footIdx][2] + dz_;

      footPos_[i] = std::get<1>(anymal_->getCollisionObj()[footIdx]);

      /// shank cylinder
      shankIdx++;
      std::get<1>(anymal_->getVisColOb()[shankIdx])[0] =
      std::get<1>(anymal_->getCollisionObj()[shankIdx])[0] =
          defaultCollisionBodyPositions_[shankIdx][0] + x_; // position
      std::get<1>(anymal_->getVisColOb()[shankIdx])[2] =
      std::get<1>(anymal_->getCollisionObj()[shankIdx])[2] =
          defaultCollisionBodyPositions_[shankIdx][2] + dz_ * 0.5; // position
      anymal_->getVisColProps()[shankIdx].second[1] =
          defaultCollisionBodyProps_[shankIdx][1] - dz_;

      /// Foot radius
      dr = shapes_[shapeIdx++]; //12
      footR_[i] = defaultCollisionBodyProps_[footIdx][0] + dr;
      anymal_->getVisColProps()[footIdx].second[0] = footR_[i];

      /// modify protector to avoid collision when rotating shank
      protectorIdx = 9 * i + 7;

      double dr2 = shapes_[shapeIdx++]; //13
      dl = shapes_[shapeIdx++]; //14

      anymal_->getVisColProps()[protectorIdx].second[0] =
          defaultCollisionBodyProps_[protectorIdx][0] + dr2;
      anymal_->getVisColProps()[protectorIdx].second[1] =
          defaultCollisionBodyProps_[protectorIdx][1] + dl; // cylinder length

      /// shank
      if (i % 2 == 1) {
        dl *= -1;
      }
      std::get<1>(anymal_->getVisColOb()[protectorIdx])[1] =
      std::get<1>(anymal_->getCollisionObj()[protectorIdx])[1] =
          defaultCollisionBodyPositions_[protectorIdx][1] + 0.5 * dl; // position

      env_->renewCollisionObjectShape(anymal_, 9 * i + 5);
      env_->renewCollisionObjectShape(anymal_, 9 * i + 6);
      env_->renewCollisionObjectShape(anymal_, 9 * i + 7);
      env_->renewCollisionObjectShape(anymal_, 9 * i + 8);
      env_->renewCollisionObjectShape(anymal_, 9 * i + 10);
      env_->renewCollisionObjectShape(anymal_, 9 * i + 12);
      env_->renewCollisionObjectShape(anymal_, 9 * i + 13);
    }

    /// base attachments
    for (int i = 1; i < 3; i++) {
      double dy = shapes_[shapeIdx++];
      double dz = shapes_[shapeIdx++];
      if (i == 2) dy = 0; // temp fix
      std::get<1>(anymal_->getVisColOb()[i])[1] =
      std::get<1>(anymal_->getCollisionObj()[i])[1] =
          defaultCollisionBodyPositions_[i][1] + dy;

      std::get<1>(anymal_->getVisColOb()[i])[2] =
      std::get<1>(anymal_->getCollisionObj()[i])[2] =
          defaultCollisionBodyPositions_[i][2] + dz;
    }

    {
      double dy = shapes_[shapeIdx++];
      double dz = shapes_[shapeIdx++];

      std::get<1>(anymal_->getVisColOb()[3])[1] =
      std::get<1>(anymal_->getCollisionObj()[3])[1] =
          defaultCollisionBodyPositions_[3][1] + dy;

      std::get<1>(anymal_->getVisColOb()[4])[1] =
      std::get<1>(anymal_->getCollisionObj()[3])[1] =
          defaultCollisionBodyPositions_[4][1] + dy;

      std::get<1>(anymal_->getVisColOb()[3])[2] =
      std::get<1>(anymal_->getCollisionObj()[3])[2] =
          defaultCollisionBodyPositions_[3][2] + dz * 0.5;

      std::get<1>(anymal_->getVisColOb()[4])[2] =
      std::get<1>(anymal_->getCollisionObj()[4])[2] =
          defaultCollisionBodyPositions_[4][2] + dz;
    }

    double dz = shapes_[shapeIdx++];
    std::get<1>(anymal_->getVisColOb()[0])[2] =
    std::get<1>(anymal_->getCollisionObj()[0])[2] =
        defaultCollisionBodyPositions_[0][2] + dz;

    double dwidth = shapes_[shapeIdx++];
    anymal_->getVisColProps()[0].second[1] = defaultCollisionBodyProps_[0][1] + dwidth;
    env_->renewCollisionObjectShape(anymal_, 0);

    anymal_->ignoreCollisionClear();
    for (int i = 0; i < 4; i++) {
      int idx1 = 9 * i + 9;
      int idx2 = 9 * i + 11;
      anymal_->ignoreCollisionBetween2(idx1, idx2);

      idx1 = 9 * i + 7;
      idx2 = 9 * i + 13;
      anymal_->ignoreCollisionBetween2(idx1, idx2);

      idx1 = 9 * i + 5;
      idx2 = 9 * i + 7;
      anymal_->ignoreCollisionBetween2(idx1, idx2);

      idx2 = 9 * i + 8;
      anymal_->ignoreCollisionBetween2(idx1, idx2);

      idx1 = 9 * i + 6;
      anymal_->ignoreCollisionBetween2(idx1, 0);

      idx1 = 9 * i + 10;
      idx2 = 9 * i + 11;
      anymal_->ignoreCollisionBetween2(idx1, idx2);
    }
  }

  void init0() {
    initTasks();

    q_ = q0;
    u_ = u0;
    loggingStep_ = 0;

    for (int i = 0; i < 18; i++) {
      u_(i) = u_initialNoiseScale(i) * rn_.sampleUniform() * 0.5; // sample uniform
    }

    anymal_->setGeneralizedCoordinate(q_);
    anymal_->setGeneralizedVelocity(u_);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(anymal_->getDOF()));

    for (int i = 0; i < 12; i++)
      previousAction_(i) = 0.6 * rn_.sampleNormal() * noiseFtr_;

    for (int i = 0; i < HistoryLength * 12; i++)
      jointVelHist_(i) = 5.0 * rn_.sampleNormal() * noiseFtr_;

    for (int i = 0; i < HistoryLength * 12; i++)
      jointPosHist_(i) = 0.05 * rn_.sampleNormal() * noiseFtr_;

    for (int i = 0; i < HistoryLength * 12; i++)
      torqueHist_(i) = 5.0 * rn_.sampleNormal() * noiseFtr_;

    if (test_) {
      previousAction_.setZero();
      jointVelHist_.setZero();
      jointPosHist_.setZero();
      torqueHist_.setZero();
    }
  }

  void init() {
    timer->startTimer("init");
    loggingStep_ = 0;
    initTasks();

    steps_ = 0;
    tau_.setZero();
    badlyConditioned_ = false;
    costMax_ = this->valueAtTermination_;

    Dtype angle;

    if (this->visualization_ON_ && vis_on_) {
      anymal_.visual()[0]->setColor({0.7f, 0.7f, 1.0f});
    }

    double rn = rn_.sampleUniform01();

    if (rn >= 0.95) {
      q_ << 0.0, 0.0, 0.14, 1.0, 0.0, 0.0, 0.0,
          0, M_PI / 2.0, -2.45,
          0, M_PI / 2.0, -2.45,
          0.0, -M_PI / 2.0, 2.45,
          0.0, -M_PI / 2.0, 2.45;
      for (int i = 0; i < 18; i++) {
        u_(i) = u_initialNoiseScale(i) * rn_.sampleUniform() * 0.5; // sample uniform
      }

      anymal_->setGeneralizedCoordinate(q_);
      anymal_->setGeneralizedVelocity(u_);
      int cnt = 0;
      do {
        cnt++;
        this->passiveStep(0.0025, 0.1);
      } while (cnt < 50);
      q_ = anymal_->getGeneralizedCoordinate();
      u_ = anymal_->getGeneralizedVelocity();

//    } else if (rn >= 0.85) {
//      q_ = q0;
//
//      for (int i = 7; i < 19; i++) {
//        q_(i) += q_initialNoiseScale(i) * rn_.sampleNormal(); // sample uniform
//      }
//
//      Eigen::Vector3d heading;
//      heading(0) = 1.0 - 2.0 * rn_.intRand(0, 1);
//      heading(1) = 0.2 * rn_.sampleUniform();
//      heading(2) = 0.2 * rn_.sampleUniform();
//      heading.normalize();
//
//      angle = 2.0 * rn_.sampleUniform01();
//
//      double sin = std::sin(angle / 2.0);
//      q_.template segment<3>(4) = heading * sin;
//      q_(3) = std::cos(angle / 2.0);
//      q_(2) = 0.4;
//
//      for (int i = 0; i < 18; i++) {
//        u_(i) = u_initialNoiseScale(i) * rn_.sampleUniform() * 0.5; // sample uniform
//      }
//
//      anymal_->setGeneralizedCoordinate(q_);
//      anymal_->setGeneralizedVelocity(u_);
//      for (int i = 0; i < 200; i++)
//        this->passiveStep(0.005, 0.1);
//
//      q_ = anymal_->getGeneralizedCoordinate();
//      u_ = anymal_->getGeneralizedVelocity();
    } else {
      this->initFromSample();
      u_.setZero();
    }

    uPrev_ = u_;
    acc_.setZero();

    anymal_->setGeneralizedCoordinate(q_);
    anymal_->setGeneralizedVelocity(u_);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(anymal_->getDOF()));

    for (int i = 0; i < 12; i++)
      previousAction_(i) = 0.6 * rn_.sampleNormal() * noiseFtr_;

    for (int i = 0; i < HistoryLength * 12; i++)
      jointVelHist_(i) = 5.0 * rn_.sampleNormal() * noiseFtr_;

    for (int i = 0; i < HistoryLength * 12; i++)
      jointPosHist_(i) = 0.05 * rn_.sampleNormal() * noiseFtr_;

    for (int i = 0; i < HistoryLength * 12; i++)
      torqueHist_(i) = 5.0 * rn_.sampleNormal() * noiseFtr_;

    if (test_) {
      previousAction_.setZero();
      jointVelHist_.setZero();
      jointPosHist_.setZero();
      torqueHist_.setZero();
    }
    timer->stopTimer("init");

    if (this->visualization_ON_ && vis_on_) {
      anymal_.visual()[0]->setColor({0.4f, 0.4f, 0.4f});
    }
  }

  void initTo(const State &state) {
    initTasks();
    badlyConditioned_ = false;
    costMax_ = this->valueAtTermination_;

    State state_temp = state;
    conversion_LearningState2GeneralizedState(state_temp, q_, u_);
    jointVelHist_.setZero();
    jointPosHist_.setZero();
    torqueHist_.setZero();

    anymal_->setGeneralizedCoordinate(q_);
    anymal_->setGeneralizedVelocity(u_);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(anymal_->getDOF()));
  }

  void noisifyState(StateBatch &stateBatch) {

    VectorXd q_temp(19), u_temp(18);
    Vector4d contac_temp(4);
    Vector12d jointTorque_temp;
    State state_temp;

    for (int colID = 0; colID < stateBatch.cols(); colID++) {
      state_temp = stateBatch.col(colID);
      conversion_LearningState2GeneralizedState(state_temp, q_temp, u_temp);
      for (int i = 0; i < 19; i++)
        q_temp(i) += q_initialNoiseScale(i) * rn_.sampleNormal() * noiseFtr_;
      q_temp.segment(3, 4) /= q_.segment(3, 4).norm();

      for (int i = 0; i < 18; i++)
        u_temp(i) += u_initialNoiseScale(i) * rn_.sampleNormal() * noiseFtr_;

      conversion_GeneralizedState2LearningState(state_temp, q_temp, u_temp);
      stateBatch.col(colID) = state_temp;
    }
  }

  void getState(State &state) {
    conversion_GeneralizedState2LearningState(state, q_, u_);
  }

  Eigen::VectorXd getGeneralizedState() { return q_; }

  Eigen::VectorXd getGeneralizedVelocity() { return u_; }

  void setRealTimeFactor(double fctr) {
    realTimeRatio_ = fctr;
  }

  void setNoiseFtr(double fctr) {
    noiseFtr_ = fctr;
  }

//	task specific implementations
  inline void integrateOneTimeStep() {
    Eigen::VectorXd q_temp = q_;
    Eigen::VectorXd u_temp = u_;

    env_->integrate1();
    anymal_->setGeneralizedForce(tau_);
    env_->integrate2();

    q_ = anymal_->getGeneralizedCoordinate();
    u_ = anymal_->getGeneralizedVelocity();

    Quaternion quat = q_.segment<4>(3);
    R_b_ = rai::Math::MathFunc::quatToRotMat(quat);

    RAIWARN_IF(isnan(u_.norm()), "error in simulation!!" << std::endl
                                                         << "action" << scaledAction_.transpose() << std::endl
                                                         << "q_" << q_.transpose() << std::endl
                                                         << "u_" << u_.transpose() << std::endl
                                                         << "q_prev" << q_temp.transpose() << std::endl
                                                         << "u_prev" << u_temp.transpose());

    if (isnan(u_.norm())) badlyConditioned_ = true;
    if (isinf(u_.norm())) badlyConditioned_ = true;
    if (isnan(q_.norm())) badlyConditioned_ = true;
    if (isinf(q_.norm())) badlyConditioned_ = true;
    if (std::abs(q_.norm()) > 1000) badlyConditioned_ = true;

    comprehendContacts();

    if (this->visualization_ON_ && vis_on_) {
      if (!vis_ready_) {
        env_->visStart();
      }

      double waitTime = std::max(0.0, simulation_dt_ / realTimeRatio_ - watch_.measure());
      usleep(waitTime * 1e6);
      watch_.start();
      if (viswatch_.measure() > 1.0 / 80.0) {
        env_->updateFrame();
        viswatch_.start();
      }
    }
  }

  void startRecordingVideo(std::string dir, std::string fileName) {
    if (this->visualization_ON_ && vis_on_) {
      env_->startRecordingVideo(dir, fileName);
      vid_on_ = true;
    }
  }

  void endRecordingVideo() {
    if (vid_on_)env_->stopRecordingVideo();
    vid_on_ = false;
  }

 private:

  inline void calculateCost(Dtype &cost) {

    double torqueCost = 0, linvelCost = 0, angVelCost = 0, velLimitCost = 0, footClearanceCost = 0,
        slipCost = 0;
    double yawRateError = u_(5) - command_[2];
    Vector2d horizontalVelocity = (R_b_.transpose() * u_.segment<3>(0)).head(2);
    double bodyHeightCost;
    double actionCost = 0;
    Quaternion quat = q_.template segment<4>(3);
    RotationMatrix R = rai::Math::MathFunc::quatToRotMat(quat);
    ///common values
    torqueCost = tau_.tail(12).norm() * simulation_dt_;

    for (int i = 0; i < 4; i++) {
      if (!footContactState_[i])
        footClearanceCost +=
            pow(std::max(0.0, 0.1 - footPos_W[i][2]), 2) * std::min(footVel_W[i].e().head(2).norm(), 5.0)
                * simulation_dt_;
      else
        slipCost += 0.01 * std::min(footContactVel_[i].head(2).norm(), 5.0) * simulation_dt_;
    }

    ////////////////////////////////////////////////////////////////////////////////
    double flipCost = 0;
    double standCost = 0;
    double extraCost = 0;
    double accCost = 0;
    cost = 0;

    double haaCost = 0;
    double hfeCost = 0;
    double kfeCost = 0;
    double footCost = 0;

    double haaError[4];
    double hfeError[4];
    double kfeError[4];

    double r, p, y;
    double r_abs, r_squared;
    double M_PI_6 = M_PI / 6.0;
    rai::Math::MathFunc::QuattoEuler(quat, r, p, y);
    r_abs = std::abs(r);

    for (int i = 0; i < 18; i++) {
      double acc_clipped = std::min(std::abs(acc_[i]), 1000.0);
      accCost += acc_clipped * acc_clipped;
    }
    accCost *= simulation_dt_;

    velLimitCost = 0;
    for (int i = 6; i < 18; i++) {
      double uAbs = std::fabs(u_(i));
      if (uAbs > 8)
        velLimitCost += std::min(uAbs, 20.0) * std::min(uAbs, 20.0);
    }

    velLimitCost *= simulation_dt_;

    /////////////////////////core cost////////////////////////////
    /// orientation

    flipCost = 7.0 * ((R(2, 2) - 1) * (R(2, 2) - 1) - 4) * simulation_dt_; // orientation

    flipCost -= 6.0 * (4 - std::min((int) numInternalContact_, 4)) * simulation_dt_; /// avoid internal contacts

    /// Joint position errors
    for (int i = 0; i < 4; i++) {
//        haaError[i] = q_(7 + 3 * i); // should go back to 0: not to rotate too much

      haaError[i] = anglemod(q_(7 + 3 * i));
      double haaExp = std::exp(7.0 * haaError[i]);
      haaCost -= haaExp / (1 + haaExp * haaExp);
      if (std::abs(q_(7 + 3 * i)) > 2.5 * M_PI) haaCost += 0.5;

//        if (std::abs(haaError[i]) < 0.2) {
      if (i < 2) { //front
        hfeError[i] = anglediff(M_PI_2, q_(8 + 3 * i));
        kfeError[i] = anglediff(-2.45, q_(9 + 3 * i));
      } else { //hind
        hfeError[i] = anglediff(-M_PI_2, q_(8 + 3 * i));
        kfeError[i] = anglediff(2.45, q_(9 + 3 * i));
      }
      double hfeExp = std::exp(7.0 * hfeError[i]);
      double kfeExp = std::exp(7.0 * kfeError[i]);

      hfeCost -= hfeExp / (1 + hfeExp * hfeExp);
//        if (std::abs(hfeError[i]) < 0.5) kfeCost -= kfeExp / (1 + kfeExp * kfeExp);
      kfeCost -= kfeExp / (1 + kfeExp * kfeExp);
//        }
    }

    if (haaError[0] < -1.55 || haaError[0] > 2.1) {
      haaCost += 0.2;
      haaCost += 0.01 * std::abs(u_[6]);
    }

    if (haaError[1] < -2.1 || haaError[1] > 1.55) {
      haaCost += 0.2;
      haaCost += 0.01 * std::abs(u_[9]);
    }

    if (r_abs < M_PI_4) { ///
      flipCost += 4.0 * haaCost * simulation_dt_;
//    }
//    if (r_abs < 0.3) {
//      flipCost -= 1.0 * simulation_dt_;
      flipCost += 4.0 * hfeCost * simulation_dt_;
      flipCost += 4.0 * kfeCost * simulation_dt_;
    }

    cost += flipCost;

    if (numBodyContact_ > 0) {
      extraCost = 5.0 * std::min(sumBodyImpulse_ / numBodyContact_, 5.0) * simulation_dt_;
      extraCost += 5.0 * costScale_ * std::min(sumBodyContactVel_ / numBodyContact_, 1.0) * simulation_dt_;
    }

//      extraCost += 8.0 * costScale_ * slipCost;
    extraCost += 0.2 * costScale2_ * velLimitCost;
    extraCost += 5e-7 * (0.5 + costScale2_) * accCost;
    extraCost += 0.0025 * costScale2_ * scaledAction_.squaredNorm() * simulation_dt_;
    extraCost += 0.0005 * torqueCost;

    cost += extraCost;

    if (isinf(cost)) {
      LOG(INFO) << "inf error in cost function!! " << std::endl;
      LOG(INFO) << "torqueCost " << torqueCost << std::endl;
      LOG(INFO) << "linvelCost " << linvelCost;
      LOG(INFO) << "bodyHeightCost" << bodyHeightCost;
      LOG(INFO) << "angVelCost " << angVelCost;
      LOG(INFO) << "velLimitCost " << velLimitCost;
      LOG(INFO) << "footClearanceCost " << footClearanceCost;
      LOG(INFO) << "u_ " << u_.transpose() << std::endl;
      LOG(INFO) << "tau_" << tau_.transpose() << std::endl;
      LOG(INFO) << "scaledAction_" << scaledAction_.transpose() << std::endl;
      LOG(INFO) << "q_" << q_.transpose() << std::endl;

      LOG(INFO) << "linvelCost" << linvelCost << std::endl
                //             <<"angVelCost" << angVelCost <<std::endl
                //      << "velMultiplier_ * velLimitCost" << velMultiplier_ * velLimitCost<<std::endl
                //      << "torqueMultiplier_ * torqueCost" << torqueMultiplier_ * torqueCost << std::endl
                << "slipMultiplier_ * slipCost" << slipMultiplier_ * slipCost << std::endl;
//                << "contactMultiplier_ * contactCost" << contactMultiplier_ * contactCost << std::endl;
      RAIWARN("BAAAAD COOOOSTTTT" << flipCost / simulation_dt_
                                  << "," << extraCost / simulation_dt_
                                  << ", " << haaCost << ", " << haaError[0]
                                  << ", " << haaError[1] << ", " << haaError[2] << ", "
                                  << haaError[3]
                                  << ", " << q_[7] << ", " << q_[10] << ", " << q_[13] << ", " << q_[16]
                                  << ", " << footCost / simulation_dt_);
      RAIWARN("extracost1" << 10.0 * costScale_ * std::min(sumBodyImpulse_ / (numContact_) / simulation_dt_, 2.0));
      RAIWARN("extracost2" << 10.0 * costScale2_ * slipCost / simulation_dt_);
      RAIWARN("extracost3" << (0.02 + 0.4 * costScale2_) * velLimitCost / simulation_dt_);
      RAIWARN("extracost4" << 2.5 * costScale2_ * scaledAction_.array().abs().maxCoeff());
      RAIWARN("extracost5" << 1e-6 * costScale2_ * accCost / simulation_dt_);
      RAIWARN("extracost6" << 0.05 * torqueCost);
      RAIWARN((numContact_));
      DRAIFATAL("?")
//      exit(0);
    }
    if (isnan(cost)) {
      std::cout << "error in cost function!! " << std::endl;
      std::cout << "torqueCost " << torqueCost << std::endl;
      std::cout << "linvelCost " << linvelCost << std::endl;
      LOG(INFO) << "bodyHeightCost" << bodyHeightCost;
      std::cout << "angVelCost " << angVelCost << std::endl;
      std::cout << "velLimitCost " << velLimitCost << std::endl;
      std::cout << "u_ " << u_.transpose() << std::endl;
      LOG(INFO) << "linvelCost" << linvelCost << std::endl
                //             <<"angVelCost" << angVelCost <<std::endl
                //      << "velMultiplier_ * velLimitCost" << velMultiplier_ * velLimitCost<<std::endl
                //      << "torqueMultiplier_ * torqueCost" << torqueMultiplier_ * torqueCost << std::endl
                << "slipMultiplier_ * slipCost" << slipMultiplier_ * slipCost << std::endl;
//                << "contactMultiplier_ * contactCost" << contactMultiplier_ * contactCost << std::endl;
      RAIWARN("BAAAAD COOOOSTTTT" << flipCost / simulation_dt_
                                  << standCost << kfeCost << hfeCost
                                  << "," << extraCost / simulation_dt_
                                  << ", " << haaCost << ", " << haaError[0]
                                  << ", " << haaError[1] << ", " << haaError[2] << ", "
                                  << haaError[3]
                                  << ", " << q_[7] << ", " << q_[10] << ", " << q_[13] << ", " << q_[16]
                                  << ", " << footCost / simulation_dt_);
      RAIWARN("extracost1" << 10.0 * costScale_ * std::min(sumBodyImpulse_ / (numContact_) / simulation_dt_, 2.0));
      RAIWARN("extracost2" << 10.0 * costScale2_ * slipCost / simulation_dt_);
      RAIWARN("extracost3" << (0.02 + 0.4 * costScale2_) * velLimitCost / simulation_dt_);
      RAIWARN("extracost4" << 2.5 * costScale2_ * scaledAction_.array().abs().maxCoeff());
      RAIWARN("extracost5" << 1e-6 * costScale2_ * accCost / simulation_dt_);
      RAIWARN("extracost6" << 0.05 * torqueCost);
      RAIWARN((numContact_));
      DRAIFATAL("?")

    }

  }

  inline double anglediff(double target, double source) {
    //https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
    return anglemod(target - source);
  }

  inline double anglemod(double a) {
    return wrapAngle((a + M_PI)) - M_PI;
  }

  inline double wrapAngle(double a) {
    double twopi = 2.0 * M_PI;
    return a - twopi * fastfloor(a / twopi);
  }

  inline int fastfloor(double a) {
    int i = int(a);
    if (i > a) i--;
    return i;
  }

  void getInitialState(State &in) {
    in = state0_;
  }

  bool isTerminalState(State &state) {
    ////////// termination due to a constrain violation///////////
    VectorXd u_term, q_term;
    u_term.resize(18);
    q_term.resize(19);
//    Vector4d contactForces_term;
//    Vector12d torque_term;
    const State state_term = state;
    conversion_LearningState2GeneralizedState(state_term, q_term, u_term);
    return isTerminalState(q_term, u_term);
  }

//  *		19 dimension, q = [body_position (3 numbers IDX 0-2)
//  *	 	 	 	 	 	   body_quaternion   (4 numbers IDX 3-6),
//  * 	 	 	 	 	 	   leg1- HAA, HFE, KFE (3 numbers, IDX 7-9)
//  *	 	 	 	 	 	   leg2- ... leg3- ... leg4- ...]

  bool isTerminalState(const VectorXd &q_term, const VectorXd &u_term) {
    ////////// termination due to a constrain violation///////////

    Quaternion quat = q_term.template segment<4>(3);
    double r, p, y;
    rai::Math::MathFunc::QuattoEuler(quat, r, p, y);
    RotationMatrix R = rai::Math::MathFunc::quatToRotMat(quat);

    if (badlyConditioned_) {
      RAIWARN("BAAAAAAAD");
      badlyConditioned_ = false;
      return true;
    }
    ///simulation diverging
    if (q_term.maxCoeff() > 1e3 || q_term.minCoeff() < -1e3) {
      RAIWARN("BAAAAAAAD" << omp_get_thread_num());
      badlyConditioned_ = false;
      return true;
    }

    return false;
  }

  /**
*       learning state =
       *      [command (horizontal velocity, yawrate)                      n =  3, si =   0
       *       z-axis in world frame expressed in body frame (R_b.row(2)), n =  3, si =   3
       *       body Linear velocities,                                     n =  3, si =   6
       *       body Angular velocities,                                    n =  3, si =   9
       *       joint position history(t0, t-2, t-4),                       n = 36, si =  12
       *       joint velocities(t0, t-2, t-4)                              n = 36, si =  48
       *       previous action                                             n = 12, si =  84
       *       ]
   */

  //@warning this mapping cannot fully reconstruct state(h missing)
  inline void conversion_LearningState2GeneralizedState(const State &state,
                                                        VectorXd &q,
                                                        VectorXd &u) {
    // inverse scaling
    State state_unscaled = state.cwiseQuotient(stateScale_) + stateOffset_;

    Vector3d xaxis, yaxis, zaxis;
    zaxis << state_unscaled(3), state_unscaled(4), state_unscaled(5);
    zaxis /= zaxis.norm();
    xaxis << 1, 0, 0;
    yaxis = xaxis.cross(zaxis);
    yaxis /= yaxis.norm();
    xaxis = yaxis.cross(zaxis);

    RotationMatrix R_b;
    R_b.row(0) = xaxis.transpose();
    R_b.row(1) = yaxis.transpose();
    R_b.row(2) = zaxis.transpose();
    Quaternion quat = rai::Math::MathFunc::rotMatToQuat(R_b);

    LinearVelocity bodyVel = R_b * state_unscaled.template segment<3>(6).
        template cast<double>();
    AngularVelocity bodyAngVel = R_b * state_unscaled.template segment<3>(9).
        template cast<double>();
    VectorXd jointVel = state_unscaled.template segment<12>(48).
        template cast<double>();

    q << 0.0, 0.0, 0.5, quat, state_unscaled.template segment<12>(12).
        template cast<double>();
    u << bodyVel.template cast<double>(), bodyAngVel.template cast<double>(), jointVel.template cast<double>();

    command_ = state_unscaled.head(3).template cast<double>();

    jointPosHist_.template segment<12>(12 * HistoryLength - 12) = state_unscaled.template segment<12>(12); // t
    jointPosHist_.template segment<12>(12 * HistoryLength - 36) = state_unscaled.template segment<12>(24); // t-2
    jointPosHist_.template segment<12>(12 * HistoryLength - 60) = state_unscaled.template segment<12>(36); // t-4

    jointVelHist_.template segment<12>(12 * HistoryLength - 12) = state_unscaled.template segment<12>(48); // t
    jointVelHist_.template segment<12>(12 * HistoryLength - 36) = state_unscaled.template segment<12>(60); // t-2
    jointVelHist_.template segment<12>(12 * HistoryLength - 60) = state_unscaled.template segment<12>(72); // t-4

    torqueHist_.setZero();

    previousAction_ = state_unscaled.template segment<12>(84);
  }
//  *		19 dimension, q = [body_position (3 numbers IDX 0-2)
//  *	 	 	 	 	 	   body_quaternion   (4 numbers IDX 3-6),
//  * 	 	 	 	 	 	   leg1- HAA, HFE, KFE (3 numbers, IDX 7-9)
//  *	 	 	 	 	 	   leg2- ... leg3- ... leg4- ...]
  /**
*       learning state =
       *      [command (horizontal velocity, yawrate)                      n =  3, si =   0
       *       z-axis in world frame expressed in body frame (R_b.row(2)), n =  3, si =   3
       *       body Linear velocities,                                     n =  3, si =   6
       *       body Angular velocities,                                    n =  3, si =   9
       *       joint position history(t0, t-2, t-4),                       n = 36, si =  12
       *       joint velocities(t0, t-2, t-4)                              n = 36, si =  48
       *       previous action                                             n = 12, si =  84
       *       ]
   */

  inline void conversion_GeneralizedState2LearningState(State &state,
                                                        const VectorXd &q,
                                                        const VectorXd &u) {

    Quaternion quat = q.template segment<4>(3);
    RotationMatrix R_b = rai::Math::MathFunc::quatToRotMat(quat);
    State state_unscaled;
    state_unscaled.head(3) = command_.template cast<Dtype>();
//    state_unscaled[0] = 0.0;
//    state_unscaled[1] = 0.0;
//    state_unscaled[2] = 0.0;

    state_unscaled.template segment<3>(3) = R_b.row(2).transpose().template cast<Dtype>();
//    std::cout <<  R_b.row(2) << std::endl;
    /// velocity in body coordinate
    LinearVelocity bodyVel = R_b.transpose() * u.template segment<3>(0);
    AngularVelocity bodyAngVel = R_b.transpose() * u.template segment<3>(3);
    VectorXd jointVel = u.template segment<12>(6);

    state_unscaled.template segment<3>(6) = bodyVel.template cast<Dtype>();
    state_unscaled.template segment<3>(9) = bodyAngVel.template cast<Dtype>();

    state_unscaled.template segment<12>(12) = q.template segment<12>(7).
        template cast<Dtype>(); /// position
    state_unscaled.template segment<12>(48) = jointVel.template cast<Dtype>();

    state_unscaled.template segment<12>(24) = jointPosHist_.template segment<12>(12 * HistoryLength - 36);
    state_unscaled.template segment<12>(36) = jointPosHist_.template segment<12>(12 * HistoryLength - 60);

    state_unscaled.template segment<12>(60) = jointVelHist_.template segment<12>(12 * HistoryLength - 36);
    state_unscaled.template segment<12>(72) = jointVelHist_.template segment<12>(12 * HistoryLength - 60);

    state_unscaled.template segment<12>(84) = previousAction_;

//    std::cout << previousAction_.transpose() << std::endl;
    // scaling
    state = (state_unscaled - stateOffset_).cwiseProduct(stateScale_);
  }

/// task spec
 public:
  double initAngle_;
  int instance_;
  int taskLabel_;

  Eigen::VectorXd shapes_;
  bool changeShape_;
  bool noisify_;
  bool test_;
 private:
  double desiredHeight_;
  double velMultiplier_;
  double torqueMultiplier_;
  double contactMultiplier_;
  double slipMultiplier_;
  double costScale_;
  double costScale2_;
  double termHeight_;
  double costMax_;
  Eigen::VectorXd uPrev_;
  Eigen::VectorXd acc_;
  int steps_;
  int maxSteps_;
  int initCounter_;
  int noisifyDynamicsInterval_;

  /// sim
  rai_sim::ArticulatedSystemHandle anymal_;
  rai_sim::SingleBodyHandle terrain_;
  std::unique_ptr<rai_sim::World_RG> env_;
  rai_sim::MaterialManager materials_;
  int terrainKey_, robotKey_;
  Eigen::Vector3d gravity_;

  rai::FuncApprox::MLP_fullyconnected<double, 6, 1, rai::FuncApprox::ActivationType::softsign> actuator_;

  Eigen::VectorXd u_, u_initialNoiseScale, u0;
  Eigen::VectorXd q_, q_initialNoiseScale, q0;
  Eigen::VectorXd q0s, shape0s_;
  int initialStateCapacity_;

  Vector18d tau_;
  Eigen::Matrix<double, 12, 1> actionMax_, actionMin_;

  Vector12d tauMax_, tauMin_;
  Matrix3d R_b_;
  Vector12d jointNominalConfig_;

  constexpr static double simulation_dt_ = 0.0025;

  // innerStates
  std::vector<Position> defaultJointPositions_;
  std::vector<Position> defaultCollisionBodyPositions_;
  std::vector<Position> defaultCollisionBodyProps_;
  std::vector<double> defaultBodyMasses_;

  std::vector<rai_sim::Vec<3>> footPos_;
  std::vector<rai_sim::Vec<3>> footPos_W;
  std::vector<rai_sim::Vec<3>> footVel_W;
  double footR_[4];

  std::vector<Position> footContactVel_;
  std::vector<Position> footNormal_;
  Position COMPosition_;

  // history buffers
  Eigen::Matrix<Dtype, 12 * HistoryLength, 1> jointVelHist_, jointPosHist_;
  Eigen::Matrix<Dtype, 12 * HistoryLength, 1> torqueHist_;

  // Buffers for contact states
  std::array<bool, 4> footContactState_;
  std::array<bool, 4> shankContactState_;
  std::array<bool, 4> thighContactState_;

  size_t numContact_;
  size_t numFootContact_;
  size_t numThighContact_;
  size_t numHipContact_;
  size_t numInternalContact_;
  size_t numBodyContact_;
  size_t numShankContact_;

  double sumBodyImpulse_;
  double sumBodyContactVel_;

  // Check for divergence
  bool badlyConditioned_;

  // Visualize
  double realTimeRatio_;
  bool vis_on_ = false;
  bool vis_ready_;
  bool vid_on_;
  int loggingStep_;

  // learning params
  Vector3d command_;
  State state0_;
  State stateOffset_;
  State stateScale_;
  Action actionOffset_;
  Action actionScale_;
  Action scaledAction_;
  Eigen::Matrix<double, 12, 1> targetPosition_;
  Action previousAction_;

  StopWatch watch_;
  StopWatch viswatch_;

  double noiseFtr_ = 1.0;

  rai::RandomNumberGenerator<Dtype> rn_;
};

}
} //namespaces