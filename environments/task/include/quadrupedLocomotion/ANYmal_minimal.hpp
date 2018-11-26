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
       *       height                                                      n =  1  si =   3
       *       z-axis in world frame expressed in body frame (R_b.row(2)), n =  3, si =   4
       *       body Linear velocities,                                     n =  3, si =   7
       *       body Angular velocities,                                    n =  3, si =  10
       *       joint position history(t0, t-2, t-4),                       n = 36, si =  13
       *       joint velocities(t0, t-2, t-4)                              n = 36, si =  49
       *       previous action                                             n = 12, si =  85
       *       ]

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
constexpr int StateDim = 97;
constexpr int CommandDim = 0;
constexpr int HistoryLength = 15;
constexpr int SmoothingWindow = 30;
constexpr int shapeDim = 68;

template<typename Dtype>
class ANYmal_minimal : public Task<Dtype, StateDim, ActionDim, CommandDim> {

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

  ANYmal_minimal() = delete;

  explicit ANYmal_minimal(bool visualize = false,
                          int instance = 0,
                          std::string actuatorParamPath = "/home/joonho/workspace/oldrai/src/anymal_raisim/data/seaModel_10000.txt",
                          std::string urdfpath = "/home/joonho/workspace/oldrai/src/anymal_raisim/task/include/quadrupedLocomotion/model/robot_nofan_nolimit.urdf"
  ) :
      vis_on_(visualize),
      vis_ready_(false),
      vid_on_(false),
      actuator_(actuatorParamPath, {32, 32}),
      instance_(instance) {

    //// set default parameters
    this->valueAtTermination_ = 10.0;
    this->discountFactor_ = 0.995;
    this->timeLimit_ = 5.0;
    this->controlUpdate_dt_ = 0.01;

    /// parameters for dynamics
    jointNominalConfig_ << 0.0, 0.4, -0.8, 0.0, 0.4, -0.8, 0.0, -0.4, 0.8, 0.0, -0.4, 0.8;

    q_.setZero(19);
    u_.setZero(18);
    q0.resize(19);

    q0 << 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, jointNominalConfig_;
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
        0.44,  /// height
        0.0, 0.0, 0.0, /// gravity axis
        VectorXD::Constant(6, 0.0), /// body lin/ang vel
        jointNominalConfig_.template cast<Dtype>(), /// joint position
        VectorXD::Constant(24, 0.0), /// position error
        VectorXD::Constant(36, 0.0), /// joint velocity history
        VectorXD::Constant(12, 0.0); /// prev. action

    stateScale_ << 1.0, 1.0 / 0.3, 1.0, /// command
        5.0, /// height
        VectorXD::Constant(3, 1.0), /// gravity axis
        1.0 / 1.5, 1.0 / 0.5, 1.0 / 0.5, 1.0 / 2.5, 1.0 / 2.5, 1.0 / 2.5, /// linear and angular velocities
        VectorXD::Constant(36, 1 / 1.0), /// joint angles
        VectorXD::Constant(36, 1 / 10.0), /// joint velocities
        VectorXD::Constant(12, 1.0 / 1.0); /// prev. action

    /// action params
    actionScale_.setConstant(0.5);
    actionOffset_ = jointNominalConfig_.template cast<Dtype>();

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
    gravity_ << 0, 0, -9.81;
    env_->setGravity(gravity_);

    /// terrain
    int ySamples = 100;
    int xSize = 10;
    int ySize = 10;
    int stride = 100;

    int loop = 1;

    std::vector<double> heights;
    for (int i = 0; i < ySamples; i++) {
      for (int j = 0; j < loop; j++) {
        for (int k = 0; k < stride; k++) {
          double depth;
          depth = 0.005 * rn_.sampleUniform01();
          if (j % 2 == 1) depth = 0;
          heights.push_back(depth);
        }
      }
    }
    int xSamples = loop * stride;

    heights.resize(xSamples * ySamples, 0);

//    terrain_ = env_->addHeightMap(xSamples, ySamples, xSize, ySize, 0, 0, heights);
    terrain_ = env_->addCheckerboard(0.5,20,20,0.2);

    /// env visual
    if (vis_on_) {
      env_->cameraFollowObject(anymal_, {1.2, 1.2, 1.0});
      env_->visStart();
//      rai_graphics::object::CheckerBoard
//          *board = dynamic_cast<rai_graphics::object::CheckerBoard *>(terrain_.visual()[0]);
//      board->setBoardColor({0.0, 0.0, 0.0}, {0.75, 0.75, 1.0});
//      board->setBoardColor({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0});

      std::vector<float> spec = {0.0, 0.0, 0.0}, amb = {5.0, 5.0, 5.0}, diff = {0.0, 0.0, 0.0};
//      board->setLightProp(amb, diff, spec, 0.2);
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
    costScale_ = 0.3;
    costScale2_ = 0.1;

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
    q_initialNoiseScale.setConstant(0.65);
    q_initialNoiseScale.segment(0, 3) << 0.00, 0.00, 0.03;

    u_initialNoiseScale.setConstant(5.0);
    u_initialNoiseScale.segment(3, 3).setConstant(1.15);
    u_initialNoiseScale(0) = 1.0;
    u_initialNoiseScale(1) = 1.0;
    u_initialNoiseScale(2) = 1.0;

    /// inner states
    u0PreviousRandom_.setZero(18);
    q0PreviousRandom_.setZero(19);

    for (int i = 0; i < 4; i++) {
      footContactState_[i] = false;
    }

    numContact_ = 0;
    previousAction_.setZero();
    jointVelHist_.setZero();
    jointPosHist_.setZero();
    torqueHist_.setZero();

    for (int i = 0; i < 4; i++) {
      footPos_.push_back(std::get<1>(anymal_->getCollisionObj()[2 * i + 3]));
      footR_[i] = anymal_->getVisColProps()[2 * i + 3].second[0];
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

    /// initialize Material property
    materials_.setMaterialNames({"terrain", "robot"});
    materials_.setMaterialPairProp("terrain", "robot", 0.8, 0.0, 0.0);
    materials_.setMaterialPairProp("robot", "robot", 0.8, 0.0, 0.0);

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
  }

  ~ANYmal_minimal() {
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

      /// info
      std::ostringstream out;
      out << std::fixed << std::setprecision(4);

      out << "Target : ["
          << command_[0] << ", "
          << command_[1] << ", "
          << command_[2] << "] (" << command_.norm() << ")";

//    out << "\nError : ["
//        << target_[0]-horizontalVelocity_[0] << ", "
//        << target_[1]-horizontalVelocity_[1] << ", "
//        << target_[2]-u_[5] << "]";

      env_->getRaiGraphics()->changeMenuText(0, false, out.str());
    }
  }


  inline void comprehendContacts() {
    numContact_ = anymal_->getContacts().size();


    numFootContact_ = 0;
    numBodyContact_ = 0;
    numBaseContact_ = 0;


    sumBodyImpulse_ = 0;
    sumBodyContactVel_ = 0;

    for (int k = 0; k < 4; k++) {
      footContactState_[k] = false;
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
          if (idx == 3 && footPos_W[0][2] < 1e-1 && !footContactState_[0]){
              footContactState_[0] = true;
              footNormal_[0] = anymal_->getContacts()[k].getNormal().e();
              anymal_->getContactPointVel(k, vec3);
              footContactVel_[0] = vec3.e();
              numFootContact_++;
          } else if (idx == 6 && footPos_W[1][2] < 1e-1 && !footContactState_[1]){
              footContactState_[1] = true;
              footNormal_[1] = anymal_->getContacts()[k].getNormal().e();
              anymal_->getContactPointVel(k, vec3);
              footContactVel_[1] = vec3.e();
              numFootContact_++;
          } else if (idx == 9 && footPos_W[2][2] < 1e-1 && !footContactState_[2]) {
              footContactState_[2] = true;
              footNormal_[2] = anymal_->getContacts()[k].getNormal().e();
              anymal_->getContactPointVel(k, vec3);
              footContactVel_[2] = vec3.e();
              numFootContact_++;
          } else if (idx == 12 && footPos_W[3][2] < 1e-1 && !footContactState_[3]) {
              footContactState_[3] = true;
              footNormal_[3] = anymal_->getContacts()[k].getNormal().e();
              anymal_->getContactPointVel(k, vec3);
              footContactVel_[3] = vec3.e();
              numFootContact_++;
          }else{
            if(idx == 0){
              numBaseContact_++;
            }
            for(size_t i = 0; i<4; i++){
              if(idx == 3*i + 1||idx == 3*i + 2){
                numBodyContact_++;
              }
            }
          }
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
    targetPosition_ = scaledAction_.template cast<double>();

    bool terminate = false;

    for (int i = 0; i < (int) (this->controlUpdate_dt_ / simulation_dt_); i++) {

      Eigen::Matrix<Dtype, HistoryLength * 12 - 12, 1> temp;
      temp = jointVelHist_.tail(HistoryLength * 12 - 12);
      jointVelHist_.head(HistoryLength * 12 - 12) = temp;
      jointVelHist_.tail(12) = u_.tail(12).template cast<Dtype>();

      temp = jointPosHist_.tail(HistoryLength * 12 - 12);
      jointPosHist_.head(HistoryLength * 12 - 12) = temp;
      jointPosHist_.tail(12) = (targetPosition_ - q_.tail(12)).template cast<Dtype>();
      Eigen::Matrix<double, 6, 1> seaInput;
      for (int actId = 0; actId < 12; actId++) {
        seaInput[0] = (jointVelHist_(actId + (HistoryLength - 7) * 12) + 0.003) * 0.474;
        seaInput[1] = (jointVelHist_(actId + (HistoryLength - 4) * 12) + 0.003) * 0.473;
        seaInput[2] = (jointVelHist_(actId + (HistoryLength - 1) * 12) + 0.003) * 0.473;

        seaInput[3] = (jointPosHist_(actId + (HistoryLength - 7) * 12) + 0.005) * 7.629;
        seaInput[4] = (jointPosHist_(actId + (HistoryLength - 4) * 12) + 0.005) * 7.629;
        seaInput[5] = (jointPosHist_(actId + (HistoryLength - 1) * 12) + 0.005) * 7.628;

        tau_(6 + actId) = actuator_.forward(seaInput)[0] * 20.0;
      }

      for(int k=6; k<18; k++) {
        tau_(k) = std::min(std::max(tauMin_(k-6), tau_(k)), tauMax_(k-6));
      }
      tau_.head(6).setZero();

      integrateOneTimeStep();

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


    /**
     *
   *
*       learning state =
       *      [command (horizontal velocity, yawrate)                      n =  3, si =   0
       *       height                                                      n =  1  si =   3
       *       z-axis in world frame expressed in body frame (R_b.row(2)), n =  3, si =   4
       *       body Linear velocities,                                     n =  3, si =   7
       *       body Angular velocities,                                    n =  3, si =  10
       *       joint position history(t0, t-2, t-4),                       n = 36, si =  13
       *       joint velocities(t0, t-2, t-4)                              n = 36, si =  49
       *       previous action                                             n = 12, si =  85
       *       ]
     */

    /// noisify body orientation
    for (int i = 4; i < 7; i++)
      state_tp1[i] += rn_.sampleUniform() * 0.01;

    if (numFootContact_ < 2) {
      /// noisify body vel
      for (int i = 7; i < 13; i++)
        state_tp1[i] += rn_.sampleUniform() * 2.0;
    } else {
      /// noisify body vel
      for (int i = 7; i < 13; i++)
        state_tp1[i] += rn_.sampleUniform() * 0.05;
    }

    /// noisify joint vel
    for (int i = 49; i < 85; i++)
      state_tp1[i] += rn_.sampleUniform() * 0.05;

    previousAction_ = targetPosition_.template cast<Dtype>();
    steps_++;
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
      int thighIdx = 3 * i + 2;
      int shankIdx = 3 * i + 3;

      ///hip
      anymal_->getJointPos_P()[hipIdx][0] = defaultJointPositions_[hipIdx][0] + x_;
      anymal_->getJointPos_P()[hipIdx][1] = defaultJointPositions_[hipIdx][1] + y_;
      anymal_->getJointPos_P()[hipIdx][2] = defaultJointPositions_[hipIdx][2] + z_; ///1


      /// thigh
      x_ = -rn_.sampleUniform01() * 0.02;
      y_ = rn_.sampleUniform() * 0.02;
      z_ = rn_.sampleUniform() * 0.02;

      anymal_->getJointPos_P()[thighIdx][0] = defaultJointPositions_[hipIdx][0] + x_;
      anymal_->getJointPos_P()[thighIdx][1] = defaultJointPositions_[hipIdx][1] + y_;
      anymal_->getJointPos_P()[thighIdx][2] = defaultJointPositions_[hipIdx][2] + z_; ///1

      /// shank
      double dy_ = rn_.sampleUniform() * 0.01;
      shapes_[shapeIdx++] = dy_;//5
      //  dy>0 -> move outwards
      if (i % 2 == 1) {
        y_ = -dy_;
      } else {
        y_ = dy_;
      }

      x_ = rn_.sampleUniform() * 0.01;
      z_ = rn_.sampleUniform() * 0.01;

      shapes_[shapeIdx++] = x_;//6
      shapes_[shapeIdx++] = z_;//7

      anymal_->getJointPos_P()[shankIdx].v[0] = defaultJointPositions_[shankIdx][0] + x_;
      anymal_->getJointPos_P()[shankIdx].v[1] = defaultJointPositions_[shankIdx][1] + y_;
      anymal_->getJointPos_P()[shankIdx].v[2] = defaultJointPositions_[shankIdx][2] + z_;

    }

    noisifyMass();

  }

  void noisifyMass() {
    anymal_->getMass()[0] = defaultBodyMasses_[0] + rn_.sampleUniform01() * 2;

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
      anymal_->getLinkCOM()[0].v[i] = COMPosition_[i] + rn_.sampleUniform() * 0.01;
    }
    anymal_->updateMassInfo();
  }

  void noisifyTerrain() {
    double newcf = 0.95 + 0.2 * rn_.sampleUniform();
    materials_.getMaterialPairProp(terrainKey_, robotKey_).c_f = newcf;
    newcf = 0.95 + 0.2 * rn_.sampleUniform();
    materials_.getMaterialPairProp(robotKey_, robotKey_).c_f = newcf;

    env_->updateMaterialProp(materials_);
  }

  void renewTerrain(double height = 0.05){
//    /// terrain
//    int ySamples = 100;
//    int xSize = 10;
//    int ySize = 10;
//    int stride = 100;
//
//    int loop = 1;
//
//    std::vector<double> heights;
//    for (int i = 0; i < ySamples; i++) {
//      for (int j = 0; j < loop; j++) {
//        for (int k = 0; k < stride; k++) {
//          double depth;
//          depth = height* rn_.sampleUniform01();
//          if (j % 2 == 1) depth = 0;
//          heights.push_back(depth);
//        }
//      }
//    }
//    int xSamples = loop * stride;
//
//    heights.resize(xSamples * ySamples, 0);
//
//    env_->removeObject(terrain_);
//    terrain_ = env_->addHeightMap(xSamples, ySamples, xSize, ySize, 0.0, 0.0, heights);

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
    numBodyContact_ = 0;
    numBaseContact_ = 0;
    Vector3d command_temp;

    double mag = rn_.sampleUniform01();
    command_temp << 1.0 * rn_.sampleUniform(), 0.4 * rn_.sampleUniform(), 1.2 * rn_.sampleUniform();
//    command_temp << 1.0 * rn_.sampleUniform(),0.0, 0.0;
    setCommand(command_temp);

    for (int i = 0; i < 4; i++) footContactState_[i] = false;

//    if (noisify_) {
//      noisifyTerrain();
//      noisifyMass();
//    }
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

    q_ = q0;

    for (int i = 4; i < 19; i++) {
      q_(i) += q_initialNoiseScale(i) * rn_.sampleNormal() * noiseFtr_ * costScale_; // sample uniform
    }

    q_[0] = 2.0*rn_.sampleUniform();
    q_[1] = 2.0*rn_.sampleUniform();

    Eigen::Vector3d heading;
    heading(0) = 1.0 - 2.0 * rn_.intRand(0, 1);
    heading(1) = 0.2 * rn_.sampleUniform();
    heading(2) = 0.2 * rn_.sampleUniform();
    heading.normalize();

    angle = 0.1 * rn_.sampleUniform01() * noiseFtr_  * costScale_;

    double sin = std::sin(angle / 2.0);
    q_.template segment<3>(4) = heading * sin;
    q_(3) = std::cos(angle / 2.0);

    for (int i = 0; i < 18; i++) {
      u_(i) = u_initialNoiseScale(i) * rn_.sampleUniform() * noiseFtr_ * costScale_; // sample uniform
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

    double torqueCost, linvelCost, angVelCost, velLimitCost = 0, footClearanceCost = 0, slipCost = 0, desiredHeightCost, previousActionCost = 0, orientationCost = 0, footVelCost = 0;
    double yawRateError = (u_(5) - command_[2]) * (u_(5) - command_[2]) * (4.0 + costScale_*5);
    const double commandNorm = command_.norm();

    torqueCost = costScale_ *  0.05 * tau_.tail<12>().norm() * simulation_dt_;
    Eigen::Vector3d linearSpeed = (R_b_.transpose() * u_.segment<3>(0)), desiredLinearSpeed, linSpeedCostScale;

    desiredLinearSpeed << command_[0], command_[1], 0;
    linSpeedCostScale <<1.0, 1.0, 0.35;
    const double velErr = std::max((4.0 + costScale_*5) * (desiredLinearSpeed - linearSpeed).cwiseProduct(linSpeedCostScale).norm(),0.0);
    
    linvelCost = -10.0 * simulation_dt_ / (exp(velErr) + 2.0 + exp(-velErr));

    angVelCost = -6.0 * simulation_dt_ / (exp(yawRateError) + 2.0 + exp(-yawRateError));
    angVelCost += costScale_ * std::min(0.25 * u_.segment<2>(3).squaredNorm() * simulation_dt_, 0.002) / std::min(0.3 + 3.0 * commandNorm, 1.0);

    double velLim = 0.0;
    for (int i = 6; i < 18; i++)
      if (fabs(u_(i)) > velLim) velLimitCost += costScale2_ * 0.3e-2 / std::min(0.09 + 2.5 * commandNorm, 1.0) * (std::fabs(u_(i)) - velLim) * (std::fabs(u_(i)) - velLim) * simulation_dt_;

    for (int i = 6; i < 18; i++)
      if (fabs(u_(i)) > velLim) velLimitCost += costScale2_ * 0.2e-2 / std::min(0.09 + 2.5 * commandNorm, 1.0) * fabs(u_(i)) * simulation_dt_;


    for (int i = 0; i < 4; i++)
      footVelCost += costScale2_ * 1e-1 / std::min(0.25 + 3.0 * commandNorm, 1.0) * footVel_W[i][2] * footVel_W[i][2] * simulation_dt_;


    for (int i = 0; i < 4; i++) {
      if (!footContactState_[i])
        footClearanceCost += costScale_ * 15.0 * pow(std::max(0.0, 0.07 - footPos_W[i][2]), 2) * footVel_W[i].e().head(2).norm() * simulation_dt_;
      else
        slipCost += (costScale_ * (2.0 * footContactVel_[i].head(2).norm())) * simulation_dt_;
    }

    previousActionCost = 0.5 * costScale_ * (previousAction_ - targetPosition_.template cast<Dtype>()).norm() * simulation_dt_;

    Vector3d identityRot(0,0,1);
    orientationCost = costScale_ * 0.4 * (R_b_.row(2).transpose()-identityRot).norm() * simulation_dt_;

    cost = torqueCost + linvelCost + angVelCost + footClearanceCost + velLimitCost + slipCost + previousActionCost + orientationCost + footVelCost;//  ;
//  cost += 0.005 * numBodyContact_ * simulation_dt_;

    if (isnan(cost)) {
      std::cout << "error in cost function!! " << std::endl;
      std::cout << "torqueCost " << torqueCost << std::endl;
      std::cout << "linvelCost " << linvelCost << std::endl;
      std::cout << "angVel " << angVelCost << std::endl;
      std::cout << "velLimitCost " << velLimitCost << std::endl;
      std::cout << "q_ " << q_.transpose() << std::endl;
      std::cout << "u_ " << u_.transpose() << std::endl;
      std::cout << "tau_ " << tau_.transpose() << std::endl;
      exit(0);
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
    
//    if(numBaseContact_ > 0 || numBodyContact_ > 8) {
//      return true;
//    }

    /// hip abduction
    for (int legId = 0; legId < 4; legId++)
      if (q_term(7 + legId * 3) > 1.2 || q_term(7 + legId * 3) < -1.2)
        return true;
    /// hip flexion
    for (int legId = 0; legId < 4; legId++)
      if (q_term(8 + legId * 3) > 1.5 || q_term(8 + legId * 3) < -1.5)
        return true;

    /// body height limit
    if (q_term(2) < 0.15 || q_term(2) > 1.2)
      return true;

    ///simulation diverging
    if (badlyConditioned_) {
      RAIWARN("BAAAAAAAD");
      badlyConditioned_ = false;
      return true;
    }

    if (q_term.maxCoeff() > 1e3 || q_term.minCoeff() < -1e3) {
      RAIWARN("BAAAAAAAD" << omp_get_thread_num());
      badlyConditioned_ = false;
      return true;
    }


    for(size_t i = 0; i < 12; i++){
      if(u_term[6 + i]  > 150.0 || u_term[6 + i] < -150.0){
        return true;
      }
    }

    for(size_t i = 3; i<6; i++){
      if(u_term[i]  > 40.0 || u_term[6 + i] < -40.0){
        return true;
      }
    }

    return false;
  }

  /**
*       learning state =
       *      [command (horizontal velocity, yawrate)                      n =  3, si =   0
       *       height                                                      n =  1  si =   3
       *       z-axis in world frame expressed in body frame (R_b.row(2)), n =  3, si =   4
       *       body Linear velocities,                                     n =  3, si =   7
       *       body Angular velocities,                                    n =  3, si =  10
       *       joint position history(t0, t-2, t-4),                       n = 36, si =  13
       *       joint velocities(t0, t-2, t-4)                              n = 36, si =  49
       *       previous action                                             n = 12, si =  85
       *       ]
   */

  //@warning this mapping cannot fully reconstruct state(h missing)
  inline void conversion_LearningState2GeneralizedState(const State &state,
                                                        VectorXd &q,
                                                        VectorXd &u) {
    // inverse scaling
    State state_unscaled = state.cwiseQuotient(stateScale_) + stateOffset_;

    Vector3d xaxis, yaxis, zaxis;
    zaxis << state_unscaled(4), state_unscaled(5), state_unscaled(6
    );
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

    LinearVelocity bodyVel = R_b * state_unscaled.template segment<3>(7).
        template cast<double>();
    AngularVelocity bodyAngVel = R_b * state_unscaled.template segment<3>(10).
        template cast<double>();
    VectorXd jointVel = state_unscaled.template segment<12>(49).
        template cast<double>();

    q << 0.0, 0.0, state_unscaled[3], quat, state_unscaled.template segment<12>(13).
        template cast<double>();
    u << bodyVel.template cast<double>(), bodyAngVel.template cast<double>(), jointVel.template cast<double>();

    command_ = state_unscaled.head(3).template cast<double>();

    jointPosHist_.template segment<12>((12 - 1)* HistoryLength) = state_unscaled.template segment<12>(13);
    jointPosHist_.template segment<12>((12 - 5)* HistoryLength) = state_unscaled.template segment<12>(25);
    jointPosHist_.template segment<12>((12 - 9)* HistoryLength) = state_unscaled.template segment<12>(37);

    jointVelHist_.template segment<12>((12 - 1)* HistoryLength) = state_unscaled.template segment<12>(49);
    jointVelHist_.template segment<12>((12 - 5) * HistoryLength) = state_unscaled.template segment<12>(61);
    jointVelHist_.template segment<12>((12 - 9) * HistoryLength) = state_unscaled.template segment<12>(73);

    torqueHist_.setZero();

    previousAction_ = state_unscaled.template segment<12>(85);
  }
//  *		19 dimension, q = [body_position (3 numbers IDX 0-2)
//  *	 	 	 	 	 	   body_quaternion   (4 numbers IDX 3-6),
//  * 	 	 	 	 	 	   leg1- HAA, HFE, KFE (3 numbers, IDX 7-9)
//  *	 	 	 	 	 	   leg2- ... leg3- ... leg4- ...]
  /**
*       learning state =
       *      [command (horizontal velocity, yawrate)                      n =  3, si =   0
       *       height                                                      n =  1  si =   3
       *       z-axis in world frame expressed in body frame (R_b.row(2)), n =  3, si =   4
       *       body Linear velocities,                                     n =  3, si =   7
       *       body Angular velocities,                                    n =  3, si =  10
       *       joint position history(t0, t-2, t-4),                       n = 36, si =  13
       *       joint velocities(t0, t-2, t-4)                              n = 36, si =  49
       *       previous action                                             n = 12, si =  85
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
    state_unscaled[3] = q[2];

    state_unscaled.template segment<3>(4) = R_b.row(2).transpose().template cast<Dtype>();
//    std::cout <<  R_b.row(2) << std::endl;
    /// velocity in body coordinate
    LinearVelocity bodyVel = R_b.transpose() * u.template segment<3>(0);
    AngularVelocity bodyAngVel = R_b.transpose() * u.template segment<3>(3);
    VectorXd jointVel = u.template segment<12>(6);

    state_unscaled.template segment<3>(7) = bodyVel.template cast<Dtype>();
    state_unscaled.template segment<3>(10) = bodyAngVel.template cast<Dtype>();

    state_unscaled.template segment<12>(13) = q.template segment<12>(7).
        template cast<Dtype>(); /// position
    state_unscaled.template segment<12>(49) = jointVel.template cast<Dtype>();

    state_unscaled.template segment<12>(25) = jointPosHist_.template segment<12>((12 - 5)* HistoryLength);
    state_unscaled.template segment<12>(37) = jointPosHist_.template segment<12>((12 - 9)* HistoryLength);

    state_unscaled.template segment<12>(61) = jointVelHist_.template segment<12>((12 - 5)* HistoryLength);
    state_unscaled.template segment<12>(73) = jointVelHist_.template segment<12>((12 - 9)* HistoryLength);

    state_unscaled.template segment<12>(85) = previousAction_;

//    std::cout << previousAction_.transpose() << std::endl;
    // scaling
    state = (state_unscaled - stateOffset_).cwiseProduct(stateScale_);
  }

/// task spec
 public:
  double initAngle_;
  int instance_;

  Eigen::VectorXd shapes_;
  bool changeShape_;
  bool noisify_;

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
//  rai_sim::HeightMapHandle terrain_;

  std::unique_ptr<rai_sim::World_RG> env_;
  rai_sim::MaterialManager materials_;
  int terrainKey_, robotKey_;
  Eigen::Vector3d gravity_;

  rai::FuncApprox::MLP_fullyconnected<double, 6, 1, rai::FuncApprox::ActivationType::softsign> actuator_;

  Eigen::VectorXd u_, u_initialNoiseScale, u0;
  Eigen::VectorXd q_, q_initialNoiseScale, q0;
  Eigen::VectorXd q0s, shape0s_;
  Eigen::VectorXd u0PreviousRandom_, q0PreviousRandom_;
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

  size_t numContact_;
  size_t numFootContact_;
  size_t numBodyContact_;
  size_t numBaseContact_;

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

  double noiseFtr_ = 0.4;

  rai::RandomNumberGenerator<Dtype> rn_;
};

}
} //namespaces