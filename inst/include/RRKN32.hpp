#ifndef _RRKN32_HPP_
#define _RRKN32_HPP_




template <class _ode_type_>
class RRKN32{

  _ode_type_* ode_;

  size_t dim_;
  size_t dimGenerated_;
  size_t dimEvent_;

  Eigen::MatrixXd force_;
  Eigen::MatrixXd ys_;
  Eigen::MatrixXd generated_;
  Eigen::MatrixXd events_;
  Eigen::MatrixXd diag_;

  Eigen::VectorXd force_tmp_;
  Eigen::VectorXd y_tmp_;
  Eigen::VectorXd gen_tmp_;
  Eigen::VectorXd diag_tmp_;

  Eigen::VectorXd q_low_;
  Eigen::VectorXd v_low_;

  Eigen::VectorXd tmpVec_,event_tmp_;
  Eigen::VectorXd eventRootInt_,eventRootIntR_;
  Eigen::VectorXd eventA_,eventB_;

  Eigen::VectorXd U_;
  odeState tmpState_,newState_;

  secondOrderRep sor_;

  inline Eigen::VectorXd calcIntPoly(const double t) const {
    double ts = t/eps_;
    double tsSq = ts*ts;
    double tsQ = ts*tsSq;
    Eigen::VectorXd intPoly(4);
    intPoly.coeffRef(0) = 1.0 + 2.0*tsQ - 3.0*tsSq;
    intPoly.coeffRef(1) = 3.0*tsSq - 2.0*tsQ;
    intPoly.coeffRef(2) = eps_*(tsQ - 2.0*tsSq + ts);
    intPoly.coeffRef(3) = eps_*(tsQ - tsSq);
    return(intPoly);
  }

  inline Eigen::VectorXd calcLevelPoly(const double t) const {
    double ts = t/eps_;
    double tsSq = ts*ts;
    Eigen::VectorXd levelPoly(4);
    levelPoly.coeffRef(0) = 6.0*(tsSq-ts)/eps_;
    levelPoly.coeffRef(1) = 6.0*(ts-tsSq)/eps_;
    levelPoly.coeffRef(2) = 1.0 - 4.0*ts + 3.0*tsSq;
    levelPoly.coeffRef(3) = 3.0*tsSq - 2.0*ts;
    return(levelPoly);
  }


  inline rootInfo nonlinRootSolver(){
    int whichDim = -1;
    double ret = eps_;

    if(dimEvent_>0){
      eventA_ = 3.0*(events_.col(0)+events_.col(2)) - 6.0*eventRootIntR_;
      eventB_ = 6.0*eventRootIntR_ - 4.0*events_.col(0) - 2.0*events_.col(2);
      double q,eventDev,r;
      for(size_t i=0;i<dimEvent_;i++){
        if(fabs(eventA_.coeff(i))<_EVENTROOTSOLVER_TOL_){
          // poly is at most linear
          if(fabs(eventB_.coeff(i))>_EVENTROOTSOLVER_TOL_){
            // poly is linear, constant case gets ignored
            r = -eps_*events_.coeff(i,0)/eventB_.coeff(i);
            if(r>0.0 && r<ret){
              ret = r;
              whichDim = i;
              //std::cout << "linear Root, t = " << t_left_ + r << std::endl;
            }
          }
        } else {
          // poly is quadratic
          eventDev = pow(eventB_.coeff(i),2) - 4.0*eventA_.coeff(i)*events_.coeff(i,0);
          if(eventDev>=0.0){
            q = (eventB_.coeff(i)>=0.0) ? -0.5*(eventB_.coeff(i) + sqrt(eventDev)) : -0.5*(eventB_.coeff(i)-sqrt(eventDev));
            r = eps_*(q/eventA_.coeff(i));
            if(0.0<r && r<ret){
              ret = r;
              whichDim = i;
            }
            r = eps_*(events_.coeff(i,0)/q);
            if(0.0<r && r<ret){
              ret = r;
              whichDim = i;
            }
          }
        }
      }
    }
    return(rootInfo(ret,0,whichDim));
  }


public:

  double absTol_;
  double relTol_;
  double eps_; // integrator step size
  double stepErr_q_;
  double stepErr_v_;
  double t_left_,t_right_; // time on adaptive mesh

  Eigen::VectorXd genIntStep_;
  Eigen::VectorXd diagInt_;
  RRKN32() : absTol_(1.0e-3), relTol_(1.0e-3), eps_(0.5) {}
  inline odeState firstState() const {return(odeState(ys_.col(0)));}
  inline double firstState(const size_t dimension){return ys_.coeff(dimension,0);}
  inline odeState lastState() const {return(odeState(ys_.col(2)));}
  inline Eigen::VectorXd firstGenerated(){return generated_.col(0);}
  inline Eigen::VectorXd generateU(rng& r) {
    Eigen::VectorXd ret(1);
    ret(0) = std::sqrt(r.runif());
    return(ret);
  }
  inline int odeOrder() const {return 1;} // note: step appears to integrator as first order ODE step
  void dumpYs(){std::cout << "ys : \n" << ys_ << std::endl;}
  void dumpStep(){
    std::cout << "dump of RKBS32 step" << std::endl;
    dumpYs();
    std::cout << "forces : \n" << force_ << std::endl;
    if(dimGenerated_>0){
      std::cout << "generated, dimGenerated = " << dimGenerated_ << std::endl << generated_ << std::endl;
    }
  }

  inline void setup(_ode_type_ &ode){
    ode_ = &ode;
    dim_ = (*ode_).dim();
    dimGenerated_ = (*ode_).generatedDim();
    dimEvent_ = (*ode_).eventRootDim();

    sor_ = (*ode_).secondOrderRepresentation();
    sor_.dump();

    force_.resize(dim_,3);
    force_tmp_.resize(dim_);
    ys_.resize(dim_,3);
    y_tmp_.resize(dim_);

    if(dimGenerated_>0){
      generated_.resize(dimGenerated_,3);
      gen_tmp_.resize(dimGenerated_);
      genIntStep_.resize(dimGenerated_);
    }

    events_.resize(dimEvent_,3);
    events_.setZero();

  }

  bool setInitialState(const odeState &y0){
    if(y0.y.size() != dim_){
      std::cout << "RKBS32::setInitialState : dimension mismatch" << std::endl;
      return(false);
    }

    t_left_ = 0.0;
    ys_.col(0) = y0.y;

    // first evaluation
    (*ode_).ode(t_left_,
     ys_.col(0),force_tmp_,gen_tmp_,diag_tmp_);

    //    std::cout << "eval done " << dimEvent_ << std::endl;

    force_.col(0) = force_tmp_;
    if(dimGenerated_>0) generated_.col(0) = gen_tmp_;
    if(diag_tmp_.size()!= diag_.rows()) diag_.resize(diag_tmp_.size(),4);
    if(diag_tmp_.size()>0) diag_.col(0) = diag_tmp_;

    events_.col(0) = (*ode_).eventRoot(0.0,odeState(ys_.col(0)),force_.col(0),true);

    // check second order property : \dot{position} should be equal to velocity

    Eigen::VectorXd testDev = ys_.col(0).segment(sor_.vel_first_,sor_.vel_last_-sor_.vel_first_+1)
      - force_.col(0).segment(sor_.pos_first_,sor_.pos_last_-sor_.pos_first_+1);
    if(testDev.cwiseAbs().maxCoeff()>1.0e-14){
      std::cout << "The ODE system does not have a second order structure" << std::endl;
      throw 1;
    } //else {
      //std::cout << "Second order structure of ODE system verified" << std::endl;
    //}



    return(force_.col(0).array().isFinite().all());
  }


  void newEpsAfterReject() {
    eps_ *= 0.9*std::min(std::pow(stepErr_q_,-1.0/3.0),std::pow(stepErr_v_,-0.5));
  }
  void newEpsAfterAccept(const Eigen::VectorXd& nextU) {
    double ufacq = std::abs((nextU(0)-0.5)/(U_(0)-0.5));
    double ufacv = std::abs((nextU(0)-2.0/3.0)/(U_(0)-2.0/3.0));
    double fac = std::fmin(pow(ufacq*stepErr_q_,-1.0/3.0),pow(ufacv*stepErr_v_,-0.5));
    eps_ *= 0.95*std::fmin(2.0,std::fmax(0.2,fac));
  }

  bool step(const Eigen::VectorXd& U){

    U_ = U; // store for potential future use
    double uu = U(0);
    double epssq = pow(eps_,2);
    //std::cout << "step, U: " << U << " eps : " << eps_ << std::endl;
    // new q
    ys_.col(1).segment(sor_.pos_first_,sor_.pos_last_-sor_.pos_first_+1) = // q
      ys_.col(0).segment(sor_.pos_first_,sor_.pos_last_-sor_.pos_first_+1) + // q0
      (eps_*uu)*(ys_.col(0).segment(sor_.vel_first_,sor_.vel_last_-sor_.vel_first_+1)) + // h*u*v0
      0.5*pow(eps_*uu,2)*(force_.col(0).segment(sor_.vel_first_,sor_.vel_last_-sor_.vel_first_+1)); // 0.5*(h*u)^2*F(q0)
    // new v
    ys_.col(1).segment(sor_.vel_first_,sor_.vel_last_-sor_.vel_first_+1) = // v
      ys_.col(0).segment(sor_.vel_first_,sor_.vel_last_-sor_.vel_first_+1) + // v0
      (eps_*uu)*(force_.col(0).segment(sor_.vel_first_,sor_.vel_last_-sor_.vel_first_+1)); // h*u*F(q0)
    // new r
    ys_.col(1).segment(sor_.int_first_,sor_.int_last_-sor_.int_first_+1) = // r
      ys_.col(0).segment(sor_.int_first_,sor_.int_last_-sor_.int_first_+1) + // r0
      (eps_*uu)*(force_.col(0).segment(sor_.int_first_,sor_.int_last_-sor_.int_first_+1)); // h*u*M(q0,v0,r0)


    (*ode_).ode(t_left_+uu*eps_,
     ys_.col(1),force_tmp_,gen_tmp_,diag_tmp_);
    force_.col(1) = force_tmp_;
    if(dimGenerated_>0) generated_.col(1) = gen_tmp_;
    if(diag_tmp_.size()>0) diag_.col(1) = diag_tmp_;

    tmpState_.y = ys_.col(1);
    events_.col(1) = (*ode_).eventRoot(t_left_+uu*eps_,
                tmpState_,force_.col(1),true);

    if(! force_.col(1).array().isFinite().all()) return(false);

    // q
    ys_.col(2).segment(sor_.pos_first_,sor_.pos_last_-sor_.pos_first_+1) = // q
      ys_.col(0).segment(sor_.pos_first_,sor_.pos_last_-sor_.pos_first_+1) + // q0
      (eps_)*(ys_.col(0).segment(sor_.vel_first_,sor_.vel_last_-sor_.vel_first_+1)) + // h*v0
      (epssq*(0.5-1.0/(6.0*uu)))*(force_.col(0).segment(sor_.vel_first_,sor_.vel_last_-sor_.vel_first_+1)) +
      (epssq/(6.0*uu))*(force_.col(1).segment(sor_.vel_first_,sor_.vel_last_-sor_.vel_first_+1));
    // v
    ys_.col(2).segment(sor_.vel_first_,sor_.vel_last_-sor_.vel_first_+1) = // v
      ys_.col(0).segment(sor_.vel_first_,sor_.vel_last_-sor_.vel_first_+1) + // v0
      (eps_*(1.0-1.0/(2.0*uu)))*(force_.col(0).segment(sor_.vel_first_,sor_.vel_last_-sor_.vel_first_+1)) +
      (eps_/(2.0*uu))*(force_.col(1).segment(sor_.vel_first_,sor_.vel_last_-sor_.vel_first_+1));
    // r
    ys_.col(2).segment(sor_.int_first_,sor_.int_last_-sor_.int_first_+1) = // r
      ys_.col(0).segment(sor_.int_first_,sor_.int_last_-sor_.int_first_+1) + // r0
      (eps_*(1.0-1.0/(2.0*uu)))*force_.col(0).segment(sor_.int_first_,sor_.int_last_-sor_.int_first_+1) +
      (eps_/(2.0*uu))*(force_.col(1).segment(sor_.int_first_,sor_.int_last_-sor_.int_first_+1));


    (*ode_).ode(t_left_ + eps_,
     ys_.col(2),force_tmp_,gen_tmp_,diag_tmp_);
    force_.col(2) = force_tmp_;
    if(dimGenerated_>0) generated_.col(2) = gen_tmp_;
    if(diag_tmp_.size()>0) diag_.col(2) = diag_tmp_;
    tmpState_.y = ys_.col(2);
    events_.col(2) = (*ode_).eventRoot(t_left_ + eps_,
                tmpState_,force_.col(2),true);
    if(! force_.col(2).array().isFinite().all()) return(false);

    // low order position
    q_low_ = ys_.col(0).segment(sor_.pos_first_,sor_.pos_last_-sor_.pos_first_+1) + // q0
      (eps_)*(ys_.col(0).segment(sor_.vel_first_,sor_.vel_last_-sor_.vel_first_+1)) + // h*v0
      (epssq*0.5)*(force_.col(0).segment(sor_.vel_first_,sor_.vel_last_-sor_.vel_first_+1)) +
      (-epssq/3.0)*(force_.col(1).segment(sor_.vel_first_,sor_.vel_last_-sor_.vel_first_+1)) +
      (epssq/3.0)*(force_.col(2).segment(sor_.vel_first_,sor_.vel_last_-sor_.vel_first_+1));
    // low order velocity
    v_low_ = ys_.col(0).segment(sor_.vel_first_,sor_.vel_last_-sor_.vel_first_+1) + // v0
      (eps_/3.0)*(force_.col(0).segment(sor_.vel_first_,sor_.vel_last_-sor_.vel_first_+1)) +
      (eps_*0.5)*(force_.col(1).segment(sor_.vel_first_,sor_.vel_last_-sor_.vel_first_+1)) +
      (eps_/6.0)*(force_.col(2).segment(sor_.vel_first_,sor_.vel_last_-sor_.vel_first_+1));

    t_right_ = t_left_+eps_;

    // step error
    tmpVec_ = (absTol_ + relTol_*ys_.col(0).array().abs().max(ys_.col(2).array().abs())).array();

    stepErr_q_ = 0.0;
    for(std::size_t i = sor_.pos_first_; i<= sor_.pos_last_; i++){
      stepErr_q_ = std::max(stepErr_q_,
                            std::abs(q_low_.coeff(i-sor_.pos_first_)-ys_.col(2).coeff(i))/tmpVec_.coeff(i));
    }
    stepErr_v_ = 0.0;
    for(std::size_t i = sor_.vel_first_; i<= sor_.vel_last_; i++){
      stepErr_v_ = std::max(stepErr_v_,
                            std::abs(v_low_.coeff(i-sor_.vel_first_)-ys_.col(2).coeff(i))/tmpVec_.coeff(i));
    }

    //std::cout << "stepErr_q_: " << stepErr_q_ << std::endl;
    //std::cout << "stepErr_v_: " << stepErr_v_ << std::endl;
    // integrated generated quantities
    if(dimGenerated_>0){
      genIntStep_ = (eps_*(1.0-1.0/(2.0*uu)))*generated_.col(0) +
        (eps_/(2.0*uu))*generated_.col(1);

    }


    if(diag_.rows()>0){
      diagInt_ = (eps_*(1.0-1.0/(2.0*uu)))*diag_.col(0) +
        (eps_/(2.0*uu))*diag_.col(1);
    } else {
      if(diagInt_.size()>0) diagInt_.resize(0);
    }

    if(dimEvent_>0){
      eventRootIntR_ = (1.0-1.0/(2.0*uu))*events_.col(0) +
        (1.0/(2.0*uu))*events_.col(1);


      eventRootInt_ = eps_*eventRootIntR_;

    }
    return(true);
  }



  inline rootInfo eventRootSolver(const rootInfo& oldRoot){
    return(nonlinRootSolver());
  }



  void prepareNext(){
    ys_.col(0) = ys_.col(2);
    force_.col(0) = force_.col(2);
    if(dimGenerated_>0) generated_.col(0) = generated_.col(2);
    if(diag_.rows()>0) diag_.col(0) = diag_.col(2);
    events_.col(0) = events_.col(2);
    t_left_ = t_right_;
  }


  inline double denseState(const int which,
                           const double t) const {
    Eigen::VectorXd intPoly = calcIntPoly(t);
    return(ys_.coeff(which,0)*intPoly.coeff(0) +
           ys_.coeff(which,2)*intPoly.coeff(1) +
           force_.coeff(which,0)*intPoly.coeff(2) +
           force_.coeff(which,2)*intPoly.coeff(3));
  }

  Eigen::VectorXd denseState(const double t) const {
    Eigen::VectorXd intPoly = calcIntPoly(t);
    return(ys_.col(0)*intPoly.coeff(0) +
           ys_.col(2)*intPoly.coeff(1) +
           force_.col(0)*intPoly.coeff(2) +
           force_.col(2)*intPoly.coeff(3));
  }

  inline void denseState(const Eigen::VectorXi &which,
                         const double t,
                         Eigen::Ref<Eigen::VectorXd> out) const {
    if(which.size()!=out.size()){
      std::cout << "Error in RKBS32::denseState" << std::endl;
      throw(1);
    }
    Eigen::VectorXd intPoly = calcIntPoly(t);
    for(size_t i=0;i<which.size();i++){
      out.coeffRef(i) = ys_.coeff(which.coeff(i),0)*intPoly.coeff(0) +
        ys_.coeff(which.coeff(i),2)*intPoly.coeff(1) +
        force_.coeff(which.coeff(i),0)*intPoly.coeff(2) +
        force_.coeff(which.coeff(i),2)*intPoly.coeff(3);
    }
  }


  inline void denseGenerated_Int(const double t,
                                 Eigen::Ref<Eigen::VectorXd> out) const {
    Eigen::VectorXd intPoly = calcIntPoly(t);
    out = genIntStep_*intPoly.coeff(1) +
      generated_.col(0)*intPoly.coeff(2) +
      generated_.col(2)*intPoly.coeff(3);
  }

  inline Eigen::VectorXd denseGenerated_Int(const double t) const {
    Eigen::VectorXd intPoly = calcIntPoly(t);
    return(genIntStep_*intPoly.coeff(1) +
           generated_.col(0)*intPoly.coeff(2) +
           generated_.col(2)*intPoly.coeff(3));
  }

  inline void denseGenerated_Level(const double t,
                                   Eigen::Ref<Eigen::VectorXd> out) const {
    Eigen::VectorXd levelPoly = calcLevelPoly(t);
    out = genIntStep_*levelPoly.coeff(1)+
      generated_.col(0)*levelPoly.coeff(2) +
      generated_.col(2)*levelPoly.coeff(3);
  }

  inline bool event(const rootInfo& rootOut){
    int whichEvent = rootOut.rootDim_;
    double eventTime = rootOut.rootTime_;



    //std::cout << "RKBS32::event : ode time : " << t_left_ + rootOut.rootTime_ << " \n" << rootOut << std::endl;

    // dense state before event
    Eigen::VectorXd intPoly = calcIntPoly(eventTime);
    tmpState_.y = ys_.col(0)*intPoly.coeff(0) +
      ys_.col(2)*intPoly.coeff(1) +
      force_.col(0)*intPoly.coeff(2) +
      force_.col(2)*intPoly.coeff(3);

    Eigen::VectorXd levelPoly = calcLevelPoly(eventTime);
    force_tmp_ = ys_.col(0)*levelPoly.coeff(0) +
      ys_.col(2)*levelPoly.coeff(1) +
      force_.col(0)*levelPoly.coeff(2) +
      force_.col(2)*levelPoly.coeff(3);

    // evaluate eventRoot before event is done
    if(rootOut.rootType_==0){
      event_tmp_ = (*ode_).eventRoot(
        t_left_+eventTime,
        tmpState_,
        force_tmp_,false);
      if(std::fabs(event_tmp_(whichEvent))>200.0*absTol_){
        std::cout << "eventRoot at interpolated state: " << event_tmp_(whichEvent) << std::endl;
        std::cout << "whichEvent : " << whichEvent << std::endl;
      }
    }
    // evaluate the new state after event occurred
    bool eventContinue = (*ode_).event(
      rootOut, // which event
      t_left_+eventTime, // time of event
      tmpState_, // state at event
      force_tmp_, // force at event
      newState_);

    // evaluate ode and root fun after event occurred to prepare for subsequent
    // step
    ys_.col(2) = newState_.y;
    t_right_ = t_left_+eventTime;
    (*ode_).ode(t_right_,
     ys_.col(2),force_tmp_,gen_tmp_,diag_tmp_);
    force_.col(2) = force_tmp_;

    if(dimGenerated_>0) generated_.col(2) = gen_tmp_;
    if(diag_.rows() != diag_tmp_.size()) diag_.resize(diag_tmp_.size(),3);
    if(diag_tmp_.size()>0) diag_.col(2) = diag_tmp_;
    events_.col(2) = (*ode_).eventRoot(
      t_right_,
      newState_,
      force_.col(2),true);

    // set the active eventRoot at event artificially to exactly zero
    // if the event root equation did not change
    // to avoid repeating the event due to numerical inaccuracies
    //std::cout << "rk::event new root eval  : \n" << event_tmp_ << std::endl;
    //std::cout << "rk::event new root eval  : \n" << events_ << std::endl;
    //std::cout << rootOut << std::endl;
    if(rootOut.rootType_==0 && std::fabs(event_tmp_(whichEvent)-events_(whichEvent,2))<1.0e-14){
      events_(whichEvent,2) = 0.0;
    }

    if(! force_.col(2).array().isFinite().all()){
      eventContinue = false;
      std::cout << "Post event Numerical problems" << std::endl;
    }
    return(eventContinue);
  }


};


#endif
