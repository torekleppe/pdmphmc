#ifndef _RKDP54_HPP_
#define _RKDP54_HPP_

/*
 *
 * Runge Kutta based on order 5(4) Dormand/Prince pair
 * with 4-th order dense output
 *
 */

template <class _ode_type_>
class RKDP54{
  _ode_type_ *ode_;
  size_t dim_;
  size_t dimGenerated_;
  size_t dimEvent_;

  Eigen::VectorXd y1_low_;
  Eigen::VectorXd y_tmp_,y_tmp2_;
  Eigen::VectorXd force_tmp_;
  Eigen::VectorXd event_tmp_;
  Eigen::VectorXd gen_tmp_;
  Eigen::VectorXd diag_tmp_;
  Eigen::VectorXd tmpVec_;

  Eigen::VectorXd denseWts(const double a) const {
    Eigen::VectorXd ret(7);
    ret(0) = 0.1e1 + (-0.28605386690370886310e1 + (0.30995778787091209227e1 + (-0.11618105836403092858e1 + 0.13917207301610327394e-1 * a) * a) * a) * a;
    ret(1) = 0.0;
    ret(2) = (0.40471414996427855732e1 + (-0.63453540469389257351e1 + (0.27954650864140050830e1 - 0.48016240824962854620e-1 * a) * a) * a) * a;
    ret(3) = (-0.39411600711126589294e1 + (0.10904003027940294347e2 + (-0.67293175092092785730e1 + 0.41751621904830982182e0 * a) * a) * a) * a;
    ret(4) = (0.28419447015870346406e1 + (-0.75476758629593859984e1 + (0.49576367249312529806e1 - 0.57428174280418464170e0 * a) * a) * a) * a;
    ret(5) = (-0.16109886359167998170e1 + (0.42189158390395179940e1 + (-0.29501038655667317753e1 + 0.47312904339639455060e0 * a) * a) * a) * a;
    ret(6) = (0.15236011748367271635e1 + (-0.43294668357906215305e1 + (0.30881301470710615705e1 - 0.28226448611716720348e0 * a) * a) * a) * a;
    return(eps_*a*ret);
  }

  Eigen::VectorXd denseDotWts(const double a) const {
    Eigen::VectorXd ret(7);
    ret(0) = 0.1e1 + (-0.57210773380741772620e1 + (0.92987336361273627681e1 + (-0.46472423345612371431e1 + 0.69586036508051636969e-1 * a) * a) * a) * a;
    ret(1) = 0;
    ret(2) = (0.80942829992855711464e1 + (-0.19036062140816777205e2 + (0.11181860345656020332e2 - 0.24008120412481427310e0 * a) * a) * a) * a;
    ret(3) = (-0.78823201422253178588e1 + (0.32712009083820883042e2 + (-0.26917270036837114292e2 + 0.20875810952415491091e1 * a) * a) * a) * a;
    ret(4) = (0.56838894031740692812e1 + (-0.22643027588878157995e2 + (0.19830546899725011922e2 - 0.28714087140209232085e1 * a) * a) * a) * a;
    ret(5) = (-0.32219772718335996339e1 + (0.12656747517118553982e2 + (-0.11800415462266927101e2 + 0.23656452169819727530e1 * a) * a) * a) * a;
    ret(6) = (0.30472023496734543271e1 + (-0.12988400507371864592e2 + (0.12352520588284246282e2 - 0.14113224305858360174e1 * a) * a) * a) * a;
    return(ret);
  }

  Eigen::VectorXd denseDotDotWts(const double a) const {
    Eigen::VectorXd ret(7);
    ret(0) = -0.57210773380741772620e1 + (0.18597467272254725536e2 + (-0.13941727003683711429e2 + 0.27834414603220654788e0 * a) * a) * a;
    ret(1) = 0;
    ret(2) = 0.80942829992855711464e1 + (-0.38072124281633554410e2 + (0.33545581036968060996e2 - 0.96032481649925709241e0 * a) * a) * a;
    ret(3) = -0.78823201422253178588e1 + (0.65424018167641766083e2 + (-0.80751810110511342876e2 + 0.83503243809661964363e1 * a) * a) * a;
    ret(4) = 0.56838894031740692812e1 + (-0.45286055177756315990e2 + (0.59491640699175035767e2 - 0.11485634856083692834e2 * a) * a) * a;
    ret(5) = -0.32219772718335996339e1 + (0.25313495034237107964e2 + (-0.35401246386800781304e2 + 0.94625808679278910119e1 * a) * a) * a;
    ret(6) = 0.30472023496734543271e1 + (-0.25976801014743729183e2 + (0.37057561764852738846e2 - 0.56452897223433440696e1 * a) * a) * a;
    return((1.0/eps_)*ret);
  }

  Eigen::MatrixXd force_;
  Eigen::MatrixXd ys_;
  Eigen::MatrixXd events_;
  Eigen::MatrixXd generated_;
  Eigen::MatrixXd diag_;

  odeState tmpState_,newState_;

public:

  double absTol_;
  double relTol_;
  double eps_; // integrator step size
  double stepErr_;
  double t_left_,t_right_; // time on adaptive mesh



  Eigen::VectorXd genIntStep_;
  Eigen::VectorXd diagInt_;


  RKDP54() : absTol_(1.0e-4), relTol_(1.0e-4), eps_(1.0) {}
  inline int odeOrder() const {return 1;}
  constexpr bool hasEventRootSolver() const {return false;}
  inline double errorOrderHigh() const {return(5.0);}
  inline void setup(_ode_type_ &ode){
    ode_ = &ode;
    dim_ = (*ode_).dim();
    dimGenerated_ = (*ode_).generatedDim();
    dimEvent_ = (*ode_).eventRootDim();

    force_.resize(dim_,7);
    force_tmp_.resize(dim_);
    ys_.resize(dim_,7);
    y_tmp_.resize(dim_);

    if(dimGenerated_>0){
      generated_.resize(dimGenerated_,7);
      gen_tmp_.resize(dimGenerated_);
      genIntStep_.resize(dimGenerated_);
    }

    events_.resize(dimEvent_,7);
    events_.setZero();

  }



  inline odeState lastState() const {return(odeState(ys_.col(6)));}
  inline odeState firstState() const {return(odeState(ys_.col(0)));}
  inline double firstState(const size_t dimension){return ys_.coeff(dimension,0);}
  inline Eigen::VectorXd firstGenerated(){return generated_.col(0);}
  inline Eigen::VectorXd lastGenerated(){return generated_.col(6);}


  void dumpYs(){std::cout << "ys : \n" << ys_ << std::endl;}
  bool step(){

    ys_.col(1) = ys_.col(0) +
      (eps_*0.2)*force_.col(0);
    (*ode_).ode(t_left_+0.2*eps_,
     ys_.col(1),force_tmp_,gen_tmp_,diag_tmp_);
    force_.col(1) = force_tmp_;
    if(dimGenerated_>0) generated_.col(1) = gen_tmp_;
    if(diag_tmp_.size()>0) diag_.col(1) = diag_tmp_;

    tmpState_.y = ys_.col(1);
    events_.col(1) = (*ode_).eventRoot(t_left_+0.2*eps_,
                tmpState_,force_.col(1));

    if(! force_.col(1).array().isFinite().all()) return(false);

    ys_.col(2) = ys_.col(0) +
      (eps_*0.075)*force_.col(0) +
      (eps_*0.225)*force_.col(1);
    (*ode_).ode(t_left_ + 0.3*eps_,
     ys_.col(2),force_tmp_,gen_tmp_,diag_tmp_);
    force_.col(2) = force_tmp_;
    if(dimGenerated_>0) generated_.col(2) = gen_tmp_;
    if(diag_tmp_.size()>0) diag_.col(2) = diag_tmp_;
    tmpState_.y = ys_.col(2);
    events_.col(2) = (*ode_).eventRoot(t_left_ + 0.3*eps_,
                tmpState_,force_.col(2));


    if(! force_.col(2).array().isFinite().all()) return(false);

    ys_.col(3) = ys_.col(0) +
      (eps_*0.9777777777777777)*force_.col(0) -
      (eps_*3.7333333333333333)*force_.col(1) +
      (eps_*3.5555555555555553)*force_.col(2);
    (*ode_).ode(t_left_ + 0.8*eps_,
     ys_.col(3),force_tmp_,gen_tmp_,diag_tmp_);
    force_.col(3) = force_tmp_;
    if(dimGenerated_>0) generated_.col(3) = gen_tmp_;
    if(diag_tmp_.size()>0) diag_.col(3) = diag_tmp_;
    tmpState_.y = ys_.col(3);
    events_.col(3) = (*ode_).eventRoot(t_left_ + 0.8*eps_,
                tmpState_,force_.col(3));


    if(! force_.col(3).array().isFinite().all()){ return(false);}

    ys_.col(4) = ys_.col(0) +
      (eps_*2.952598689224203)*force_.col(0) -
      (eps_*11.59579332418839)*force_.col(1) +
      (eps_*9.822892851699436)*force_.col(2) -
      (eps_*0.2908093278463649)*force_.col(3);

    (*ode_).ode(t_left_ + (8.0/9.0)*eps_,
     ys_.col(4),force_tmp_,gen_tmp_,diag_tmp_);
    force_.col(4) = force_tmp_;
    if(dimGenerated_>0) generated_.col(4) = gen_tmp_;
    if(diag_tmp_.size()>0) diag_.col(4) = diag_tmp_;
    tmpState_.y = ys_.col(4);
    events_.col(4) = (*ode_).eventRoot(t_left_ + (8.0/9.0)*eps_,
                tmpState_,force_.col(4));

    if(! force_.col(4).array().isFinite().all()) return(false);

    ys_.col(5) = ys_.col(0) +
      (eps_*2.846275252525253)*force_.col(0) -
      (eps_*10.75757575757576)*force_.col(1) +
      (eps_*8.906422717743473)*force_.col(2) +
      (eps_*0.2784090909090909)*force_.col(3) -
      (eps_*0.2735313036020583)*force_.col(4);
    (*ode_).ode(t_left_ + eps_,
     ys_.col(5),force_tmp_,gen_tmp_,diag_tmp_);
    force_.col(5) = force_tmp_;
    if(dimGenerated_>0) generated_.col(5) = gen_tmp_;
    if(diag_tmp_.size()>0) diag_.col(5) = diag_tmp_;
    tmpState_.y = ys_.col(5);
    events_.col(5) = (*ode_).eventRoot(t_left_ + eps_,
                tmpState_,force_.col(5));

    if(! force_.col(5).array().isFinite().all()) return(false);


    // final (high order) position stored in ys_.col(6)
    ys_.col(6) = ys_.col(0) +
      (eps_*0.09114583333333333)*force_.col(0) +
      (eps_*0.4492362982929021)*force_.col(2) +
      (eps_*0.6510416666666666)*force_.col(3) -
      (eps_*0.322376179245283)*force_.col(4) +
      (eps_*0.130952380952381)*force_.col(5);
    (*ode_).ode(t_left_ + eps_,
     ys_.col(6),force_tmp_,gen_tmp_,diag_tmp_);
    force_.col(6) = force_tmp_;
    if(dimGenerated_>0) generated_.col(6) = gen_tmp_;
    if(diag_tmp_.size()>0) diag_.col(6) = diag_tmp_;
    tmpState_.y = ys_.col(6);
    events_.col(6) = (*ode_).eventRoot(t_left_ + eps_,
                tmpState_,force_.col(6));

    if(! force_.col(6).array().isFinite().all()) return(false);


    // low order position
    y1_low_ = ys_.col(0) +
      (eps_*0.08991319444444444)*force_.col(0) +
      (eps_*0.4534890685834082)*force_.col(2) +
      (eps_*0.6140625)*force_.col(3) -
      (eps_*0.2715123820754717)*force_.col(4) +
      (eps_*0.08904761904761904)*force_.col(5) +
      (eps_*0.025)*force_.col(6);

    t_right_ = t_left_+eps_;


    // step error
    tmpVec_ = (absTol_ + relTol_*ys_.col(0).array().abs().max(ys_.col(6).array().abs())).array();
    tmpVec_ = ((ys_.col(6)-y1_low_).array().abs()/tmpVec_.array());
    stepErr_ = tmpVec_.maxCoeff();


    // integrated generated quantities
    if(dimGenerated_>0){
      genIntStep_ = (eps_*0.09114583333333333)*generated_.col(0) +
        (eps_*0.4492362982929021)*generated_.col(2) +
        (eps_*0.6510416666666666)*generated_.col(3) -
        (eps_*0.322376179245283)*generated_.col(4) +
        (eps_*0.130952380952381)*generated_.col(5);
    }

    if(diag_.rows()>0){
      diagInt_ = (eps_*0.09114583333333333)*diag_.col(0) +
        (eps_*0.4492362982929021)*diag_.col(2) +
        (eps_*0.6510416666666666)*diag_.col(3) -
        (eps_*0.322376179245283)*diag_.col(4) +
        (eps_*0.130952380952381)*diag_.col(5);
    } else {
      if(diagInt_.size()>0) diagInt_.resize(0);
    }

    return(true);
  }


  inline bool event(const int whichEvent,
                    const double eventTime){
    // dense state before event
    tmpState_.y = ys_.col(0) + (force_*denseWts(eventTime/eps_));
    force_tmp_ = force_*denseDotWts(eventTime/eps_);

    // evaluate eventRoot before event is done
    event_tmp_ = (*ode_).eventRoot(
        t_left_+eventTime,
        tmpState_,
        force_tmp_);
    if(std::fabs(event_tmp_(whichEvent))>200.0*absTol_){
      std::cout << "eventRoot at interpolated state: " << event_tmp_(whichEvent) << std::endl;
      std::cout << "whichEvent : " << whichEvent << std::endl;
    }

    // evaluate the new state after event occurred
    bool eventContinue = (*ode_).event(
        whichEvent, // which event
        t_left_+eventTime, // time of event
        tmpState_, // state at event
        force_tmp_, // force at event
        newState_);

    // evaluate ode and root fun after event occurred to prepare for subsequent
    // step
    ys_.col(6) = newState_.y;
    t_right_ = t_left_+eventTime;
    (*ode_).ode(t_right_,
     ys_.col(6),force_tmp_,gen_tmp_,diag_tmp_);
    force_.col(6) = force_tmp_;

    if(dimGenerated_>0) generated_.col(6) = gen_tmp_;
    if(diag_.rows() != diag_tmp_.size()) diag_.resize(diag_tmp_.size(),7);
    if(diag_tmp_.size()>0) diag_.col(6) = diag_tmp_;
    events_.col(6) = (*ode_).eventRoot(
      t_right_,
      newState_,
      force_.col(6));

    // set the active eventRoot at event artificially to exactly zero
    // if the event root equation did not change
    // to avoid repeating the event due to numerical inaccuracies
    if(std::fabs(event_tmp_(whichEvent)-events_(whichEvent,6))<1.0e-14){
      events_(whichEvent,6) = 0.0;
    }

    if(! force_.col(6).array().isFinite().all()){
      eventContinue = false;
      std::cout << "Post event Numerical problems" << std::endl;
    }
    return(eventContinue);
  }


  void prepareNext(){
    ys_.col(0) = ys_.col(6);
    force_.col(0) = force_.col(6);
    if(dimGenerated_>0) generated_.col(0) = generated_.col(6);
    if(diag_.rows()>0) diag_.col(0) = diag_.col(6);
    events_.col(0) = events_.col(6);
    t_left_ = t_right_;
  }


  bool setInitialState(const odeState &y0){
    if(y0.y.size() != dim_){
      std::cout << "RKDP54::setInitialState : dimension mismatch" << std::endl;
      return(false);
    }
    t_left_ = 0.0;
    ys_.col(0) = y0.y;


    // first evaluation
    (*ode_).ode(t_left_,
     ys_.col(0),force_tmp_,gen_tmp_,diag_tmp_);


    force_.col(0) = force_tmp_;
    if(dimGenerated_>0) generated_.col(0) = gen_tmp_;
    if(diag_tmp_.size()!= diag_.rows()) diag_.resize(diag_tmp_.size(),7);
    if(diag_tmp_.size()>0) diag_.col(0) = diag_tmp_;

    events_.col(0) = (*ode_).eventRoot(0.0,odeState(ys_.col(0)),force_.col(0));
    return(force_.col(0).array().isFinite().all());
  }

  inline double denseEventRoot_Level(const int which,
                                     const double t){
    return(events_.row(which).dot(denseDotWts(t/eps_)));
  }

  inline double denseEventRoot_LevelDot(const int which,
                                        const double t){
    return(events_.row(which).dot(denseDotDotWts(t/eps_)));
  }

  inline double denseState(const int which,
                           const double t) const {
    return(ys_.coeff(which,0) + force_.row(which).dot(denseWts(t/eps_)));
  }
  inline void denseState(const Eigen::VectorXi &which,
                         const double t,
                         Eigen::Ref<Eigen::VectorXd> out) const {
    if(which.size()!=out.size()){
      std::cout << "Error in RKDP54::denseState" << std::endl;
      throw(1);
    }
    Eigen::VectorXd wts = denseWts(t/eps_);
    for(size_t i=0;i<which.size();i++){
      out.coeffRef(i) = ys_.coeff(which.coeff(i),0) + force_.row(which.coeff(i)).dot(wts);
    }
  }

  inline void denseGenerated_Int(const double t,
                                 Eigen::Ref<Eigen::VectorXd> out) const {
    out = generated_*denseWts(t/eps_);
  }

  inline Eigen::VectorXd denseGenerated_Int(const double t) const {
    return(generated_*denseWts(t/eps_));
  }

  inline void denseGenerated_Level(const double t,
                                   Eigen::Ref<Eigen::VectorXd> out) const {
    out = generated_*denseDotWts(t/eps_);
  }

  inline double eventRootSolver(int &whichDim){
    std::cout << "EventRootSolver should not be called!" << std::endl;
    throw(567);
    whichDim = -1;
    return(eps_);
  }

};


#endif
