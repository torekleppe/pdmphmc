#ifndef _RKNDP64_HPP_
#define _RKNDP64_HPP_

/*
 * The following class implements the order 6,4,6 Runge Kutta Nystöm triple
 * RKN6(4)6FD taken from Dormand, Prince, Runge-Kutta-Nystrom Triples,
 * Comput. Math. .Applic.Vol. 13, No. 12, pp. 937-949, 1987.
 *
 * It provides an embedded order 6 and 4 pair, along with dense
 * formula that allow evaluation at each time-point within the
 * current step, with error also of order 6
 *
 */

template <class _ode_type_>
class RKNDP64{
  _ode_type_ *ode_;
  size_t dim_;
  size_t dimGenerated_;
  size_t dimEvent_;






  /*
   * Polynomials used for interpolation purposes
   */
  // for position
  inline Eigen::Matrix<double,6,1> bqstar(const double s) const {
    Eigen::Matrix<double,6,1> ret;
    ret <<
      0.5+(-1.2450142450142450142+(1.5161443494776828110+(-.90669515669515669516+.21367521367521367521*s)*s)*s)*s,
      0.0,
      (1.9859371988705031985+(-3.5776528722949209438+(2.5384011164109503556-.65839370179676583254*s)*s)*s)*s,
      (-1.8559358276926614325+(5.5475867510523291130+(-5.1641335539080937350+1.5949081677229964647*s)*s)*s)*s,
      (2.1786492374727668845+(-6.9269873191441818893+(7.2733366851013909837-2.5138260432378079437*s)*s)*s)*s,
      (-1.0636363636363636364+(3.4409090909090909091+(-3.7409090909090909091+1.3636363636363636364*s)*s)*s)*s;

    return(ret);
  }
  // for position first derivative
  inline Eigen::Matrix<double,6,1> bqdstar(const double s) const {
    Eigen::Matrix<double,6,1> ret;
    ret <<
      1.0+(-3.7350427350427350427+(6.0645773979107312441+(-4.5334757834757834758+1.2820512820512820513*s)*s)*s)*s,
      0.0,
      (5.9578115966115095956+(-14.310611489179683776+(12.692005582054751778-3.9503622107805949953*s)*s)*s)*s,
      (-5.5678074830779842974+(22.190347004209316452+(-25.820667769540468676+9.5694490063379787879*s)*s)*s)*s,
      (6.5359477124183006536+(-27.707949276576727557+(36.366683425506954919-15.082956259426847662*s)*s)*s)*s,
      (-3.1909090909090909091+(13.763636363636363636+(-18.704545454545454545+8.1818181818181818182*s)*s)*s)*s;

    return(ret);
  }

  // derivative wrt s of bqdstar
  inline Eigen::Matrix<double,6,1> dbqdstar(const double s) const {
    Eigen::Matrix<double,6,1> ret;
    ret << -3.7350427350427350427+(12.129154795821462488+(-13.600427350427350427+5.1282051282051282052*s)*s)*s,
           0.0,
           5.9578115966115095956+(-28.621222978359367552+(38.076016746164255334-15.801448843122379981*s)*s)*s,
           -5.5678074830779842974+(44.380694008418632904+(-77.462003308621406028+38.277796025351915152*s)*s)*s,
           6.5359477124183006536+(-55.415898553153455114+(109.10005027652086476-60.331825037707390648*s)*s)*s,
           -3.1909090909090909091+(27.527272727272727272+(-56.113636363636363635+32.727272727272727272*s)*s)*s;

    return(ret);
  }
  // second derivative wrt s of bqdstar
  inline Eigen::Matrix<double,6,1> ddbqdstar(const double s) const {
    Eigen::Matrix<double,6,1> ret;
    ret << 12.129154795821462488 + (-27.200854700854700855 + 15.384615384615384615*s)*s,
           0,
           -28.621222978359367552 + (76.152033492328510668 - 47.404346529367139944*s)*s,
           44.380694008418632904 + (-154.92400661724281205 + 114.83338807605574546*s)*s,
           -55.415898553153455114 + (218.20010055304172952 - 180.99547511312217195*s)*s,
           27.527272727272727272 + (-112.22727272727272727 + 98.181818181818181816*s)*s;

    return(ret);
  }


  Eigen::MatrixXd qs_;
  Eigen::MatrixXd qds_;

  Eigen::MatrixXd fs_;
  Eigen::MatrixXd ms_;

  Eigen::MatrixXd events_;
  Eigen::MatrixXd generated_;
  Eigen::MatrixXd diag_;

  Eigen::VectorXd M0_,M1_,M_tmp_;


  Eigen::VectorXd force_tmp_;
  Eigen::VectorXd gen_tmp_;
  Eigen::VectorXd diag_tmp_;
  Eigen::VectorXd event_tmp_;
  Eigen::VectorXd monitor_tmp_;
  Eigen::VectorXd ptmp_;
  Eigen::VectorXd q1low_;
  Eigen::VectorXd qd1low_;

  odeState tmpState_,newState_;

public:

  double absTol_;
  double relTol_;
  double eps_; // integrator step size
  double stepErr_;
  double t_left_,t_right_; // time on adaptive mesh



  Eigen::VectorXd genIntStep_;
  Eigen::VectorXd diagInt_;

  RKNDP64() : absTol_(1.0e-4), relTol_(1.0e-4), eps_(0.5) {}
  inline int odeOrder() const {return 2;}
  inline double errorOrderHigh() const {return(6.0);}
  constexpr bool hasEventRootSolver() const {return false;}
  inline void setup(_ode_type_ &ode){
    ode_ = &ode;
    dim_ = (*ode_).dim();
    dimGenerated_ = (*ode_).generatedDim();
    dimEvent_ = (*ode_).eventRootDim();
    //dimMonitor_ = (*ode_).monitorDim();

    fs_.resize(dim_,6);
    force_tmp_.resize(dim_);
    qs_.resize(dim_,6);
    qds_.resize(dim_,6);

    if(dimGenerated_>0){
      generated_.resize(dimGenerated_,6);
      gen_tmp_.resize(dimGenerated_);
      gen_tmp_.setZero();
      genIntStep_.resize(dimGenerated_);
    }
    /*
     if(dimMonitor_>0){
     ms_.resize(dimMonitor_,6);
     ms_.setZero();
     M0_.resize(dimMonitor_);
     M0_.setZero();
     M_tmp_.resize(dimMonitor_);
     tmpState_.M.resize(dimMonitor_);
     }
     */
    if(dimEvent_>0){
      events_.resize(dimEvent_,6);
      events_.setZero();
    }

  }


  inline double firstState(const size_t dimension){return qs_.coeff(dimension,0);}
  inline Eigen::VectorXd firstGenerated(){return generated_.col(0);}

  inline void denseY(const double t,
                     Eigen::Ref<Eigen::VectorXd> denseVal) const {
    denseVal = qs_.col(0) + t*(qds_.col(0)+fs_*(t*bqstar(t/eps_)));
  }

  inline Eigen::VectorXd denseY(const double t) const {
    return(qs_.col(0) + t*(qds_.col(0)+fs_*(t*bqstar(t/eps_))));
  }

  inline void denseYdot(const double t,
                        Eigen::Ref<Eigen::VectorXd> denseVal) const {
    denseVal = qds_.col(0) + fs_*(t*bqdstar(t/eps_));
  }

  inline Eigen::VectorXd denseYdot(const double t) const {
    return(qds_.col(0) + fs_*(t*bqdstar(t/eps_)));
  }

  inline void denseM(const double t,
                     Eigen::Ref<Eigen::VectorXd> denseVals) const {
    if(ms_.rows()>0){
    denseVals = M0_ + ms_*(t*bqdstar(t/eps_));
    } else if(denseVals.size()!=0) {
      denseVals.resize(0);
    }
  }

  inline Eigen::VectorXd denseM(const double t) const {
    Eigen::VectorXd ret(ms_.rows());
    denseM(t,ret);
    return(ret);
  }

  inline void denseForce(const double t,
                         Eigen::Ref<Eigen::VectorXd> denseVals) const {
    double s = t/eps_;
    denseVals = fs_*(bqdstar(s)+s*dbqdstar(s));
  }



  inline void denseState(const Eigen::VectorXi &which,
                         const double t,
                         Eigen::Ref<Eigen::VectorXd> out) const {
    if(which.size()!=out.size()){
      std::cout << "Error in RKBDP64::denseState" << std::endl;
      throw(1);
    }
    Eigen::VectorXd wts = t*bqstar(t/eps_);
    for(size_t i=0;i<which.size();i++){
      out.coeffRef(i) = qs_.coeff(which.coeff(i),0) +
        t*(qds_.coeff(which.coeff(i),0)+fs_.row(which.coeff(i)).dot(wts));
    }
  }

  inline void denseGenerated_Level(const double t,
                                   Eigen::Ref<Eigen::VectorXd> out) const {
    double s = t/eps_;
    out = generated_*(bqdstar(s)+s*dbqdstar(s));
  }

  inline void denseGenerated_Int(const double t,
                                 Eigen::Ref<Eigen::VectorXd> out) const {
    out = generated_*(t*bqdstar(t/eps_));
  }

  inline Eigen::VectorXd denseGenerated_Int(const double t) const {
    Eigen::VectorXd ret(generated_.rows());
    denseGenerated_Int(t,ret);
    return(ret);
  }

  inline double denseEventRoot_Level(const int which,
                                     const double t) const {
    double s = t/eps_;
    return(events_.row(which).dot(bqdstar(s)+s*dbqdstar(s)));
  }
  inline double denseEventRoot_LevelDot(const int which,
                                        const double t) const {
    double s = t/eps_;
    return(events_.row(which).dot((1.0/eps_)*(2.0*dbqdstar(s) + s*ddbqdstar(s))));
  }




  bool setInitialState(const odeState &y0){
    if(y0.y.size() != dim_ || y0.ydot.size() != dim_){
      std::cout << "RKNDP64::setInitialState : dimension mismatch " << std::endl;
      return(false);
    }
    t_left_ = 0.0;
    qs_.col(0) = y0.y;
    qds_.col(0) = y0.ydot;

    (*ode_).ode(t_left_,qs_.col(0),
     force_tmp_,
     gen_tmp_,
     diag_tmp_);
    fs_.col(0) = force_tmp_;
    if(dimGenerated_>0) generated_.col(0) = gen_tmp_;
    if(diag_tmp_.size()!= diag_.rows()) diag_.resize(diag_tmp_.size(),7);
    if(diag_tmp_.size()>0) diag_.col(0) = diag_tmp_;


    monitor_tmp_  = (*ode_).monitor(t_left_,
                     qs_.col(0),
                     qds_.col(0),
                     fs_.col(0));
    if(ms_.rows()!=monitor_tmp_.size() || ms_.cols() != 6) ms_.resize(monitor_tmp_.size(),6);
    ms_.col(0) = monitor_tmp_;
    if(y0.M.size()==monitor_tmp_.size()){
      M0_ = y0.M;
    } else {
      M0_.resize(monitor_tmp_.size());
      M0_.setZero();
    }


    if(dimEvent_>0){
      tmpState_.y = qs_.col(0);
      tmpState_.ydot = qds_.col(0);
      tmpState_.M = M0_;
      events_.col(0) = (*ode_).eventRoot(t_left_,
                  tmpState_,
                  fs_.col(0));
    }
    return(fs_.col(0).array().isFinite().all());
  }



  void prepareNext(){
    qs_.col(0) = qs_.col(5);
    qds_.col(0) = qds_.col(5);
    fs_.col(0) = fs_.col(5);
    if(dimGenerated_>0) generated_.col(0) = generated_.col(5);
    if(diag_.rows()>0) diag_.col(0) = diag_.col(5);
    if(dimEvent_>0) events_.col(0) = events_.col(5);
    if(ms_.rows()>0){
      M0_ = M1_;
      ms_.col(0) = ms_.col(5);
    }
    t_left_ = t_right_;
  }


  bool step(){
    //std::cout << "step start" << std::endl;
    double epssq = pow(eps_,2);
    t_right_ = t_left_ + eps_;

    //std::cout << "step" << std::endl;
    // first eval
    qs_.col(1) = qs_.col(0) + (0.12929590313670441529*eps_)*qds_.col(0)+
      epssq*0.008358715283968025328*fs_.col(0);
    (*ode_).ode(t_left_ + 0.12929590313670441529*eps_,qs_.col(1),
     force_tmp_,
     gen_tmp_,
     diag_tmp_);
    fs_.col(1) = force_tmp_;
    if(dimGenerated_>0) generated_.col(1) = gen_tmp_;
    if(diag_tmp_.size()>0) diag_.col(1) = diag_tmp_;

    if(! fs_.col(1).array().isFinite().all()) return(false);

    // second eval
    qs_.col(2) = qs_.col(0) + (0.25859180627340883058*eps_)*qds_.col(0) +
      (epssq*0.011144953711957367104)*fs_.col(0) +
      (epssq*0.022289907423914734209)*fs_.col(1);
    (*ode_).ode(t_left_ + 0.25859180627340883058*eps_,qs_.col(2),
     force_tmp_,
     gen_tmp_,
     diag_tmp_);
    fs_.col(2) = force_tmp_;
    if(dimGenerated_>0) generated_.col(2) = gen_tmp_;
    if(diag_tmp_.size()>0) diag_.col(2) = diag_tmp_;

    if(! fs_.col(2).array().isFinite().all()) return(false);

    // third eval
    qs_.col(3) = qs_.col(0) + (0.67029708261548005830*eps_)*qds_.col(0) +
      (epssq*0.14547474280109178590)*fs_.col(0) +
      (epssq*(-0.22986064052264747311))*fs_.col(1) +
      (epssq*0.30903498720296753653)*fs_.col(2);

    (*ode_).ode(t_left_ + 0.67029708261548005830*eps_,qs_.col(3),
     force_tmp_,
     gen_tmp_,
     diag_tmp_);
    fs_.col(3) = force_tmp_;
    if(dimGenerated_>0) generated_.col(3) = gen_tmp_;
    if(diag_tmp_.size()>0) diag_.col(3) = diag_tmp_;
    if(! fs_.col(3).array().isFinite().all()) return(false);

    // fourth eval
    qs_.col(4) = qs_.col(0) + (0.9*eps_)*qds_.col(0) +
      (epssq*(-0.20766826295078995434))*fs_.col(0) +
      (epssq*0.68636678429251431227)*fs_.col(1) +
      (epssq*(-0.19954927787234925220))*fs_.col(2) +
      (epssq*0.12585075653062489426)*fs_.col(3);

    (*ode_).ode(t_left_ + 0.9*eps_,qs_.col(4),
     force_tmp_,
     gen_tmp_,
     diag_tmp_);
    fs_.col(4) = force_tmp_;
    if(dimGenerated_>0) generated_.col(4) = gen_tmp_;
    if(diag_tmp_.size()>0) diag_.col(4) = diag_tmp_;
    if(! fs_.col(4).array().isFinite().all()) return(false);


    // fifth eval
    qs_.col(5) = qs_.col(0) + eps_*qds_.col(0) +
      (epssq*0.078110161443494776828)*fs_.col(0) +
      (epssq*0.28829174118976677768)*fs_.col(2) +
      (epssq*0.12242553717457041018)*fs_.col(3) +
      (epssq*0.011172560192168035305)*fs_.col(4);

    (*ode_).ode(t_left_ + eps_,qs_.col(5),
     force_tmp_,
     gen_tmp_,
     diag_tmp_);
    fs_.col(5) = force_tmp_;
    if(dimGenerated_>0) generated_.col(5) = gen_tmp_;
    if(diag_tmp_.size()>0) diag_.col(5) = diag_tmp_;
    if(! fs_.col(5).array().isFinite().all()) return(false);

    /* Done evals; qs_.col(5) is the high order position approximation
     *
     */

    // high order q-dot (used for advancing the integrator)
    qds_.col(5) = qds_.col(0) +
      (eps_*0.078110161443494776828)*fs_.col(0) +
      (eps_*0.38884347870598260272)*fs_.col(2) +
      (eps_*0.37132075792884226740)*fs_.col(3) +
      (eps_*0.11172560192168035305)*fs_.col(4) +
      (eps_*0.05)*fs_.col(5);

    //std::cout << "evals done" << std::endl;

    /*
     * Error estimation
     */

    // low order approximations for error estimates
    q1low_ = qs_.col(0) + eps_*qds_.col(0) +
      (epssq*1.0588592603704182782)*fs_.col(0) +
      (epssq*(-2.4067513719244520532))*fs_.col(1) +
      (epssq*1.8478921115540337750)*fs_.col(2);

    qd1low_ = qds_.col(0) + (eps_*0.054605887939221272555)*fs_.col(0) +
      (eps_*0.46126678590362684429)*fs_.col(2) +
      (eps_*0.19588085947931265629)*fs_.col(3) +
      (eps_*0.38824646667783922686)*fs_.col(4) +
      (eps_*(-0.1))*fs_.col(5);


    // from position
    ptmp_ = ((relTol_*(qs_.col(5).cwiseAbs().cwiseMax(qs_.col(0).cwiseAbs()))).array()+absTol_).matrix();
    double errPos = (qs_.col(5)-q1low_).cwiseAbs().cwiseQuotient(ptmp_).maxCoeff();
    // from position dot
    ptmp_ = ((relTol_*(qds_.col(5).cwiseAbs().cwiseMax(qds_.col(0).cwiseAbs()))).array()+absTol_).matrix();
    double errDot = (qds_.col(5)-qd1low_).cwiseAbs().cwiseQuotient(ptmp_).maxCoeff();
    stepErr_ = std::fmax(errPos,errDot);

    /*
     * integrated quantities
     *
     */
    if(dimGenerated_>0){
      genIntStep_ = (eps_*0.078110161443494776828)*generated_.col(0) +
        (eps_*0.38884347870598260272)*generated_.col(2) +
        (eps_*0.37132075792884226740)*generated_.col(3) +
        (eps_*0.11172560192168035305)*generated_.col(4) +
        (eps_*0.05)*generated_.col(5);
    }

    if(diag_.rows()>0){
      diagInt_ = (eps_*0.078110161443494776828)*diag_.col(0) +
        (eps_*0.38884347870598260272)*diag_.col(2) +
        (eps_*0.37132075792884226740)*diag_.col(3) +
        (eps_*0.11172560192168035305)*diag_.col(4) +
        (eps_*0.05)*diag_.col(5);
    } else {
      if(diagInt_.size()>0) diagInt_.resize(0);
    }

    /*
     * make interpolated ydot values
     */
    if(dimEvent_>0 || ms_.rows()>0){
      // first eval
      denseYdot(0.12929590313670441529*eps_,qds_.col(1));
      denseYdot(0.25859180627340883058*eps_,qds_.col(2));
      denseYdot(0.67029708261548005830*eps_,qds_.col(3));
      denseYdot(0.9*eps_,qds_.col(4));
    }

    /*
     * Evals of the monitor function
     */
    //std::cout << "eval monitor start" << std::endl;
    if(ms_.rows()>0){
      ms_.col(1) = (*ode_).monitor(t_left_+0.12929590313670441529*eps_,
              qs_.col(1),
              qds_.col(1),
              fs_.col(1));
      ms_.col(2) = (*ode_).monitor(t_left_+0.25859180627340883058*eps_,
              qs_.col(2),
              qds_.col(2),
              fs_.col(2));
      ms_.col(3) = (*ode_).monitor(t_left_+0.67029708261548005830*eps_,
              qs_.col(3),
              qds_.col(3),
              fs_.col(3));
      ms_.col(4) = (*ode_).monitor(t_left_+0.9*eps_,
              qs_.col(4),
              qds_.col(4),
              fs_.col(4));
      ms_.col(5) = (*ode_).monitor(t_right_,
              qs_.col(5),
              qds_.col(5),
              fs_.col(5));
      M1_ = M0_ + (eps_*0.078110161443494776828)*ms_.col(0) +
        (eps_*0.38884347870598260272)*ms_.col(2) +
        (eps_*0.37132075792884226740)*ms_.col(3) +
        (eps_*0.11172560192168035305)*ms_.col(4) +
        (eps_*0.05)*ms_.col(5);
    }
    //std::cout << "eval monitor end" << std::endl;


    /*
     * Evals of the event root function
     */

    if(dimEvent_>0){
      if(tmpState_.M.size()!=ms_.rows()) tmpState_.M.resize(ms_.rows());
      if(ms_.rows()>0) denseM(0.12929590313670441529*eps_,tmpState_.M);
      tmpState_.y = qs_.col(1);
      tmpState_.ydot = qds_.col(1);
      events_.col(1) = (*ode_).eventRoot(t_left_+0.12929590313670441529*eps_,
                  tmpState_,
                  fs_.col(1));

      if(ms_.rows()>0) denseM(0.25859180627340883058*eps_,tmpState_.M);
      tmpState_.y = qs_.col(2);
      tmpState_.ydot = qds_.col(2);
      events_.col(2) = (*ode_).eventRoot(t_left_+0.25859180627340883058*eps_,
                  tmpState_,
                  fs_.col(2));

      if(ms_.rows()>0) denseM(0.67029708261548005830*eps_,tmpState_.M);
      tmpState_.y = qs_.col(3);
      tmpState_.ydot = qds_.col(3);
      events_.col(3) = (*ode_).eventRoot(t_left_+0.67029708261548005830*eps_,
                  tmpState_,
                  fs_.col(3));

      if(ms_.rows()>0) denseM(0.9*eps_,tmpState_.M);
      tmpState_.y = qs_.col(4);
      tmpState_.ydot = qds_.col(4);
      events_.col(4) = (*ode_).eventRoot(t_left_+0.9*eps_,
                  tmpState_,
                  fs_.col(4));

      tmpState_.y = qs_.col(5);
      tmpState_.ydot = qds_.col(5);
      if(ms_.rows()>0) tmpState_.M = M1_;
      events_.col(5) = (*ode_).eventRoot(t_right_,
                  tmpState_,
                  fs_.col(5));
    }
    //std::cout << "step end" << std::endl;
    return(true);

  }

  inline odeState lastState() const {
    if(ms_.rows()>0){
      return(odeState(qs_.col(5),qds_.col(5),M1_));
    }
    return(odeState(qs_.col(5),qds_.col(5)));
  }

  inline int monitorDim() const {return ms_.rows();}
  inline Eigen::VectorXd monitorIntStep() const {return M1_-M0_;}

  inline bool event(const int whichEvent,
                    const double eventTime){
    //std::cout << "event start" << std::endl;
    // state before event
    denseY(eventTime,tmpState_.y);
    denseYdot(eventTime,tmpState_.ydot);
    if(ms_.rows()>0) denseM(eventTime,tmpState_.M);
    denseForce(eventTime,force_tmp_);
    // evaluate eventRoot
    if(dimEvent_>0){
      event_tmp_ = (*ode_).eventRoot(t_left_+eventTime,
                    tmpState_,force_tmp_);
      if(std::fabs(event_tmp_(whichEvent))>100.0*absTol_){
        std::cout << "eventRoot at interpolated state: " << event_tmp_(whichEvent) << std::endl;
      }
    }
    bool eventContinue = (*ode_).event(
      whichEvent,
      t_left_+eventTime,
      tmpState_,
      force_tmp_,
      newState_);

    qs_.col(5) = newState_.y;
    qds_.col(5) = newState_.ydot;
    M1_ = newState_.M;
    t_right_ = t_left_+eventTime;

    (*ode_).ode(t_right_,qs_.col(5),
     force_tmp_,
     gen_tmp_,
     diag_tmp_);
    fs_.col(5) = force_tmp_;
    if(dimGenerated_>0) generated_.col(5) = gen_tmp_;
    if(diag_tmp_.size()>0) diag_.col(5) = diag_tmp_;

    monitor_tmp_ = (*ode_).monitor(t_right_,
                    qs_.col(5),
                    qds_.col(5),
                    fs_.col(5));
    if(monitor_tmp_.size() != ms_.rows()){
      ms_.resize(monitor_tmp_.size(),6);
      if(M1_.size() != monitor_tmp_.size()){
        std::cout << "WARNING: post event mismatch in size of M and monitor output; M reset to zero" << std::endl;
        M1_.resize(monitor_tmp_.size());
        if(M1_.size()>0) M1_.setZero();
        newState_.M.resize(monitor_tmp_.size());
        if(newState_.M.size()>0) newState_.M.setZero();
      }
    }

    if(ms_.rows()>0) ms_.col(5) = monitor_tmp_;

    if(dimEvent_>0){
      events_.col(5) = (*ode_).eventRoot(t_right_,
                  newState_,force_tmp_);
      if(std::fabs(event_tmp_(whichEvent)-events_(whichEvent,5))<1.0e-14){
        events_(whichEvent,5) = 0.0;
      }
    }

    if(! fs_.col(5).array().isFinite().all()) {
      eventContinue=false;
      std::cout << "Post event Numerical problems" << std::endl;
    }
    //std::cout << "event done" << std::endl;

    return(eventContinue);
  }
  inline double eventRootSolver(int &whichDim){
    std::cout << "EventRootSolver should not be called!" << std::endl;
    throw(567);
    whichDim = -1;
    return(eps_);
  }

};





#endif
