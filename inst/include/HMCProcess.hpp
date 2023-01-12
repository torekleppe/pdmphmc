#ifndef _HMCPROCESSFIRSTORDERODE_HPP_
#define _HMCPROCESSFIRSTORDERODE_HPP_


template <class targetType,
          class varType,
          template <typename,template<typename> class,typename> class integratorType,
          template <typename> class stepType,
          class metricTensorType, // not used in this process
          class TMType,
          class lambdaType,
          class diagnosticsType>
class HMCProcess{
private:
  typedef HMCProcess< targetType,
                      varType,
                      integratorType,
                      stepType,
                      metricTensorType,
                      TMType,
                      lambdaType,
                      diagnosticsType> thisType_;
  targetType* t_;
  identityMass Id_mass_;
public:
  TMType TM_;
  lambdaType lambda_;
  rng r_;
private:
  // wrapper for *this object used for auxiliary integrator
  NUTwrapFirstOrderODE< thisType_ > nutWrap_;
  //startWrapHMCFirstOrderODE< thisType_ > startWrap_;


public:
  // auxiliary integrators
  integratorType< NUTwrapFirstOrderODE< thisType_> , stepType , diagnosticsType> auxInt_;
  //integratorType< startWrapHMCFirstOrderODE< thisType_>, stepType , diagnosticsType> startInt_;



private:
  diagnosticsType *diag_;



  // exponentially distributed variate
  double u_;



  size_t dim_;
  size_t dimGen_;
  Eigen::VectorXd q_tmp_,qdot_tmp_,p_tmp_,grad_tmp_,gen_tmp_,par_tmp_;
  Eigen::VectorXd y_tmp_;


  // adaptation related storage
  bool warmup_;
  double lastWarmupTime_;
  Eigen::VectorXd q_last_event_;
  double time_last_event_;
  double nut_time_;
  double nut_Lam_;
  Eigen::VectorXi aux_storePars_;


  // gradient
  VectorXad dpar_;
  double ret_;


  stan::math::var dret_;

  // metric tensor dummy
  metricTensorDummy tensorDummy_;

  amt::amtModel<varType,metricTensorDummy,false> mdl_;



  inline double targetGrad(){
    par_tmp_ = q_tmp_;
    TM_.toPar(par_tmp_);
    //dpar_ = par_tmp_;
    mdl_.setIndependent(par_tmp_,dimGen_);
    try{
      //dret_ = t_(dpar_,gen_tmp_,tensorDummy_);
      t_->operator()(mdl_);

    }
    catch(...){
      stan::math::recover_memory();
#ifdef __DEBUG__
      std::cout << "Bad function eval" << std::endl;
      std::cout << "par : \n " << par_tmp_ << std::endl;
#endif
      grad_tmp_.setConstant(std::numeric_limits<double>::quiet_NaN());
      return std::numeric_limits<double>::quiet_NaN();
    }
    mdl_.getGenerated(gen_tmp_);
    ret_ = mdl_.getTargetDouble(); //dret_.val();
    mdl_.getTargetGradient(grad_tmp_);
    //dret_.grad();
    //for(int i=0;i<dim_;i++) grad_tmp_.coeffRef(i)=dpar_.coeff(i).adj();
    //stan::math::recover_memory();
    TM_.toParJacTransposed(grad_tmp_);
    return(ret_);
  }

  specialRootSpec sps_;

public:

  HMCProcess(){}

  inline int targetCopies() const {return 1;}
  inline bool massAllowsFixedSubvector() const {
    return TM_.massAllowsFixedSubvector();
  }
  inline void massFixedMiSubvector(const Eigen::VectorXd &fixedMi){
    TM_.setFixedMinv(fixedMi);
  }
  void seed(const int seed){r_.seed(seed);}
  void setup(targetType& target,
             const int dim,
             const int dimGen,
             const amt::constraintInfo& ci){

    if(ci.nonTrivial()){
      std::cout << "HMCProcess cannot be used with constraints!" << std::endl;
      throw(1);
    }

    t_ = &target;
    dim_ = dim;
    dimGen_ = dimGen;
    q_tmp_.resize(dim_);
    qdot_tmp_.resize(dim_);
    p_tmp_.resize(dim_);
    grad_tmp_.resize(dim_);
    gen_tmp_.resize(dimGen_);
    y_tmp_.resize(2*dim_+1);



    nutWrap_.setup(*this);
    //startWrap_.setup(*this);
    auxInt_.setup(nutWrap_);
    //startInt_.setup(startWrap_);
    aux_storePars_.resize(0);

    TM_.setup(dim_);
    Id_mass_.setup(dim_);
    warmup_ = true;

    mdl_.setTenPtr(tensorDummy_);

  }

  inline const specialRootSpec& spr() const {return sps_;}
  void lastWarmupTime(const double t){
    lastWarmupTime_ = t;
    if(t<1.0e-14) warmup_ = false;
  }


  inline void registerDiagnostics(diagnosticsType &diag){
    diag_ = &diag;
    auxInt_.registerDiagnostics(diag,1.0);
    //startInt_.registerDiagnostics(diag,2.0);
  }

  int systemDim() const {return dim_;}
  int generatedDim() const {return dimGen_;}

  // evaluate target

  double target(const Eigen::VectorXd &q){
    if(q.size()!=dim_) return(std::numeric_limits<double>::quiet_NaN());
    par_tmp_ = q;
    TM_.toPar(par_tmp_);

    mdl_.setIndependent(par_tmp_,dimGen_);
    (*t_)(mdl_);
    mdl_.getGenerated(gen_tmp_);

    return mdl_.getTargetDouble();
  }

  // check validity of intial conditions
  bool firstEval(const Eigen::VectorXd &q){
    if(q.size()!=dim_) return(false);
    par_tmp_ = q;
    TM_.toPar(par_tmp_);
    mdl_.setIndependent(par_tmp_,dimGen_);
    (*t_)(mdl_);
    mdl_.getGenerated(gen_tmp_);
    double eval = mdl_.getTargetDouble(); //t_(par_tmp_,gen_tmp_,tensorDummy_);
    std::cout << "first evaluation: value = " << eval << std::endl;
    stan::math::recover_memory();
    if(! std::isfinite(eval)) return(false);
    if(! gen_tmp_.array().isFinite().all()){
      std::cout << "generated quantities not finite" << std::endl;
      return(false);
    }

    return(true);
  }


  // set of ordinary differential equation y'=f(y)
  int dim(){return 2*dim_+1;}
  void ode(const double time,
           const Eigen::VectorXd &y, // y = (q,p,lambda)
           Eigen::VectorXd &f,
           Eigen::VectorXd &gen,
           Eigen::VectorXd &diagInt){
    //std::cout << "proc.ode, t = "  << time << std::endl;
    // qdot
    f.head(dim_) = y.segment(dim_,dim_);

    // pdot
    q_tmp_ = y.head(dim_);
    targetGrad();
    f.segment(dim_,dim_) = grad_tmp_;
    gen = gen_tmp_;

    // lambda
    f.coeffRef(2*dim_) = lambda_(q_tmp_, // q
               y.segment(dim_,dim_), // p
               y.segment(dim_,dim_), // dotq
               grad_tmp_, // dotp
               Id_mass_);


    // diagInt : used for adapting the mass matrix
    if(warmup_ && TM_.allowsAdaptation()){
      TM_.monitor(q_tmp_, // q
                  y.segment(dim_,dim_), // p
                    grad_tmp_, // dotq
                    diagInt);
    } else {
      if(diagInt.size()!=0) diagInt.resize(0);
    }
  }

  void SimulateIntialState(const int odeOrder,
                           const Eigen::VectorXd &par0,
                           odeState &s0){

    if(odeOrder!=1){
      std::cout << "bad odeOrder" << std::endl;
      throw(1);
    }


    Eigen::VectorXd q0 = par0;
    TM_.toQ(q0);
    q_tmp_ = q0;

/*
    // set initial momentum proportional to gradient
    targetGrad();
    // ensure
    double snorm = grad_tmp_.squaredNorm();
    if(snorm>1.0e-3){
      p_tmp_ = sqrt(static_cast<double>(dim_)/snorm)*grad_tmp_;
    } else{
      r_.rnorm(p_tmp_);
    }

    if(odeOrder==1){
      y_tmp_.head(dim_) = q0; // q
      y_tmp_.segment(dim_,dim_) = p_tmp_; // p
      y_tmp_(2*dim_) = 0.0; //Lam

      //startWrap_.setQLastEvent(q_last_event_);
      startInt_.setInitialState(odeState(y_tmp_));
      std::cout << "startint initial state done" << std::endl;
      startInt_.setAbsTol(1.0e-3);
      startInt_.setRelTol(1.0e-3);
      startInt_.run(aux_storePars_,1000.0,2);
      std::cout << "done startInt" << std::endl;


      s0 = startInt_.getStateLastIntegrated();
 */
      if(s0.y.size()!=2*dim_+1) s0.y.resize(2*dim_+1);
      if(p_tmp_.size()!=dim_) p_tmp_.resize(dim_);
      r_.rnorm(p_tmp_);
      s0.y.head(dim_) = q_tmp_;
      s0.y.segment(dim_,dim_) = p_tmp_;
      s0.y(2*dim_) = 0.0;
      q_last_event_ = s0.y.head(dim_);
      nut_time_ = 0.0;
/*
    } else {
      std::cout << "HMCprocess::SimulateIntialState not implemented yet for odeOrder==2";
      throw(1);
    }
*/



    u_ = -log(r_.runif());
  }



  inline int eventRootDim() const {return 3;}
  inline Eigen::VectorXd eventRoot(const double time,
                                   const odeState &state,
                                   const Eigen::VectorXd &f,
                                   const bool afterOde) const {
    Eigen::VectorXd ret(eventRootDim());
    ret(0) = state.y.coeff(2*dim_)-u_; // regular PDP event
    ret(1) = 1.0;
    ret(2) = (warmup_) ? time-lastWarmupTime_ : 1.0;

    if(nut_time_<1.0e-14 && warmup_ && lambda_.acceptsNUTAdapt()){
      ret(1) = (state.y.head(dim_)-q_last_event_).dot(state.y.segment(dim_,dim_)); // NUT criterion
    }
    return(ret);
  }

  inline int warmupRoot() const {return 2;}




  bool event(const rootInfo& rootOut,
             const double time,
             const odeState &oldState,
             const Eigen::VectorXd &f,
             odeState &newState){

    int EventType = rootOut.rootDim_;
    //std::cout << "proc.event" << std::endl;
    oldState.copyTo(newState);
    //return(true);

    if(EventType==0){ // regular PDP event

      if(warmup_ && lambda_.acceptsNUTAdapt()){

        if(nut_time_<1.0e-14){ // PDP event occurred before NUT time found
          //std::cout << "running auxiliary integrator" << std::endl;
          nutWrap_.setQLastEvent(q_last_event_);
          auxInt_.setInitialState(oldState);
          nut_time_ = time - time_last_event_ + auxInt_.run(aux_storePars_,1000.0,2);
          nut_Lam_ = auxInt_.getStateLastIntegrated().y(2*dim_);
          //std::cout << "done auxiliary integrator, nut_Lam = " << nut_Lam_ << std::endl;
        }

        // nut time ready here
        (*diag_).push("nutTime",nut_time_);
        (*diag_).push("nutLam",nut_Lam_);
        lambda_.pushNUTAdaptInfo(nut_time_,nut_Lam_);


        // prepare for calculating new nut time

        time_last_event_ = time;
        nut_time_ = 0.0;
      }



      // note adaptation before momentumUpdate
      // ensure that par does not move
      if(warmup_ && TM_.allowsAdaptation()) {
        par_tmp_ = oldState.y.head(dim_);
        TM_.toPar(par_tmp_);
        TM_.adaptUpdate();
        TM_.toQ(par_tmp_);
        newState.y.head(dim_) = par_tmp_;
      }

      // update momentum
      p_tmp_ = oldState.y.segment(dim_,dim_);
      lambda_.momentumUpdate(newState.y.head(dim_), // q
                             p_tmp_, // p
                             oldState.y.segment(dim_,dim_), // qdot
                             f.segment(dim_,dim_), // pdot
                             Id_mass_,
                             r_);
      newState.y.segment(dim_,dim_) = p_tmp_; // momentum updated


      newState.y(2*dim_) = 0.0; // reset integrated lambda
      u_ = -log(r_.runif()); // simulate new exp(1) variable

      q_last_event_ = newState.y.head(dim_);



    } else if(EventType==1){ // NUT before PDP event

      //ynew = y;
      nut_time_ = time-time_last_event_;
      nut_Lam_ = oldState.y.coeff(2*dim_);

    } else if(EventType==2){ // end of warmup period
      warmup_ = false;

    }

    (*diag_).push("lambda",lambda_.getPars());
    (*diag_).push("L_max",TM_.LMax());
    (*diag_).push("L_min",TM_.LMin());



    return(true);
  }


  /*
  inline void pushDiag(const double eps_,
                       const Eigen::VectorXd &diagInt){
    mass_.push(eps_,diagInt);
  }
   */

  template <class RKstepType>
  void push_RK_step(const int integratorID,
                    const double lastTime,
                    const RKstepType &step){

    if(integratorID==0 && warmup_){

      if(step.diagInt_.size()>0){
        TM_.push(step.eps_,step.diagInt_);
      }
    }
  }

  template <class RKstepType>
  inline void getStorePars(const Eigen::VectorXi &storeParsInds,
                           const double stepTime,
                           const RKstepType &step,
                           Eigen::Ref< Eigen::VectorXd > out){

    step.denseState(storeParsInds,stepTime,out);
    TM_.toPar(storeParsInds,out);
  }

  void auxiliaryDiagnosticsInfo(jsonOut &outf) const {
    TM_.toJSON(outf);
  }

};




#endif

