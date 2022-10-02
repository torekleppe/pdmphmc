#ifndef _RMHMCPROCESS_HPP_
#define _RMHMCPROCESS_HPP_


template <class targetType,
          class varType,
          template <typename,template <typename> class,typename> class integratorType,
          template <typename> class stepType,
          class metricTensorType,
          class TMType,
          class lambdaType,
          class diagnosticsType>
class RMHMCProcess{
private:
  typedef RMHMCProcess< targetType,
                        varType,
                        integratorType,
                        stepType,
                        metricTensorType,
                        TMType,
                        lambdaType,
                        diagnosticsType> thisType_;
  targetType* t_;
public:
  lambdaType lambda_;
  //massMatrixType fixedMass_;
private:
  // wrapper for this object used for auxiliary integrator
  NUTwrapFirstOrderODE< thisType_ > nutWrap_;
public:
  // auxiliary integrator
  integratorType< NUTwrapFirstOrderODE< thisType_> , stepType ,diagnosticsType > auxInt_;
private:
  diagnosticsType *diag_;

  rng r_;

  // exponentially distributed variate
  double u_;



  size_t dim_;
  size_t dimGen_;
  Eigen::VectorXd q_tmp_,qdot_tmp_,p_tmp_,po_tmp_,grad_tmp_,gen_tmp_,par_tmp_;
  Eigen::VectorXd y_tmp_;
  Eigen::VectorXd Qstore_;

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
  VectorXad dqdot_;
  double ret_;
  stan::math::var dret_;

  // metric tensor
  metricTensorType tensor_;

  // fixed transport map
  TMType TM_;
  STDmetricTensorAdapter<metricTensorType,TMType> asMass_;

  // amtModel

  amt::amtModel<varType,
                typename mt_ad_type<metricTensorType>::type,
                false> mdl_;
  amt::amtModel<varType,
                metricTensorSymbolic,
                false> mdlSym_;

  inline double targetGrad(){
    // assumes input q_tmp_, p_tmp_ are in standardized variables
    par_tmp_ = q_tmp_;
    TM_.toPar(par_tmp_);
    po_tmp_ = p_tmp_;
    TM_.toParInverseJacTransposed(po_tmp_);

    dpar_ = par_tmp_;
    try{
      tensor_.adv.zeroG();
      mdl_.setIndependent(dpar_,dimGen_);
      (*t_)(mdl_);
      dret_ = mdl_.getTarget();
      mdl_.getGenerated(gen_tmp_);

      //dret_ = t_(dpar_,gen_tmp_,tensor_.adv); // target eval
      tensor_.adv.finalize(TM_);
      if(! tensor_.adv.chol()) return std::numeric_limits<double>::quiet_NaN();; // cholesky factorization
      dqdot_ = tensor_.adv.solveG(po_tmp_);
      TM_.toParInverseJac(dqdot_);
      for(int i=0;i<dim_;i++) qdot_tmp_.coeffRef(i) = dqdot_.coeff(i).val();
      tensor_.copyLtoDouble();
      dret_ -= 0.5*p_tmp_.dot(dqdot_) + tensor_.adv.logDetL();
    }
    catch(...){
      stan::math::recover_memory();
      std::cout << "Bad function eval" << std::endl;
      return std::numeric_limits<double>::quiet_NaN();
    }

    ret_ = dret_.val();
    dret_.grad();
    for(int i=0;i<dim_;i++) grad_tmp_.coeffRef(i)=dpar_.coeff(i).adj();
    stan::math::recover_memory();
    TM_.toParJacTransposed(grad_tmp_);
    return(ret_);
  }

public:

  RMHMCProcess(){}
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
             const int dimGen){
    t_ = &target;

    dim_ = dim;
    dimGen_ = dimGen;
    q_tmp_.resize(dim_);
    qdot_tmp_.resize(dim_);
    dqdot_.resize(dim_);
    p_tmp_.resize(dim_);
    grad_tmp_.resize(dim_);
    gen_tmp_.resize(dimGen_);
    y_tmp_.resize(2*dim_+1);



    nutWrap_.setup(*this);
    auxInt_.setup(nutWrap_);
    aux_storePars_.resize(0);


    tensor_.allocate(dim_);
    TM_.setup(dim_);
    asMass_.setup(tensor_,TM_);
    warmup_ = true;

    mdl_.setTenPtr(tensor_.adv);
    mdlSym_.setTenPtr(tensor_.sym);
  }

  void lastWarmupTime(const double t){
    lastWarmupTime_ = t;
    if(t<1.0e-14) warmup_ = false;
  }

  void registerDiagnostics(diagnosticsType &diag){
    diag_ = &diag;
    auxInt_.registerDiagnostics(diag,1);
  }

  int systemDim(){return dim_;}
  int generatedDim(){return dimGen_;}

  // check validity of initial conditions and
  // perform symbolic analysis for the metric tensor
  bool firstEval(const Eigen::VectorXd &q){
    if(q.size()!=dim_){
      std::cout << "bad dimension of initial q" << std::endl;
      return(false);
    }
    // evaluate with symbolic tensor
    mdlSym_.setIndependent(q,dimGen_);

    (*t_)(mdlSym_);

    mdlSym_.getGenerated(gen_tmp_);
    double eval = mdlSym_.getTargetDouble(); //t_(q,gen_tmp_,tensor_.sym);
    std::cout << "first evaluation: value = " << eval << std::endl;
    stan::math::recover_memory();

    // check initial value
    if(! std::isfinite(eval)){
      return(false);
    }

    // check initial generated quantities
    if(! gen_tmp_.array().isFinite().all()){
      std::cout << "generated quantities not finite" << std::endl;
      return(false);
    }

    tensor_.symbolicAnalysis();

    // eval once more to verify also the tensor
    tensor_.adv.zeroG();
    mdl_.setIndependent(q,dimGen_);
    (*t_)(mdl_);
    //eval = t_(q,gen_tmp_,tensor_.dbv);
    tensor_.adv.finalize(TM_);

    if(! tensor_.adv.isFinite()){
      std::cout << "metric tensor not finite" << std::endl;
      stan::math::recover_memory();
      return(false);
    }

    if(! tensor_.adv.chol()) {
      std::cout << "metric tensor not SPD" << std::endl;
      stan::math::recover_memory();
      return(false);
    }
    stan::math::recover_memory();
    return(true);
  }

  // set of ordinary differential equations y'=f(y)
  int dim(){return 2*dim_+1;}
  void ode(const double time,
           const Eigen::VectorXd &y, // y = (q,p,Lambda)
           Eigen::VectorXd &f,
           Eigen::VectorXd &gen,
           Eigen::VectorXd &diagInt){
    //std::cout << "ode start" << std::endl;
    // evaluate Hamiltonian and its gradients
    q_tmp_ = y.head(dim_);
    p_tmp_ = y.segment(dim_,dim_);
    targetGrad();
    //std::cout << "grad done" << std::endl;
    // qdot  = G(q)^{-1}p
    f.head(dim_) = qdot_tmp_;

    // pdot
    f.segment(dim_,dim_) = grad_tmp_;
    gen = gen_tmp_;

    // lambda
    f.coeffRef(2*dim_) = lambda_(q_tmp_, // q
               p_tmp_, // p
               qdot_tmp_, // dotq
               grad_tmp_, // dotp
               asMass_);


    // diagInt : used for adapting the Transport map
    if(warmup_ && TM_.allowsAdaptation()){
      // note uses original parameterization of the system
      TM_.monitor(q_tmp_, // q
                  p_tmp_, // p
                  qdot_tmp_, // dotq
                  grad_tmp_, // dotp
                  diagInt);
    } else {
      if(diagInt.size()!=0) diagInt.resize(0);
    }
    //std::cout << "ode end" << std::endl;
  }

  void SimulateIntialState(const int odeOrder,
                           const Eigen::VectorXd &par0,
                           odeState &state){
    std::cout << "simulating initial state" << std::endl;
#ifdef __TENSOR_DEBUG__
    std::cout << "par0 : \n" << par0 << std::endl;
#endif
    if(odeOrder!=1){
      std::cout << "bad integrator for RMHMCProcess" << std::endl;
      throw(1);
    }
    Eigen::VectorXd q0 = par0;
    TM_.toQ(q0);
    q_last_event_ = q0;
    nut_time_ = 0.0;
#ifdef __TENSOR_DEBUG__
    std::cout << "q0 : \n" << q0 << std::endl;
#endif
    // evaluate the tensor to be able to simulate p
    tensor_.adv.zeroG();
    //tensor_.adv.dumpG();
    mdl_.setIndependent(par0,dimGen_);
    //t_(par0,gen_tmp_,tensor_.dbv);
    (*t_)(mdl_);
    mdl_.getGenerated(gen_tmp_);
    tensor_.adv.finalize(TM_);
    tensor_.copyGtoDouble();
#ifdef __TENSOR_DEBUG__
    std::cout << "initial metric tensor" << std::endl;
    tensor_.adv.dumpG();
    tensor_.dbv.dumpG();
#endif
    tensor_.dbv.chol();
    //std::cout << "SimulateIntialState 1" << std::endl;
    // simulate p
    r_.rnorm(p_tmp_);
    //tensor_.dbv.sqrtM(p_tmp_);
    //TM_.toParJacTransposed(p_tmp_);
    asMass_.sqrtM(p_tmp_);
    stan::math::recover_memory();
    //std::cout << "SimulateIntialState 2" << std::endl;
    if(state.y.size()!=dim()) state.y.resize(dim());
    state.y.head(dim_) = q0; // q
    state.y.segment(dim_,dim_) = p_tmp_; // p
    state.y(2*dim_) = 0.0;
    u_ = -log(r_.runif());
    //std::cout << "SimulateIntialState end" << std::endl;
  }



  int eventRootDim() const {return 3;}
  inline Eigen::VectorXd eventRoot(const double time,
                                   const odeState &state,
                                   const Eigen::VectorXd &f) const {
    Eigen::VectorXd ret(3);
    ret(0) = state.y.coeff(2*dim_)-u_; // regular PDP event
    ret(1) = 1.0;
    ret(2) = (warmup_) ? time-lastWarmupTime_ : 1.0;
    if(nut_time_<1.0e-14 && warmup_ && lambda_.acceptsNUTAdapt()){
      ret(1) = (state.y.head(dim_)-q_last_event_).dot(state.y.segment(dim_,dim_)); // NUT criterion
    }

    return(ret);
  }
  inline int warmupRoot() const {return 2;}



  bool event(const int EventType,
             const double time,
             const odeState &oldState,
             const Eigen::VectorXd &f,
             odeState &newState){
    //std::cout << "Event!, type = " << EventType << std::endl;

    oldState.copyTo(newState);

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
        q_last_event_ = oldState.y.head(dim_);
        time_last_event_ = time;
        nut_time_ = 0.0;
      }



      // note adaptation before momentumUpdate
      if(warmup_ && TM_.allowsAdaptation()) {
        // ensure that par remains unchangend
        par_tmp_ = oldState.y.head(dim_);
        TM_.toPar(par_tmp_);
        TM_.adaptUpdate();
        TM_.toQ(par_tmp_);
        newState.y.head(dim_) = par_tmp_;
        q_last_event_ = par_tmp_;
      }

      par_tmp_ = newState.y.head(dim_);
      TM_.toPar(par_tmp_); // par at event
      p_tmp_ = newState.y.segment(dim_,dim_); // p before update

      // get new tensor at this point
      tensor_.adv.zeroG();
      mdl_.setIndependent(par_tmp_,dimGen_);
      //t_(par_tmp_,gen_tmp_,tensor_.dbv);
      (*t_)(mdl_);
      tensor_.adv.finalize(TM_);
      mdl_.getGenerated(gen_tmp_);
      tensor_.copyGtoDouble();
      tensor_.dbv.chol();
      stan::math::recover_memory();

      lambda_.momentumUpdate(q_tmp_, // q
                             p_tmp_, // p
                             f.head(dim_), // qdot
                             f.segment(dim_,dim_), // pdot
                             asMass_,
                             r_);

      newState.y.segment(dim_,dim_) = p_tmp_; // momentum updated


      newState.y(2*dim_) = 0.0; // reset integrated lambda
      u_ = -log(r_.runif()); // simulate new exp(1) variable




    } else if(EventType==1){ // NUT before PDP event

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
   tensor_.push(eps_,diagInt);
   }*/

  template <class RKstepType>
  void push_RK_step(const int integratorID,
                    const double lastTime,
                    const RKstepType &step){
    if(warmup_ && integratorID==0) TM_.push(step.eps_,step.diagInt_);
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
    tensor_.toJSON(outf);
  }

};
















#endif
