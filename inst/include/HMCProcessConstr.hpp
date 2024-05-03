#ifndef _HMCPROCESSFIRSTORDERODE_CONSTR_HPP_
#define _HMCPROCESSFIRSTORDERODE_CONSTR_HPP_


#ifndef _BOUNDARY_KERNEL_TYPE_
#define _BOUNDARY_KERNEL_TYPE_ 2
#endif

#ifdef _COLLISION_BOUNDARY_KERNEL_
#undef _BOUNDARY_KERNEL_TYPE_
#define _BOUNDARY_KERNEL_TYPE_ 0
#endif

#ifdef _EXIT_ANGLE_THRESH_BOUNDARY_KERNEL_
#undef _BOUNDARY_KERNEL_TYPE_
#define _BOUNDARY_KERNEL_TYPE_ 1
#endif

#ifdef _RANDOMIZED_BOUNDARY_KERNEL_
#undef _BOUNDARY_KERNEL_TYPE_
#define _BOUNDARY_KERNEL_TYPE_ 2
#endif

#ifndef _EXIT_ANGLE_THRESH_
#define _EXIT_ANGLE_THRESH_ 0.08715574 // corresponds to 85 degrees
#endif




template < class _process_type_ >
class NUTwrap_HMCProcessConstr{
  _process_type_ *proc_;
  Eigen::VectorXd q_last_event_;
  Eigen::VectorXd genDummy_;
  Eigen::VectorXd diagIntDummy_;
  int dim_;

public:
  NUTwrap_HMCProcessConstr(){}
  void setup(_process_type_ &proc){
    proc_ = &proc;
    dim_ = (*proc_).systemDim();
    genDummy_.resize((*proc_).generatedDim());
  }
  int dim(){return (*proc_).dim();}
  int generatedDim(){ return 0;}
  int eventRootDim(){return (*proc_).eventRootDim();}

  secondOrderRep secondOrderRepresentation() const {return (*proc_).secondOrderRepresentation();}

  void setQLastEvent(const Eigen::VectorXd &q){q_last_event_=q;}

  void ode(const double time,
           const Eigen::VectorXd &y, // y = (q,p,lambda)
           Eigen::VectorXd &f,
           Eigen::VectorXd &gen,
           Eigen::VectorXd &diagInt){

    (*proc_).ode(time,y,f,genDummy_,diagIntDummy_);
    if(gen.size()>0) gen.resize(0);
    if(diagInt.size()>0) diagInt.resize(0);
  }
  inline Eigen::VectorXd eventRoot(const double time,
                                   const odeState &state,
                                   const Eigen::VectorXd &f,
                                   const bool afterOde) const {
    // compute original event root to get constraints in the correct place
    Eigen::VectorXd ret = (*proc_).eventRoot(time,state,f,afterOde);
    // overwrite roots not needed here, first NUT criterion
    ret(0) = (state.y.head(dim_)-q_last_event_).dot(state.y.segment(dim_,dim_));
    ret(1) = 1.0;
    ret(2) = 1.0;
    return(ret);
  }
  inline int warmupRoot() const {return -1;} // no warmup root

  bool event(const rootInfo& eventOut,
             const double time,
             const odeState &oldState,
             const Eigen::VectorXd &f,
             odeState &newState){
    if(eventOut.rootType_==0 && eventOut.rootDim_==0){
      // done with NUT integration leg
      newState.y = oldState.y;
      return(false);
    } else {
      // constraint-related events
      return((*proc_).event(eventOut,time,oldState,f,newState));
    }
  }


  template <class stepType>
  void push_RK_step(const int integratorID,
                    const double lastTime,
                    const stepType &step){}

  template <class stepType>
  inline void getStorePars(const Eigen::VectorXi &storeParsInds,
                           const double stepTime,
                           const stepType &step,
                           Eigen::Ref<Eigen::VectorXd> out){
    (*proc_).getStorePars(storeParsInds,stepTime,step,out);
  }
  /*
   inline void specialRoots(specialRootSpec& sps) {
   (*proc_).specialRoots(sps);
   std::cout << "HMCProcessConstrNUTwrap::specialRoots:\n" << sps << std::endl;
   }
   */
  inline const specialRootSpec& spr() const {return (*proc_).spr();}



}; // nut-wrap template



template <class targetType,
          class varType,
          template <typename,template<typename> class,typename> class integratorType,
          template <typename> class stepType,
          class metricTensorType, // not used in this process
          class TMType,
          class lambdaType,
          class diagnosticsType>
class HMCProcessConstr{
private:
  typedef HMCProcessConstr< targetType,
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
  NUTwrap_HMCProcessConstr< thisType_ > nutWrap_;
  //startWrapHMCFirstOrderODE< thisType_ > startWrap_;


public:
  // auxiliary integrators
  integratorType< NUTwrap_HMCProcessConstr< thisType_> , stepType , diagnosticsType> auxInt_;
  //integratorType< startWrapHMCFirstOrderODE< thisType_>, stepType , diagnosticsType> startInt_;



private:
  diagnosticsType *diag_;



  // exponentially distributed variate
  double u_;



  size_t dim_;
  size_t dimGen_;
  amt::constraintInfo ci_;
  specialRootSpec sps_;

  Eigen::VectorXd q_tmp_,qdot_tmp_,p_tmp_,grad_tmp_,gen_tmp_,par_tmp_;
  Eigen::VectorXd y_tmp_;
  Eigen::VectorXd normal_vec_;
  Eigen::VectorXd linJacSNorms_,splinJacSNorms_;
  Eigen::VectorXd linJacTmp_;
  Eigen::VectorXd boundary_z_;

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



  void updateSPR(){
    if(ci_.numLin_>0){
      if(sps_.linRootJac_.rows()!=ci_.linJac_.rows() ||
         sps_.linRootJac_.cols()!=dim()){
        sps_.linRootJac_.resize(ci_.linJac_.rows(),dim());
        sps_.linRootJac_.setZero();
        linJacSNorms_.resize(ci_.linJac_.rows());
      }

      //sps_.linRootJac_.leftCols(ci_.linJac_.cols()) = ci_.linJac_;
      for(size_t i=0;i<sps_.linRootJac_.rows();i++){
        linJacTmp_ = ci_.linJac_.row(i);
        TM_.toParJac(linJacTmp_);
        sps_.linRootJac_.row(i).head(dim_) = linJacTmp_;
      }
      sps_.linRootConst_ = ci_.linConst_ + ci_.linJac_*TM_.getMu();

      for(size_t i=0;i<ci_.linJac_.rows();i++) linJacSNorms_.coeffRef(i) = sps_.linRootJac_.row(i).head(dim_).squaredNorm();
    }
    if(ci_.numspLin_>0){
      if(sps_.spLinRootJac_.rows()!=ci_.splinJac_.rows()){
        sps_.spLinRootJac_ = ci_.splinJac_;
        sps_.spLinRootJac_.setCols(dim());
        splinJacSNorms_.resize(ci_.splinJac_.rows());
      } else {
        sps_.spLinRootJac_.cpValsFrom(ci_.splinJac_);
      }
      sps_.spLinRootJac_.rightMultiplyDiag(TM_.toParJacDiag());
      sps_.spLinRootConst_ = ci_.splinConst_ + ci_.splinJac_*TM_.getMu();
      for(size_t i=0;i<ci_.splinJac_.rows();i++) splinJacSNorms_.coeffRef(i) = sps_.spLinRootJac_.rowSquaredNorm(i);
    }

    if(ci_.splinL1Jac_.size()>0){
      if(sps_.spLinL1RootJac_.size()==0){
        for(size_t i=0;i<ci_.splinL1Jac_.size();i++){
          sps_.spLinL1RootJac_.push_back(ci_.splinL1Jac_[i]);
          sps_.spLinL1RootJac_.back().setCols(dim());
          sps_.spLinL1RootConst_.push_back(ci_.splinL1Const_[i]);
          sps_.spLinL1RootRhs_.push_back(ci_.splinL1Rhs_[i]);
        }
      } else {
        for(size_t i=0;i<ci_.splinL1Jac_.size();i++){
          sps_.spLinL1RootJac_[i].cpValsFrom(ci_.splinL1Jac_[i]);
        }
      }
      for(size_t i=0;i<ci_.splinL1Jac_.size();i++){
        sps_.spLinL1RootJac_[i].rightMultiplyDiag(TM_.toParJacDiag());
        sps_.spLinL1RootConst_[i] = ci_.splinL1Const_[i] + ci_.splinL1Jac_[i]*TM_.getMu();
      }
    }

    if(ci_.splinL2Jac_.size()>0){
      if(sps_.spLinL2RootJac_.size()==0){
        for(size_t i=0;i<ci_.splinL2Jac_.size();i++){
          sps_.spLinL2RootJac_.push_back(ci_.splinL2Jac_[i]);
          sps_.spLinL2RootJac_.back().setCols(dim());
          sps_.spLinL2RootConst_.push_back(ci_.splinL2Const_[i]);
          sps_.spLinL2RootRhs_.push_back(ci_.splinL2Rhs_[i]);
        }
      } else {
        for(size_t i=0;i<ci_.splinL2Jac_.size();i++){
          sps_.spLinL2RootJac_[i].cpValsFrom(ci_.splinL2Jac_[i]);
        }
      }
      for(size_t i=0;i<ci_.splinL2Jac_.size();i++){
        sps_.spLinL2RootJac_[i].rightMultiplyDiag(TM_.toParJacDiag());
        sps_.spLinL2RootConst_[i] = ci_.splinL2Const_[i] + ci_.splinL2Jac_[i]*TM_.getMu();
      }
    }


    if(ci_.splinFJac_.size()>0){
      if(sps_.spLinFRootJac_.size()==0){
        for(size_t i=0;i<ci_.splinFJac_.size();i++){
          sps_.spLinFRootJac_.push_back(ci_.splinFJac_[i]);
          sps_.spLinFRootJac_.back().setCols(dim());
          sps_.spLinFRootConst_.push_back(ci_.splinFConst_[i]);
        }
        sps_.spLinFRootFun_ = ci_.splinFfun_;
      } else {
        for(size_t i=0;i<ci_.splinFJac_.size();i++){
          sps_.spLinFRootJac_[i].cpValsFrom(ci_.splinFJac_[i]);
        }
      }
      for(size_t i=0;i<ci_.splinFJac_.size();i++){
        sps_.spLinFRootJac_[i].rightMultiplyDiag(TM_.toParJacDiag());
        sps_.spLinFRootConst_[i] = ci_.splinFConst_[i] + ci_.splinFJac_[i]*TM_.getMu();
      }
    }



  }


public:

  HMCProcessConstr(){}

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

    t_ = &target;
    ci_ = ci;
    dim_ = dim;
    dimGen_ = dimGen;
    q_tmp_.resize(dim_);
    qdot_tmp_.resize(dim_);
    p_tmp_.resize(dim_);
    grad_tmp_.resize(dim_);
    gen_tmp_.resize(dimGen_);
    y_tmp_.resize(2*dim_+1);



    nutWrap_.setup(*this);
    auxInt_.setup(nutWrap_);
    aux_storePars_.resize(0);

    TM_.setup(dim_);
    Id_mass_.setup(dim_);
    warmup_ = true;

    mdl_.setTenPtr(tensorDummy_);

    updateSPR();
  }

  void lastWarmupTime(const double t){
    lastWarmupTime_ = t;
    if(t<1.0e-14) warmup_ = false;
  }


  inline void registerDiagnostics(diagnosticsType &diag){
    diag_ = &diag;
    auxInt_.registerDiagnostics(diag,1.0);
    //startInt_.registerDiagnostics(diag,2.0);
  }

  inline int systemDim() const {return dim_;}
  inline int generatedDim() const {return dimGen_;}

  secondOrderRep secondOrderRepresentation() const {
    secondOrderRep ret;
    ret.pos_first_ = 0;
    ret.pos_last_ = dim_-1;
    ret.vel_first_ = dim_;
    ret.vel_last_ = 2*dim_-1;
    ret.int_first_ = 2*dim_;
    ret.int_last_ = 2*dim_;
    ret.nonTrivial_ = true;
    ret.hasInt_ = true;
    return ret;
  }


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
    //std::cout << "first evaluation: value = " << eval << std::endl;
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
    //std::cout << "SimulateIntialState" << std::endl;
    Eigen::VectorXd q0 = par0;
    TM_.toQ(q0);
    q_tmp_ = q0;
    if(s0.y.size()!=2*dim_+1) s0.y.resize(2*dim_+1);
    s0.y.head(dim_) = q0;

    if(p_tmp_.size()!=dim_) p_tmp_.resize(dim_);
    r_.rnorm(p_tmp_);

    s0.y.segment(dim_,dim_) = p_tmp_;
    s0.y(2*dim_) = 0.0;




    q_last_event_ = s0.y.head(dim_);
    nut_time_ = 0.0;
    u_ = -log(r_.runif());
    //std::cout << "SimulateIntialState done" << std::endl;
  }



  inline int eventRootDim() const {return 3+ci_.numNonLin_;}
  inline Eigen::VectorXd eventRoot(const double time,
                                   const odeState &state,
                                   const Eigen::VectorXd &f,
                                   const bool afterOde){
    Eigen::VectorXd ret(eventRootDim());
    ret(0) = state.y.coeff(2*dim_)-u_; // regular PDP event
    ret(1) = 1.0;
    ret(2) = (warmup_) ? time-lastWarmupTime_ : 1.0;

    if(nut_time_<1.0e-14 && warmup_ && lambda_.acceptsNUTAdapt()){
      ret(1) = (state.y.head(dim_)-q_last_event_).dot(state.y.segment(dim_,dim_)); // NUT criterion
    }
    // non-linear constraints
    if(ci_.numNonLin_>0){
      if(! afterOde){

        par_tmp_ = state.y.head(dim_);
        TM_.toPar(par_tmp_);
        mdl_.setIndependent(par_tmp_,dimGen_);
        try{t_->operator()(mdl_);}
        catch(...){
          stan::math::recover_memory();
          std::cout << "bad function eval in eventRoot()" << std::endl;
        }
      }
      ret.tail(ci_.numNonLin_) = mdl_.getNonLinConstraint();
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
    oldState.copyTo(newState);

    if(rootOut.rootType_==0){

      if(EventType==0){ // regular PDP event

        if(warmup_ && lambda_.acceptsNUTAdapt()){

          if(nut_time_<1.0e-14){ // PDP event occurred before NUT time found

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
          updateSPR();
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
      } else if(EventType>2) { // non-linear constraint
        // calculate gradient of the relevant constraint
        // note that target eval as this point was already done
        //std::cout << "collision event" << std::endl;
        mdl_.nonLinConstraintGradient(EventType-3,normal_vec_);
        //std::cout << "normal vector \n " << normal_vec_ << std::endl;
        // normal vector in standardized parameterization
        TM_.toParJacTransposed(normal_vec_);
        double fac = 2.0*oldState.y.segment(dim_,dim_).dot(normal_vec_)/normal_vec_.squaredNorm();
        if(fac<0.0){
          newState.y.segment(dim_,dim_) -= fac*normal_vec_;
        } else {
          std::cout << "bad direction into constraint" << std::endl;
        }
      } else {
        std::cout << "bad rootDim into ODE class::event()" <<std::endl;
        throw(45);
      }

    } else if (rootOut.rootType_==1) {
      //std::cout << "linear root event #" <<rootOut.rootDim_ << " at time " << rootOut.rootTime_ << std::endl;
      double fac = 2.0*sps_.linRootJac_.row(rootOut.rootDim_).head(dim_).dot(oldState.y.segment(dim_,dim_))/linJacSNorms_.coeff(rootOut.rootDim_);
      if(fac<0.0) {
        if constexpr(_BOUNDARY_KERNEL_TYPE_==1){// safeguared collision event
          double phi = 0.5*fac*sqrt(linJacSNorms_.coeff(rootOut.rootDim_))/oldState.y.segment(dim_,dim_).norm();
          if(std::fabs(phi)>_EXIT_ANGLE_THRESH_){
            newState.y.segment(dim_,dim_) -= fac*sps_.linRootJac_.row(rootOut.rootDim_).head(dim_);
          } else {
            for(size_t i=0;i<dim_;i++){
              if(std::fabs(sps_.linRootJac_.coeff(rootOut.rootDim_,i))>1.0e-14){
                newState.y.coeffRef(dim_+i) = -oldState.y.coeff(dim_+i);
              }
            }
          }
        } else if constexpr(_BOUNDARY_KERNEL_TYPE_==2) { // randomized collision event
          Eigen::VectorXd zz(dim_);
          r_.rnorm(zz);
          double theta = (zz+oldState.y.segment(dim_,dim_)).dot(sps_.linRootJac_.row(rootOut.rootDim_).head(dim_))/linJacSNorms_.coeff(rootOut.rootDim_);
          for(size_t i=0;i<dim_;i++){
            if(std::fabs(sps_.linRootJac_.coeff(rootOut.rootDim_,i))>1.0e-14){
              newState.y.coeffRef(dim_+i) = zz(i)-theta*sps_.linRootJac_.coeff(rootOut.rootDim_,i);
            }
          }
        } else { // regular collision event
          newState.y.segment(dim_,dim_) -= fac*sps_.linRootJac_.row(rootOut.rootDim_).head(dim_);
        }
      } else {
        std::cout << "lin: trajectory passing into allowed region!!!, fac = " << fac << std::endl;
        std::cout << rootOut << std::endl;
      }
    } else if(rootOut.rootType_==2){
      double fac = 2.0*sps_.spLinRootJac_.rowHeadDot(rootOut.rootDim_,oldState.y.segment(dim_,dim_))/splinJacSNorms_.coeff(rootOut.rootDim_);
      //std::cout << "fac sparse " << fac << std::endl;
      if(fac<0.0) {
        if constexpr(_BOUNDARY_KERNEL_TYPE_==1){
          double phi = 0.5*fac*sqrt(splinJacSNorms_.coeff(rootOut.rootDim_))/oldState.y.segment(dim_,dim_).norm();
          //std::cout << "phi=" << phi << std::endl;
          if(std::fabs(phi) > _EXIT_ANGLE_THRESH_){
            sps_.spLinRootJac_.scaledRowHeadIncrement(rootOut.rootDim_,-fac,newState.y.segment(dim_,dim_));
          } else {
            size_t rowNz = sps_.spLinRootJac_.rowNz(rootOut.rootDim_);
            size_t idx;
            for(size_t i=0;i<rowNz;i++){
              idx = sps_.spLinRootJac_.getColIndex(rootOut.rootDim_,i);
              newState.y.coeffRef(dim_+idx) = -newState.y.coeff(dim_+idx);
            }
          }
        } else if constexpr(_BOUNDARY_KERNEL_TYPE_==2) {
          double phi = 0.5*fac*sqrt(splinJacSNorms_.coeff(rootOut.rootDim_))/oldState.y.segment(dim_,dim_).norm();

          if(std::fabs(phi)>0.0001){
            size_t rowNz = sps_.spLinRootJac_.rowNz(rootOut.rootDim_);
            size_t idx;
            if(boundary_z_.size()!=rowNz) boundary_z_.resize(rowNz);
            r_.rnorm(boundary_z_);
            double innerProd = 0.0;
            for(size_t i=0;i<rowNz;i++){
              idx = sps_.spLinRootJac_.getColIndex(rootOut.rootDim_,i);
              innerProd += (boundary_z_.coeff(i)+oldState.y.coeff(dim_+idx))*sps_.spLinRootJac_.getValColIndex(rootOut.rootDim_,i);
            }
            innerProd /= splinJacSNorms_.coeff(rootOut.rootDim_);
            for(size_t i=0;i<rowNz;i++){
              idx = sps_.spLinRootJac_.getColIndex(rootOut.rootDim_,i);
              newState.y.coeffRef(dim_+idx) = boundary_z_.coeff(i) - innerProd*sps_.spLinRootJac_.getValColIndex(rootOut.rootDim_,i);
            }
          } else {
            std::cout << "phi : " << phi << std::endl;
            std::cout << "independent momentum refresh at boundary" << std::endl;
            size_t rowNz = sps_.spLinRootJac_.rowNz(rootOut.rootDim_);
            size_t idx;
            if(boundary_z_.size()!=rowNz) boundary_z_.resize(rowNz);
            r_.rnorm(boundary_z_);
            double innerProd = 0.0;
            for(size_t i=0;i<rowNz;i++){
              idx = sps_.spLinRootJac_.getColIndex(rootOut.rootDim_,i);
              innerProd += boundary_z_.coeff(i)*sps_.spLinRootJac_.getValColIndex(rootOut.rootDim_,i);
            }
            if(innerProd>0.0){
              for(size_t i=0;i<rowNz;i++){
                idx = sps_.spLinRootJac_.getColIndex(rootOut.rootDim_,i);
                newState.y.coeffRef(dim_+idx) = boundary_z_.coeff(i);
              }
            } else {
              for(size_t i=0;i<rowNz;i++){
                idx = sps_.spLinRootJac_.getColIndex(rootOut.rootDim_,i);
                newState.y.coeffRef(dim_+idx) = -boundary_z_.coeff(i);
              }
            }

          }
        } else {
          sps_.spLinRootJac_.scaledRowHeadIncrement(rootOut.rootDim_,-fac,newState.y.segment(dim_,dim_));
        }
      } else {
        std::cout << "sparse lin: trajectory passing into allowed region!!!, fac = " << fac << std::endl;
        std::cout << rootOut << std::endl;
      }
    } else if(rootOut.rootType_==3){
      double fac;
      if constexpr(_BOUNDARY_KERNEL_TYPE_==1){
        fac = sps_.spLinL1RootJac_[rootOut.rootDim_].splinStandardizedCollisionMomentumUpdate(
          (-(sps_.spLinL1RootJac_[rootOut.rootDim_]*oldState.y + sps_.spLinL1RootConst_[rootOut.rootDim_])).array().sign().matrix(),
          newState.y.segment(dim_,dim_),
          _EXIT_ANGLE_THRESH_);
      } else if constexpr(_BOUNDARY_KERNEL_TYPE_==2){
        fac = sps_.spLinL1RootJac_[rootOut.rootDim_].splinStandardizedRandomizedMomentumUpdate(
          r_,
          (-(sps_.spLinL1RootJac_[rootOut.rootDim_]*oldState.y + sps_.spLinL1RootConst_[rootOut.rootDim_])).array().sign().matrix(),
          newState.y.segment(dim_,dim_));
        //std::cout << "fac: " << fac << std::endl;
      } else {
        fac = sps_.spLinL1RootJac_[rootOut.rootDim_].splinStandardizedCollisionMomentumUpdate(
          (-(sps_.spLinL1RootJac_[rootOut.rootDim_]*oldState.y + sps_.spLinL1RootConst_[rootOut.rootDim_])).array().sign().matrix(),
          newState.y.segment(dim_,dim_));

      }
      if(fac>0.0){
        std::cout << "sparse lin L1: trajectory passing into allowed region!!!, fac : " << fac << std::endl;
        throw 234;
      }
    } else if(rootOut.rootType_==4){
      double fac;
      //std::cout << "boundary kernel, rootOut: " << rootOut << std::endl;
      if constexpr(_BOUNDARY_KERNEL_TYPE_==1){
        fac = sps_.spLinL2RootJac_[rootOut.rootDim_].splinStandardizedCollisionMomentumUpdate(
          (-(sps_.spLinL2RootJac_[rootOut.rootDim_]*oldState.y + sps_.spLinL2RootConst_[rootOut.rootDim_])),
          newState.y.segment(dim_,dim_),
          _EXIT_ANGLE_THRESH_);
      } else if constexpr(_BOUNDARY_KERNEL_TYPE_==2){
        fac = sps_.spLinL2RootJac_[rootOut.rootDim_].splinStandardizedRandomizedMomentumUpdate(
          r_,
          (-(sps_.spLinL2RootJac_[rootOut.rootDim_]*oldState.y + sps_.spLinL2RootConst_[rootOut.rootDim_])),
          newState.y.segment(dim_,dim_));
      } else {
        fac = sps_.spLinL2RootJac_[rootOut.rootDim_].splinStandardizedCollisionMomentumUpdate(
          (-(sps_.spLinL2RootJac_[rootOut.rootDim_]*oldState.y + sps_.spLinL2RootConst_[rootOut.rootDim_])),
          newState.y.segment(dim_,dim_));
      }
      //std::cout << "p : " << newState.y.segment(dim_,dim_).transpose() << std::endl;
      if(fac>0.0) {
        std::cout << "sparse lin L2: trajectory passing into allowed region!!!, fac = " << fac << std::endl;
      }
    } else if(rootOut.rootType_==5){
      double fac;
      if constexpr(_BOUNDARY_KERNEL_TYPE_==1){
        fac = sps_.spLinFRootJac_[rootOut.rootDim_].splinStandardizedCollisionMomentumUpdate(
          rootOut.auxInfo_,
          newState.y.segment(dim_,dim_),
          _EXIT_ANGLE_THRESH_);
      } else if constexpr(_BOUNDARY_KERNEL_TYPE_==2) {
        fac = sps_.spLinFRootJac_[rootOut.rootDim_].splinStandardizedRandomizedMomentumUpdate(
          r_,
          rootOut.auxInfo_,
          newState.y.segment(dim_,dim_));

      } else {
        fac = sps_.spLinFRootJac_[rootOut.rootDim_].splinStandardizedCollisionMomentumUpdate(
          rootOut.auxInfo_,
          newState.y.segment(dim_,dim_));
      }
      if(fac>0.0) {
        std::cout << "sparse lin Fun: trajectory passing into allowed region!!!, fac = " << fac << std::endl;
      }
    }else {
      std::cout << "root type" << rootOut.rootType_ << " not implemented yet" << std::endl;
      throw(12);
    }

    (*diag_).push("lambda",lambda_.getPars());
    (*diag_).push("L_max",TM_.LMax());
    (*diag_).push("L_min",TM_.LMin());



    return(true);
  }

  inline const specialRootSpec& spr() const {return sps_;}


  /*
   inline void specialRoots(specialRootSpec& sps) {
   if(! sps_.nonTrivial() && ci_.nonTrivialSpecial()){
   if(ci_.numLin_>0){
   sps.linRootJac_.resize(ci_.linJac_.rows(),dim());
   sps.linRootJac_.setZero();
   sps.linRootJac_.leftCols(ci_.linJac_.cols()) = ci_.linJac_;
   sps.linRootConst_ = ci_.linConst_;
   linJacSNorms_.resize(ci_.linJac_.rows());
   for(size_t i=0;i<ci_.linJac_.rows();i++) linJacSNorms_.coeffRef(i) = ci_.linJac_.row(i).squaredNorm();
   }
   sps_ = sps;
   //std::cout << "HMCProcessConstr::specialRoots:\n" << sps << std::endl;
   } else {
   sps = sps_;
   }
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
    ci_.toJSON(outf);
  }

};




#endif

