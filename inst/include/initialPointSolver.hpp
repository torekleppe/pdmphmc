#ifndef _INITIALPOINTSOLVER_HPP_
#define _INITIALPOINTSOLVER_HPP_
#include <iostream>
/*
 * Heavily robustified initial pdmphmc-based solver to work out initial configurations
 * Based on RKBS32 steps
 */

#define _IPS_MAX_STEPS_PER_LEG_ 1000
#ifndef _IPS_MIN_EPS_
#define _IPS_MIN_EPS_ 1.0e-5
#endif
#ifndef _IPS_NSTDS_
#define _IPS_NSTDS_ 50.0
#endif

//#define _IPS_DEBUG_


template <class targetType>
class IPSode{
  targetType* t_;
  metricTensorDummy mtd_;
  amt::amtModel<stan::math::var,metricTensorDummy,false> mdl_;
  size_t dim_;
  size_t dimGen_;
  amt::constraintInfo ci_;
  specialRootSpec sps_;
  Eigen::VectorXd linJacSNorms_,splinJacSNorms_;
  //specialRootSpec sps_;
  Eigen::VectorXd par_tmp_,grad_tmp_,q_curr_;
  Eigen::VectorXd L1tmp_,L2tmp_;

  inline double targetGrad(){
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
    //mdl_.getGenerated(gen_tmp_);
    double ret_ = mdl_.getTargetDouble(); //dret_.val();
    mdl_.getTargetGradient(grad_tmp_);
    return(ret_);
  }

  inline double constraintGrad(const int whichConstr){
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
    //mdl_.getGenerated(gen_tmp_);
    double ret_ = mdl_.getNonLinConstraint(whichConstr);
    mdl_.nonLinConstraintGradient(whichConstr,grad_tmp_);
    return(ret_);
  }

  inline void specialRoots() {
    if(ci_.numLin_>0){
      sps_.linRootJac_.resize(ci_.linJac_.rows(),dim());
      sps_.linRootJac_.setZero();
      sps_.linRootJac_.leftCols(dim_) = ci_.linJac_;
      sps_.linRootConst_ = ci_.linConst_;
      linJacSNorms_.resize(ci_.linJac_.rows());
      for(size_t i=0;i<ci_.linJac_.rows();i++) linJacSNorms_.coeffRef(i) = ci_.linJac_.row(i).squaredNorm();
    }
    if(ci_.numspLin_>0){
      sps_.spLinRootJac_ = ci_.splinJac_;
      sps_.spLinRootJac_.setCols(dim());
      sps_.spLinRootConst_ = ci_.splinConst_;
      splinJacSNorms_.resize(ci_.splinJac_.rows());
      for(size_t i=0;i<ci_.splinJac_.rows();i++) splinJacSNorms_.coeffRef(i) = ci_.splinJac_.rowSquaredNorm(i);
    }

    if(ci_.splinL1Jac_.size()>0){
      for(size_t i=0;i<ci_.splinL1Jac_.size();i++){
        sps_.spLinL1RootJac_.push_back(ci_.splinL1Jac_[i]);
        sps_.spLinL1RootJac_.back().setCols(dim());
        sps_.spLinL1RootConst_.push_back(ci_.splinL1Const_[i]);
        sps_.spLinL1RootRhs_.push_back(ci_.splinL1Rhs_.coeff(i));
      }
    }

    if(ci_.splinL2Jac_.size()>0){
      for(size_t i=0;i<ci_.splinL2Jac_.size();i++){
        sps_.spLinL2RootJac_.push_back(ci_.splinL2Jac_[i]);
        sps_.spLinL2RootJac_.back().setCols(dim());
        sps_.spLinL2RootConst_.push_back(ci_.splinL2Const_[i]);
        sps_.spLinL2RootRhs_.push_back(ci_.splinL2Rhs_.coeff(i));
      }
    }

    if(ci_.splinFJac_.size()>0){
      sps_.spLinFRootFun_ = ci_.splinFfun_;
      for(size_t i=0;i<ci_.splinFJac_.size();i++){
        sps_.spLinFRootJac_.push_back(ci_.splinFJac_[i]);
        sps_.spLinFRootJac_.back().setCols(dim());
        sps_.spLinFRootConst_.push_back(ci_.splinFConst_[i]);
      }
    }
  }

public:
  IPSode(targetType& t,
         const size_t dim,
         const size_t dimGen,
         const amt::constraintInfo& ci) : t_(&t), dim_(dim), dimGen_(dimGen), ci_(ci) {
    specialRoots();
  }

  void setCurrentQ(const Eigen::VectorXd& cq){ q_curr_ = cq;}

  inline size_t generatedDim() const {return 5;}
  inline size_t eventRootDim() const {return ci_.numNonLin_;}
  inline size_t dim() const {return 2*dim_;}
  void ode(const double time,
           const Eigen::VectorXd &y, // y = (q,p,lambda)
           Eigen::VectorXd &f,
           Eigen::VectorXd &gen,
           Eigen::VectorXd &diagInt){
    // qdot
    f.head(dim_) = y.segment(dim_,dim_);
    // pdot
    par_tmp_ = y.head(dim_);
    gen.coeffRef(0) = targetGrad(); // log-density = negative potential energy
    gen.coeffRef(1) = 0.5*y.segment(dim_,dim_).squaredNorm(); // kinetic energy
    gen.coeffRef(2) = grad_tmp_.dot(y.segment(dim_,dim_)); // log-density time derivative
    gen.coeffRef(3) = (y.head(dim_)-q_curr_).dot(y.segment(dim_,dim_)); // NUT-criterion
    gen.coeffRef(4) = -gen.coeff(0) + gen.coeff(1); // hamiltonian
    f.segment(dim_,dim_) = grad_tmp_;



    if(diagInt.size()!=0) diagInt.resize(0);
  }

  inline Eigen::VectorXd eventRoot(const double time,
                                   const odeState &state,
                                   const Eigen::VectorXd &f,
                                   const bool afterOde) {
    if(!afterOde){
      par_tmp_ = state.y.head(dim_);
      targetGrad();
    }
    Eigen::VectorXd ret(eventRootDim());
    if(ret.size()>0) ret = mdl_.getNonLinConstraint();
    return(ret);
  }


  inline double targetGradient(const Eigen::VectorXd& q,
                               Eigen::VectorXd& grad){
    par_tmp_ = q;
    double ret = targetGrad();
    grad = grad_tmp_;
    return(ret);
  }

  inline double minConstraint() const {return mdl_.minConstraint();}

  bool event(const rootInfo& rootOut,
             const double time,
             const odeState &oldState,
             const Eigen::VectorXd &f,
             odeState &newState){
    oldState.copyTo(newState);
    if(rootOut.rootType_==0){
      // nonLinear root
      par_tmp_ = oldState.y.head(dim_);
      double constr = constraintGrad(rootOut.rootDim_);
      std::cout << "constraint at root " << constr << std::endl;
      double fac = 2.0*grad_tmp_.dot(oldState.y.segment(dim_,dim_))/grad_tmp_.squaredNorm();
      std::cout << "fac " << fac << std::endl;
      if(fac<0.0) {
        newState.y.segment(dim_,dim_) -= fac*grad_tmp_;
      } else {
        std::cout << "nonlin: trajectory passing into allowed region!!!" << std::endl;
      }
    } else if(rootOut.rootType_==1) {
      double fac = 2.0*ci_.linJac_.row(rootOut.rootDim_).dot(oldState.y.segment(dim_,dim_))/linJacSNorms_.coeff(rootOut.rootDim_);
      //std::cout << "fac " << fac << std::endl;
      if(fac<0.0) {
        newState.y.segment(dim_,dim_) -= fac*ci_.linJac_.row(rootOut.rootDim_);
      } else {
        std::cout << "lin: trajectory passing into allowed region!!!" << std::endl;
      }
    } else if(rootOut.rootType_==2){
      double fac = 2.0*sps_.spLinRootJac_.rowHeadDot(rootOut.rootDim_,oldState.y.segment(dim_,dim_))/splinJacSNorms_.coeff(rootOut.rootDim_);
      //std::cout << "fac sparse " << fac << std::endl;
      if(fac<0.0) {
        sps_.spLinRootJac_.scaledRowHeadIncrement(rootOut.rootDim_,-fac,newState.y.segment(dim_,dim_));
      } else {
        std::cout << "sparse lin: trajectory passing into allowed region!!!" << std::endl;
      }
    } else if(rootOut.rootType_==3){
      L1tmp_ = (-(ci_.splinL1Jac_[rootOut.rootDim_]*oldState.y.head(dim_) + ci_.splinL1Const_[rootOut.rootDim_])).array().sign().matrix();
      double fac = sps_.spLinL1RootJac_[rootOut.rootDim_].splinStandardizedCollisionMomentumUpdate(L1tmp_,newState.y.segment(dim_,dim_));
      //std::cout << "L1 norm at event : " << (sps_.spLinL1RootJac_[rootOut.rootDim_]*oldState.y + sps_.spLinL1RootConst_[rootOut.rootDim_]).dot(L1tmp_) << std::endl;
      //std::cout << "L1 fac " << fac <<  std::endl;
      if(fac>0.0)  std::cout << "sparse lin L1: trajectory passing into allowed region!!!" << std::endl;
    } else if(rootOut.rootType_==4){
      L2tmp_ = (-((ci_.splinL2Jac_[rootOut.rootDim_]*oldState.y.head(dim_) + ci_.splinL2Const_[rootOut.rootDim_])));
      double fac = sps_.spLinL2RootJac_[rootOut.rootDim_].splinStandardizedCollisionMomentumUpdate(L2tmp_,newState.y.segment(dim_,dim_));
      //std::cout << "L2 norm at event : " << sqrt(L2tmp_.squaredNorm()) << std::endl;
      //std::cout << "L2 fac " << fac <<  std::endl;
      if(fac>0.0)  std::cout << "sparse lin L2: trajectory passing into allowed region!!!" << std::endl;
    } else if(rootOut.rootType_==5){
      //std::cout << "event : " << rootOut.auxInfo_ << std::endl;
      double fac = sps_.spLinFRootJac_[rootOut.rootDim_].splinStandardizedCollisionMomentumUpdate(rootOut.auxInfo_,newState.y.segment(dim_,dim_));
      //std::cout << "event: fac: " << fac << std::endl;
      if(fac>0.0)  std::cout << "sparse lin fun: trajectory passing into allowed region!!!" << std::endl;


    } else {
      std::cout << "special roots type " << rootOut.rootType_ << " not implemented in IPSode" << std::endl;
      throw(12);
    }

    return(true);
  }


  inline const specialRootSpec& spr() const {return sps_;}
};



template <class targetType>
class initialPointSolver{
  //targetType* t_;
  //metricTensorDummy mtd_;
  //amt::amtModel<stan::math::var,metricTensorDummy,false> mdl_;

  IPSode<targetType> ode_;
  RKBS32< IPSode<targetType> > rk_;
  size_t dim_;
  size_t dimGen_;
  //constraintInfo ci_;

  Eigen::VectorXd par_tmp_,grad_tmp_,y_tmp_,q_curr_,rootState_;

  rng r_;
  stabilityMonitor mon_;

  double kinEnergyThresh_;

  //specialRootSpec sps_;


  void updateEps(){
    rk_.eps_ *= std::min(5.0,std::max(0.2,0.95*std::pow(rk_.stepErr_,-0.333)));
  }

  int optimize(){
    // simple steepest descent method with fairly accurate line search
    par_tmp_ = q_curr_;
    double obj0 = ode_.targetGradient(par_tmp_,grad_tmp_);
#ifdef _IPS_DEBUG_
    std::cout << "optimize: target : " << obj0 << std::endl;
#endif
    Eigen::VectorXd sdir = (1.0/std::max(grad_tmp_.norm(),1.0))*grad_tmp_;
    Eigen::VectorXd lastGood = q_curr_;

    double a = 1.0;
    double anext;
    double dirder=0.0;
    double lb=0.0;
    double obj;
    double ub;
    double lf = grad_tmp_.dot(sdir);
    double uf;
    bool successful = false;
    bool progressMade = false;
    // bracket

    for(size_t i=1;i<=50;i++){
      par_tmp_ = q_curr_+a*sdir;
      obj = ode_.targetGradient(par_tmp_,grad_tmp_);
#ifdef _IPS_DEBUG_
      std::cout << "bisection: target : " << obj0 << "\t minConstraint: " << mdl_.minConstraint() << std::endl;
#endif
      if(std::isfinite(obj) && ode_.minConstraint()>-1.0e-4){
        if(obj>obj0){
          progressMade = true;
          lastGood = par_tmp_;
          obj0 = obj;
        }
        dirder = grad_tmp_.dot(sdir);

        if(dirder<0.0){
#ifdef _IPS_DEBUG_
          std::cout << "dirder = " << dirder << std::endl;
          std::cout << "negative directional derivative, exiting bracket method" << std::endl;
#endif
          successful = true;
          ub = a;
          uf = dirder;
          break;
        } else {
          lb = a;
          lf = dirder;
          a*=2.0;
        }
      } else { // if numerical problems were encountered, try a shorter step
        a=0.5*(lb+a);
      }
    }

    if(! successful || uf*lf>=0.0){
      std::cout << "bracketing failed" << std::endl;
      if(progressMade){
        q_curr_ = lastGood;
        ode_.setCurrentQ(q_curr_);
        return(0);
      } else {
        return(1);
      }
    }
#ifdef _IPS_DEBUG_
    std::cout << "lb = " << lb << " , ub = " << ub << std::endl;
    std::cout << "lf = " << lf << " , uf = " << uf << std::endl;
#endif
    // bisection search
    double safel,safeu;

    for(size_t i=0;i<30;i++){

      if(i % 2 != 0){

        // safeguarded linear interpolation based
        safel = lb + 1.0e-4*(ub-lb);
        safeu = ub - 1.0e-4*(ub-lb);
        a = std::min(safeu,std::max(safel,- (lb*uf - lf*ub)/(lf-uf)));
      } else {
        // halving as a safeguard in problematic cases
        a = 0.5*(ub+lb);
      }
      par_tmp_ = q_curr_+a*sdir;
      obj = ode_.targetGradient(par_tmp_,grad_tmp_);

      if(std::isfinite(obj) && ode_.minConstraint()>-1.0e-3){
        if(obj>obj0){
          progressMade = true;
          lastGood = par_tmp_;
          obj0 = obj;
        }
        dirder = grad_tmp_.dot(sdir);

        if(std::fabs(dirder)<1.0e-3*std::max(std::fabs(obj),1.0)){
#ifdef _IPS_DEBUG_
          std::cout << "exit: abs(directional derivative)" << std::endl;
#endif
          if(progressMade){
            q_curr_ = lastGood;
            ode_.setCurrentQ(q_curr_);
            return(0);
          } else {
            return(1);
          }
        }
        if(dirder<0.0){
          ub = a;
          uf = dirder;
        } else {
          lb = a;
          lf = dirder;
        }
      } else {
        std::cout << "bad function eval in bisection, exiting" << std::endl;
        if(progressMade){
          q_curr_ = lastGood;
          ode_.setCurrentQ(q_curr_);
          return(0);
        } else {
          return(1);
        }
      }

      if(ub-lb<1.0e-3*std::max(1.0,0.5*(lb+ub))){
        std::cout << "exit: interval" << std::endl;
        if(progressMade){
          q_curr_ = lastGood;
          ode_.setCurrentQ(q_curr_);
          return(0);
        } else {
          return(1);
        }
      }
#ifdef _IPS_DEBUG_
      std::cout << "iteration # " << i+1 << std::endl;
      std::cout << "lb = " << lb << " , ub = " << ub << std::endl;
      std::cout << "lf = " << lf << " , uf = " << uf << std::endl;
#endif
    } // bisection loop
    // bisection loop ended before regular exit
    if(progressMade){
      q_curr_ = lastGood;
      ode_.setCurrentQ(q_curr_);
      return(0);
    } else {
      return(1);
    }
  }


  int integrate(){
    int eflag = 0;
#ifdef _IPS_DEBUG_
    std::cout << "initial generated \n " << rk_.firstGenerated() << std::endl;
#endif
    Eigen::VectorXd lastGen(ode_.generatedDim()),firstGen;

    size_t nstep = 0;
    size_t nacc = 0;
    rootInfo oldRoot;
    bool stepGood,flag,noRoot;
    while(nstep <= _IPS_MAX_STEPS_PER_LEG_){
      stepGood = false;
      while(nstep <= _IPS_MAX_STEPS_PER_LEG_){
        flag = rk_.step();
        nstep++;
        if(flag){ // no numerical problems
          if(rk_.stepErr_<1.0){
            nacc++;
            stepGood = true;
            break;
          } else {
            rk_.eps_ *= std::max(0.2,0.9*std::pow(rk_.stepErr_,-0.333));
          }
        } else { // current step returned NaNs
          rk_.eps_ *= 0.1;
        }
        // cut short integration if epsilon is too small
        if(rk_.eps_ < _IPS_MIN_EPS_){
          std::cout << "ips not making progress: eps = " << rk_.eps_ << std::endl;
          rk_.eps_ = 1.0;
          return(-1);
        }
      } // step trials loop
      if(! stepGood){
        std::cout << "exhausted integration steps, nacc = " << nacc << std::endl;
        eflag = 2;
        break;
      }

      //std::cout << "step accepted, eps = " << rk_.eps_ << std::endl;



      // check if collisions occurred
      rootInfo ri = rk_.eventRootSolver(oldRoot);
      //std::cout << ri << std::endl;
      oldRoot = ri;

      //throw(1);


      firstGen = rk_.firstGenerated();
      rk_.denseGenerated_Level(ri.rootTime_,lastGen);

      // do not stop short if an event occurred
      noRoot = ri.rootDim_<0;


      // check progress
      // rescale momentum if kinetic energy is too large
      if(noRoot && lastGen(1)>kinEnergyThresh_){
#ifdef _IPS_DEBUG_
        std::cout << "kinetic energy too large, nacc = " << nacc << std::endl;
#endif
        updateEps();
        eflag = 1;
        break;
      }

      if(noRoot && lastGen(2)<0.0 && nacc>5){
#ifdef _IPS_DEBUG_
        std::cout << "negative potential energy time derivative , nacc = " << nacc << std::endl;
#endif
        updateEps();
        eflag = 3;
        break;
      }

      if(noRoot && lastGen(3)<0.0 && nacc>2){
#ifdef _IPS_DEBUG_
        std::cout << "NUT criterion , nacc = " << nacc << std::endl;
#endif
        updateEps();
        eflag = 4;
        break;
      }




      // prepare for next integration step

      if(ri.rootDim_>=0){
        rk_.event(ri);
      }



      rk_.prepareNext();

      //std::cout << "first event root\n" << rk_.firstEventRoot() << std::endl;

      updateEps();


    } // main integration loop
    mon_.push(lastGen.coeff(0));
    return(eflag);
  }

public:
  Eigen::VectorXd bestQ(){return q_curr_;}
  void seed(const size_t s){r_.seed(s);}




  initialPointSolver(targetType &t,
                     const size_t dim,
                     const size_t dimGen,
                     const amt::constraintInfo& ci) : ode_(t,dim,dimGen,ci), dim_(dim),dimGen_(dimGen), mon_(6) {

    rk_.setup(ode_);
    rk_.absTol_ = 1.0e-3;
    rk_.relTol_ = 1.0e-3;
    par_tmp_.resize(dim);
    grad_tmp_.resize(dim);
    y_tmp_.resize(ode_.dim());
    y_tmp_.setZero();
    kinEnergyThresh_ = 0.5*(static_cast<double>(dim) +  _IPS_NSTDS_*sqrt(2.0*static_cast<double>(dim)));
    //std::cout << "IPS constructor done" << std::endl;
    //ode_.specialRoots(sps_);
  }



  bool run(const Eigen::VectorXd& q0){
    q_curr_ = q0;
    ode_.setCurrentQ(q_curr_);
    par_tmp_ = q0;
    ode_.targetGradient(par_tmp_,grad_tmp_);
    double gnorm = grad_tmp_.norm();
    if(gnorm>1.0e-6){
      y_tmp_.segment(dim_,dim_) = (std::sqrt(static_cast<double>(dim_))/gnorm)*grad_tmp_;
    } else {
      r_.rnorm(y_tmp_.segment(dim_,dim_));
    }
    y_tmp_.head(dim_) = q0;
    bool firstEvalOK = rk_.setInitialState(odeState(y_tmp_));
    if(! firstEvalOK){
      std::cout << "first gradient evaluation failed; please provide a different initial point" << std::endl;
      return(false);
    }


    int iret;
    int oret;

    for(size_t l=0;l<200; l++){
      //std::cout << "l : " << l << std::endl;
      if(mon_.hasSufficientData()){
        if(mon_.isStable_regression()){
          //std::cout << "initialPointSolver done" << std::endl;
          return(true);
        }
      }
      //std::cout << "integrate:" << std::endl;
      iret = integrate();
      //std::cout << "integrate exit flag: " << iret << std::endl;
      if(iret<0){
        // do optimization step
        //std::cout << "optimize:" << std::endl;
        optimize();
        // reset to last good point and, refresh momentum and try again
        y_tmp_.head(dim_) = q_curr_;
        r_.rnorm(y_tmp_.tail(dim_));
        rk_.eps_ =0.5;

      } else if(iret==1){
        // kinetric energy reduction
        y_tmp_ = rk_.lastState().y;
        q_curr_ = y_tmp_.head(dim_);
        ode_.setCurrentQ(q_curr_);
        oret = optimize();

        if(oret==0){
          y_tmp_.head(dim_) = q_curr_;
          gnorm = grad_tmp_.norm();
          if(gnorm>1.0e-6){
            y_tmp_.tail(dim_) = (std::sqrt(static_cast<double>(dim_))/gnorm)*grad_tmp_;
          } else {
            r_.rnorm(y_tmp_.tail(dim_));
          }
        } else {
          r_.rnorm(y_tmp_.tail(dim_));
        }
        rk_.eps_ = 0.5;





      } else if(iret==2){
        y_tmp_ = rk_.firstState().y; // note first state as last step was not done
        q_curr_ = y_tmp_.head(dim_);
        ode_.setCurrentQ(q_curr_);
        r_.rnorm(y_tmp_.tail(dim_));

      } else {

        // regular momentum updated
        y_tmp_ = rk_.lastState().y;
        q_curr_ = y_tmp_.head(dim_);
        ode_.setCurrentQ(q_curr_);
        r_.rnorm(y_tmp_.tail(dim_));
      }
      firstEvalOK = rk_.setInitialState(odeState(y_tmp_));
    }
    return(true);
  }


};


#endif

