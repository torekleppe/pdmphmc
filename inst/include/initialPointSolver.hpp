#ifndef _INITIALPOINTSOLVER_HPP_
#define _INITIALPOINTSOLVER_HPP_
#include <iostream>
/*
 * Heavily robustified initial pdmphmc-based solver to work out intial configurations
 * Based on RKDP54 steps
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
class initialPointSolver{
  targetType* t_;
  metricTensorDummy mtd_;
  amt::amtModel<stan::math::var,metricTensorDummy,false> mdl_;
  RKDP54< initialPointSolver<targetType> > rk_;
  size_t dim_;
  size_t dimGen_;

  Eigen::VectorXd par_tmp_,grad_tmp_,y_tmp_,q_curr_;

  rng r_;
  stabilityMonitor mon_;

  double kinEnergyThresh_;

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

  void updateEps(){
    rk_.eps_ *= std::min(5.0,std::max(0.2,0.95*std::pow(rk_.stepErr_,-0.2)));
  }

  int optimize(){
    // simple steepest descent method with fairly accurate line search
    par_tmp_ = q_curr_;
    double obj0 = targetGrad();
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
      obj = targetGrad();
#ifdef _IPS_DEBUG_
      std::cout << "bisection: target : " << obj0 << std::endl;
#endif
      if(std::isfinite(obj)){
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
      obj = targetGrad();

      if(std::isfinite(obj)){
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
          return(0);
        } else {
          return(1);
        }
      }

      if(ub-lb<1.0e-3*std::max(1.0,0.5*(lb+ub))){
        std::cout << "exit: interval" << std::endl;
        if(progressMade){
          q_curr_ = lastGood;
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
    Eigen::VectorXd lastGen,firstGen;

    size_t nstep = 0;
    size_t nacc = 0;

    bool stepGood,flag;
    while(nstep <= _IPS_MAX_STEPS_PER_LEG_){
      stepGood = false;
      while(nstep <= _IPS_MAX_STEPS_PER_LEG_){
        flag = rk_.step();
        if(flag){ // no numerical problems
          if(rk_.stepErr_<1.0){
            nacc++;
            stepGood = true;
            break;
          } else {
            rk_.eps_ *= std::max(0.2,0.9*std::pow(rk_.stepErr_,-0.2));
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
/*
      if(nacc>=100){
        std::cout << "max steps, nacc = " << nacc << std::endl;
        break;
      }
*/
      // check progress
      firstGen = rk_.firstGenerated();
      lastGen = rk_.lastGenerated();



      // rescale momentum if kinetic energy is too large
      if(lastGen(1)>kinEnergyThresh_){
#ifdef _IPS_DEBUG_
        std::cout << "kinetic energy too large, nacc = " << nacc << std::endl;
#endif
        updateEps();
        eflag = 1;
        break;
      }

      if(lastGen(2)<0.0 && nacc>5){
#ifdef _IPS_DEBUG_
        std::cout << "negative potential energy time derivative , nacc = " << nacc << std::endl;
#endif
        updateEps();
        eflag = 3;
        break;
      }

      if(lastGen(3)<0.0 && nacc>2){
#ifdef _IPS_DEBUG_
        std::cout << "NUT criterion , nacc = " << nacc << std::endl;
#endif
        updateEps();
        eflag = 4;
        break;
      }




      // prepare for next integration step

      rk_.prepareNext();
      updateEps();


    } // main integration loop
    mon_.push(lastGen.coeff(0));
    return(eflag);
  }

public:
  Eigen::VectorXd bestQ(){return q_curr_;}
  void seed(const size_t s){r_.seed(s);}
  inline size_t dim() const {return 2*dim_;}
  inline size_t generatedDim() const {return 5;}
  inline size_t eventRootDim() const {return 1;}

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
                                   const Eigen::VectorXd &f) const {
    Eigen::VectorXd ret(eventRootDim());
    ret.setConstant(1.0);
    return(ret);
  }




  initialPointSolver(targetType &t, const size_t dim, const size_t dimGen) : t_(&t), mdl_(mtd_), dim_(dim), dimGen_(dimGen), mon_(6) {
    rk_.setup(*this);
    rk_.absTol_ = 1.0e-3;
    rk_.relTol_ = 1.0e-3;
    par_tmp_.resize(dim);
    grad_tmp_.resize(dim);
    y_tmp_.resize(2*dim);
    kinEnergyThresh_ = 0.5*(static_cast<double>(dim) +  _IPS_NSTDS_*sqrt(2.0*static_cast<double>(dim)));
  }



  bool run(const Eigen::VectorXd& q0){
    q_curr_ = q0;
    par_tmp_ = q0;
    targetGrad();
    double gnorm = grad_tmp_.norm();
    if(gnorm>1.0e-6){
      y_tmp_.tail(dim_) = (std::sqrt(static_cast<double>(dim_))/gnorm)*grad_tmp_;
    } else {
      r_.rnorm(y_tmp_.tail(dim_));
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

      if(mon_.hasSufficientData()){
        if(mon_.isStable_regression()){
          std::cout << "initialPointSolver done" << std::endl;
          return(true);
        }
      }

      iret = integrate();
      if(iret<0){
        // do optimization step
        optimize();
        // reset to last good point and, refresh momentum and try again
        y_tmp_.head(dim_) = q_curr_;
        r_.rnorm(y_tmp_.tail(dim_));
        rk_.eps_ =0.5;

      } else if(iret==1){
        // kinetric energy reduction
        y_tmp_ = rk_.lastState().y;
        q_curr_ = y_tmp_.head(dim_);
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
        r_.rnorm(y_tmp_.tail(dim_));

      } else {

        // regular momentum updated
        y_tmp_ = rk_.lastState().y;
        q_curr_ = y_tmp_.head(dim_);
        r_.rnorm(y_tmp_.tail(dim_));
      }
      firstEvalOK = rk_.setInitialState(odeState(y_tmp_));
    }
    return(true);
  }


};


#endif

