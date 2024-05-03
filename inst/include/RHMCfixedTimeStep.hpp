#ifndef _RHMCFIXEDTIMESTEP_HPP_
#define _RHMCFIXEDTIMESTEP_HPP_

template <class targetType>
class RHMCfixedTimeStep{
  targetType* t_;
  std::size_t dim_;
  std::size_t dimGen_;

  Eigen::VectorXd center_;
  Eigen::VectorXd scale_;

  Eigen::VectorXd par_tmp_,grad_tmp_,gen_tmp_;

  amt::amtModel<stan::math::var,metricTensorDummy,false> mdl_;

  rng r_;

  Eigen::VectorXd toPar(const Eigen::VectorXd& q){return center_ + scale_.cwiseProduct(q);}

  double evalGrad(const Eigen::VectorXd& q){
    par_tmp_ = toPar(q);
    mdl_.setIndependent(par_tmp_,dimGen_);
    try{
      t_->operator()(mdl_);
    }
    catch(...){
      stan::math::recover_memory();
      grad_tmp_.setConstant(std::numeric_limits<double>::quiet_NaN());
      return std::numeric_limits<double>::quiet_NaN();
    }
    mdl_.getGenerated(gen_tmp_);
    double ret = mdl_.getTargetDouble(); //dret_.val();
    mdl_.getTargetGradient(grad_tmp_);
    grad_tmp_ = grad_tmp_.cwiseProduct(scale_);
    return(ret);
  }

// cubic Hermite interpolation for q

  Eigen::VectorXd Q_poly(const double t, const double t_left, const double eps){
    Eigen::VectorXd ret(4);
    double t1 = (t-t_left)/eps;
    double t2 = t1*t1;
    double t3 = t1*t2;
    ret(0) = 1.0 - 3.0*t2 + 2.0*t3; // q_old
    ret(1) = eps*(t1 - 2.0*t2 + t3); // v_old
    ret(2) = 3.0*t2 - 2.0*t3; // q_new
    ret(3) = eps*(-t2+t3); // v_new
    return(ret);
  }

public:

  RHMCfixedTimeStep(targetType& target,
                    const std::size_t dim,
                    const std::size_t dimGen) : t_(&target), dim_(dim), dimGen_(dimGen){
    center_.resize(dim);
    center_.setZero();
    scale_.resize(dim);
    scale_.setOnes();
  }

  Eigen::MatrixXd samples_;

  void seed(const int seed){r_.seed(seed);}
  template<int stepType>
  void run(const Eigen::VectorXd& q0,
           const int nsample,
           const double Tmax,
           const double lambda,
           const double fixedEps){

    samples_.resize(nsample+1,dim_);
    samples_.setZero();
    double f = evalGrad(q0);
    samples_.row(0) = toPar(q0);




    double t = 0.0;
    double t_right;
    double sampleSpacing = Tmax/static_cast<double>(nsample);
    double nextSample = sampleSpacing;
    int sampleCount = 1;
    double nextEvent = -log(r_.runif())/lambda;
    double eps;

    Eigen::VectorXd q = q0;
    Eigen::VectorXd v(dim_);
    Eigen::VectorXd vh,oldq,oldv,qpoly;
    Eigen::VectorXd grad_grid_;
    grad_grid_ = grad_tmp_;
    double U;
    //r_.rnorm(v);
    v(0) = -1.0;
    v(1) = 0.5;


    while(t<Tmax-1.0e-14){ // main integration loop
      std::cout << "outer loop" << std::endl;
      while(t<nextEvent-1.0e-14){
        eps = std::min(fixedEps,nextEvent-t);
        t_right = t + eps;
        //std::cout << "t: " << t << " t_right: " << t_right << " nextEvent : " << nextEvent << std::endl;
        oldq = q;
        oldv = v;
        if constexpr(stepType==0){
          // leap frog integrator
          vh = v + 0.5*eps*grad_tmp_;
          q = q + eps*vh;
          evalGrad(q);
          v = vh + 0.5*eps*grad_tmp_;
        } else if constexpr(stepType==1) {
          // Bou-Rabee / Marsden
          U = r_.runif();
          evalGrad(q + (eps*U)*v);
          q = q + eps*v + (0.5*eps*eps)*grad_tmp_;
          v = v + eps*grad_tmp_;
        } else if constexpr(stepType==2) {
          // Bou-Rabee / Kleppe
          U = sqrt(r_.runif());
          evalGrad(q + (eps*U)*v + (0.5*std::pow(eps*U,2))*grad_grid_);
          q = q + eps*v + 0.5*(eps*eps)*grad_grid_ + ((eps*eps)/(6.0*U))*(grad_tmp_-grad_grid_);
          v = v + eps*grad_grid_ + (eps/(2.0*U))*(grad_tmp_-grad_grid_);
          evalGrad(q);
          grad_grid_ = grad_tmp_;
        } else {
          std::cout << "bad step type" << std::endl;
          throw 100;
        }

        // work out if any samples are to be made

        while(nextSample<=t_right){
          std::cout << "sample time: " << nextSample << std::endl;
          qpoly = Q_poly(nextSample,t,eps);
          samples_.row(sampleCount) = toPar(qpoly(0)*oldq + qpoly(1)*oldv + qpoly(2)*q + qpoly(3)*v);
          sampleCount++;
          nextSample = static_cast<double>(sampleCount)*sampleSpacing;

        }
        // prepare for next step
        t = t_right;
        if(t>=Tmax) break;
      } // done between events integration leg
      r_.rnorm(v); // resample momentum
      nextEvent = t + (-log(r_.runif()))/lambda; // new event time
    } // events loop





  }


};


#endif

