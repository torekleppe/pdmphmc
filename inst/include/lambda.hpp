#ifndef _LAMBDA_HPP_
#define _LAMBDA_HPP_


#include <Eigen/Dense>




class constantLambda{
  double beta_;
  double fac_;
  bool adapt_on_;
  
public:
  constantLambda() : beta_(1.0), fac_(2.0), adapt_on_(true) {}
  inline int numPars() const { return 3;}
  void setPars(Eigen::VectorXd pars){
    if(pars(0)>1.0e-14){
      beta_ = pars(0);
    }
    
    if(pars(1)>1.0e-14){
      fac_ = pars(1);
    }
    
    if(pars(2)>1.0e-14){
      adapt_on_ = true;
    } else if(std::abs(pars(2))<1.0e-14){
      adapt_on_ = false;
    }
  }
  
  Eigen::VectorXd getPars() const {
    Eigen::VectorXd pars(numPars());
    pars(0) = beta_;
    pars(1) = fac_;
    pars(2) = static_cast<double>(adapt_on_);
    return(pars);
  }
  
  template <class massMatrix_type>
  double operator()(const Eigen::VectorXd &q,
                  const Eigen::VectorXd &p,
                  const Eigen::VectorXd &qdot,
                  const Eigen::VectorXd &pdot,
                  massMatrix_type &mass) const {
    return(1.0/beta_);
  }
  
  template <class massMatrix_type>
  void momentumUpdate(const Eigen::VectorXd &q,
                      Eigen::VectorXd &p,
                      const Eigen::VectorXd &qdot,
                      const Eigen::VectorXd &pdot,
                      massMatrix_type &mass,
                      rng &r){
    r.rnorm(p);
    mass.sqrtM(p);
  }
  
  inline bool acceptsNUTAdapt() const {return adapt_on_;}
  void pushNUTAdaptInfo(const double nutTime,
                        const double nutLam){
    
    double propnext = _LAMBDA_EMA_ALPHA_*log(nutTime) + (1.0-_LAMBDA_EMA_ALPHA_)*log(beta_/fac_);
    beta_ = fmin(2.0*beta_,fmax(0.1*beta_,fac_*exp(propnext)));  
  }
};







#endif




