#ifndef _AMTPOISSON_HPP_
#define _AMTPOISSON_HPP_

namespace amt{


/*
 * Regular Poisson distributions with expectation exp(eta)
 *
 */

template <class etaType>
class poisson_log_lm{
public:
  poisson_log_lm(const int y, const etaType& eta){}
};


template <>
class poisson_log_lm<amtVar>{
  const amtVar* etaPtr_;
  const stan::math::var lpdf_;
public:
  poisson_log_lm(const int y, const amtVar& eta) : etaPtr_(&eta),
  lpdf_(stan::math::poisson_log_lpmf(y,eta.val_)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    sparseVec::syr(etaPtr_->Jac_,stan::math::exp(etaPtr_->val_),tensor);
    return(lpdf_);
  }
};

template <>
class poisson_log_lm<stan::math::var>{
  const stan::math::var lpdf_;
public:
  poisson_log_lm(const int y,
                 const stan::math::var& eta) : lpdf_(stan::math::poisson_log_lpmf(y,eta)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{return(lpdf_);}
};

/*
 * Zero-inflated Poisson distributions:
 *
 * probability p of a point-mass in 0, and (1-p) of a regular Poisson distribuion with
 * expectation exp(eta). The parameterization p=inv_logit(g) is used.
 *
 */

inline stan::math::var ziPoisson_log_core(const int y,
                                          const stan::math::var& eta,
                                          const stan::math::var& g){
  if(y==0){
    return(stan::math::log_sum_exp(-stan::math::exp(eta),g)-stan::math::log1p_exp(g));
  } else {
    return(stan::math::poisson_log_lpmf(y,eta) - stan::math::log1p_exp(g));
  }
}


template <class etaType, class gType>
class ziPoisson_log_lm{
public:
  ziPoisson_log_lm(const int y,
                   const etaType& eta,
                   const gType& g){}
};

template <>
class ziPoisson_log_lm<amtVar,amtVar>{
  const amtVar* etaPtr_;
  const amtVar* gPtr_;
  const stan::math::var lpdf_;
public:
  ziPoisson_log_lm(const int y,
                   const amtVar& eta,
                   const amtVar& g) : etaPtr_(&eta), gPtr_(&g), lpdf_(ziPoisson_log_core(y,eta.val_,g.val_)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    stan::math::var eta = etaPtr_->val_;
    stan::math::var g = gPtr_->val_;
    stan::math::var expEta = stan::math::exp(eta);
    stan::math::var expG = stan::math::exp(g);
    stan::math::var t1 = stan::math::exp(g+expEta);
    stan::math::var t2 = expG+1.0;

    sparseVec::syr(etaPtr_->Jac_,
                   (expEta + stan::math::exp(eta+g+expEta) - stan::math::exp(g+2.0*eta))/
                     ((t1+1.0)*t2),
                     tensor);
    sparseVec::syr2(etaPtr_->Jac_,
                    gPtr_->Jac_,
                    -stan::math::exp(eta-expEta+g)/(t2*(expG+stan::math::exp(-expEta))),
                    tensor);
    sparseVec::syr(gPtr_->Jac_,
                   (stan::math::exp(2.0*g+expEta)-stan::math::exp(2.0*g))/
                     (stan::math::square(t2)*(1.0+stan::math::exp(g+expEta))),
                     tensor);




    return(lpdf_);
  }
};


template <>
class ziPoisson_log_lm<stan::math::var,stan::math::var>{
  const stan::math::var lpdf_;
public:
  ziPoisson_log_lm(const int y,
                   const stan::math::var& eta,
                   const stan::math::var& g) : lpdf_(ziPoisson_log_core(y,eta,g)){}

  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{return(lpdf_);}
};




} // namespace
#endif
