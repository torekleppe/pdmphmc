#ifndef AMTINVLOGITBETA_HPP
#define AMTINVLOGITBETA_HPP

#include "amtTraits.hpp"
#include "amtVar.hpp"
#include <stan/math.hpp>

namespace amt{

template <class argType, class aType, class bType>
inline typename amtReturnType3<argType,aType,bType>::type invLogitBeta_lpdf(const argType& x,
                                                                            const aType& a,
                                                                            const bType& b){
  argType bx = cmn::inv_logit(x);
  return(stan::math::beta_lpdf(bx,a,b) + cmn::log(bx-cmn::square(bx)));
}


template <class argType, class aType, class bType>
class invLogitBeta_ld{
public:
  invLogitBeta_ld(const argType& x,
                  const aType& a,
                  const bType& b){}
};


template<>
class invLogitBeta_ld<amtVar,double,double>{
  const amtVar* x_;
  const stan::math::var lpdf_;
  const double prec_;
public:
  invLogitBeta_ld(const amtVar& x,
                  const double a,
                  const double b) : x_(&x), lpdf_(invLogitBeta_lpdf(x.value(),a,b)), prec_(a*b/(a+b+1.0)){}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    sparseVec::syr(x_->Jac_,prec_,tensor);
    return(lpdf_);
  }
};

template<>
class invLogitBeta_ld<stan::math::var,double,double>{
  const stan::math::var lpdf_;
public:
  invLogitBeta_ld(const stan::math::var& x,
                  const double a,
                  const double b) : lpdf_(invLogitBeta_lpdf(x,a,b)){}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    return(lpdf_);
  }
};



} // namespace


#endif
