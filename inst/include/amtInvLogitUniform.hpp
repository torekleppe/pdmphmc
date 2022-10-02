#ifndef _AMTINVLOGITUNIFORM_HPP_
#define _AMTINVLOGITUNIFORM_HPP_

#include <stan/math.hpp>
#include "amtTraits.hpp"
#include "amtVar.hpp"

namespace amt{

template <class argType>
inline argType invLogitUniform_lpdf(const argType& x){
  argType tmp = (asDouble(x)>0.0) ? -x : x;
  return(tmp - 2.0*log(1.0+exp(tmp)));
}

template <class argType>
class invLogitUniform_ld{
public:
  invLogitUniform_ld(const argType& x){}
};

template <>
class invLogitUniform_ld<amtVar>{
  const amtVar* x_;
  const stan::math::var lpdf_;
public:
  invLogitUniform_ld(const amtVar& x) : x_(&x), lpdf_(invLogitUniform_lpdf(x.value())) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    sparseVec::syr(x_->Jac_,1.0/3.0,tensor);
    return(lpdf_);
  }
};

template <>
class invLogitUniform_ld<stan::math::var>{
  const stan::math::var lpdf_;
public:
  invLogitUniform_ld(const stan::math::var& x) : lpdf_(invLogitUniform_lpdf(x)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    return(lpdf_);
  }
};




}

#endif
