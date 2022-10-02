#ifndef _AMTNORMALAR1_HPP_
#define _AMTNORMALAR1_HPP_

#include "amtVar.hpp"

namespace amt{

/*
 * Stationary Gaussian AR(1) process
 * x_{t+1} = \mu + \phi*(x_t-\mu) + N(0,sigma^2)
 * mu real
 * -1<phi<1
 * sigma>0
 */

template <class argType, class muType, class phiType, class sigmaType>
inline typename amtReturnType4<argType,muType,phiType,sigmaType>::type
  normalAR1_lpdf(const Eigen::Matrix<argType,Eigen::Dynamic,1>& x,
                 const muType& mu,
                 const phiType& phi,
                 const sigmaType& sigma){

    if(x.size()<2) throw std::runtime_error("normalAR1_ld : x must be at least of dimension 2");
    // x_1:
    typename amtReturnType4<argType,muType,phiType,sigmaType>::type lpdf_ = stan::math::normal_lpdf(x.coeff(0),mu,sigma/cmn::sqrt(1.0-cmn::square(phi)));
    // x_2...X_T
    sigmaType fac = -0.5/cmn::square(sigma);
    for(size_t t=1;t<x.size();t++){
      lpdf_ += fac*cmn::square(x.coeff(t)-(mu+phi*(x.coeff(t-1)-mu)));
    }
    // normalization factor
    return(lpdf_ - static_cast<double>(x.size()-1)*(cmn::log(sigma) + 0.918938533204672745));
  }


template <class argType, class muType, class phiType, class sigmaType>
class normalAR1_ld{
public:
  normalAR1_ld(const Eigen::Matrix<argType,Eigen::Dynamic,1>& arg,
               const muType& mu,
               const phiType& phi,
               const sigmaType& sigma){
    std::cout << "normalAR1_ld : this constructor should never be called!!" << std::endl;
  }
};

template <>
class normalAR1_ld<amtVar,double,double,double>{
  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* arg_;
  const double mu_;
  const double phi_;
  const double sigma_;
  const size_t T_;
public:
  normalAR1_ld(const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& arg,
               const double mu,
               const double phi,
               const double sigma) : arg_(&arg), mu_(mu), phi_(phi), sigma_(sigma), T_(arg.size()) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> darg(T_);
    double prec = 1.0/stan::math::square(sigma_);
    double od = -prec*phi_;
    double dd = prec*(1.0 + std::pow(phi_,2));
    sparseVec::syr(arg_->coeff(0).Jac_,
                   prec,
                   tensor);
    darg.coeffRef(0) = arg_->coeff(0).val_;
    for(size_t t=1; t<T_-1;t++){
      sparseVec::syr(arg_->coeff(t).Jac_,
                     dd,
                     tensor);
      sparseVec::syr2(arg_->coeff(t-1).Jac_,
                      arg_->coeff(t).Jac_,
                      od,
                      tensor);
      darg.coeffRef(t) = arg_->coeff(t).val_;
    }
    sparseVec::syr(arg_->coeff(T_-1).Jac_,
                   prec,
                   tensor);
    sparseVec::syr2(arg_->coeff(T_-2).Jac_,
                    arg_->coeff(T_-1).Jac_,
                    od,
                    tensor);
    darg.coeffRef(T_-1) = arg_->coeff(T_-1).val_;
    return(normalAR1_lpdf(darg,mu_,phi_,sigma_));
  }
};


template <>
class normalAR1_ld<amtVar,double,double,amtVar>{
  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* arg_;
  const double mu_;
  const double phi_;
  const amtVar* sigma_;
  const size_t T_;
public:
  normalAR1_ld(const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& arg,
               const double mu,
               const double phi,
               const amtVar& sigma) : arg_(&arg), mu_(mu), phi_(phi), sigma_(&sigma), T_(arg.size()) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> darg(T_);
    stan::math::var prec = 1.0/stan::math::square(sigma_->val_);
    stan::math::var od = -prec*phi_;
    stan::math::var dd = prec*(1.0 + stan::math::square(phi_));
    sparseVec::syr(arg_->coeff(0).Jac_,
                   prec,
                   tensor);
    darg.coeffRef(0) = arg_->coeff(0).val_;
    for(size_t t=1; t<T_-1;t++){
      sparseVec::syr(arg_->coeff(t).Jac_,
                     dd,
                     tensor);
      sparseVec::syr2(arg_->coeff(t-1).Jac_,
                      arg_->coeff(t).Jac_,
                      od,
                      tensor);
      darg.coeffRef(t) = arg_->coeff(t).val_;
    }
    sparseVec::syr(arg_->coeff(T_-1).Jac_,
                   prec,
                   tensor);
    sparseVec::syr2(arg_->coeff(T_-2).Jac_,
                    arg_->coeff(T_-1).Jac_,
                    od,
                    tensor);
    darg.coeffRef(T_-1) = arg_->coeff(T_-1).val_;

    sparseVec::syr(sigma_->Jac_,
                   static_cast<double>(2*T_)*prec,
                   tensor);

    return(normalAR1_lpdf(darg,mu_,phi_,sigma_->val_));
  }
};



template <>
class normalAR1_ld<stan::math::var,double,double,double>{
  const stan::math::var lpdf_;
public:
  normalAR1_ld(const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& arg,
               const double mu,
               const double phi,
               const double sigma) : lpdf_(normalAR1_lpdf(arg,mu,phi,sigma)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{return(lpdf_);}
};

template <>
class normalAR1_ld<stan::math::var,double,double,stan::math::var>{
  const stan::math::var lpdf_;
public:
  normalAR1_ld(const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& arg,
               const double mu,
               const double phi,
               const stan::math::var& sigma) : lpdf_(normalAR1_lpdf(arg,mu,phi,sigma)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{return(lpdf_);}
};



} // namespace

#endif
