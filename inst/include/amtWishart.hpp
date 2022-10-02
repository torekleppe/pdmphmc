#ifndef _AMTWISHART_HPP_
#define _AMTWISHART_HPP_

#include <Eigen/Dense>
#include "amtSPDmatrix.hpp"
#include "amtNormal.hpp"
#include "amtExpGamma.hpp"
#include "amtTraits.hpp"


namespace amt{

template <class argType, class scaleType, class dfType>
inline typename amtReturnType3<argType,scaleType,dfType>::type
  wishartDiagScale_lpdf(const SPDmatrix<argType>& arg,
                        const Eigen::Matrix<scaleType,Eigen::Dynamic,1>& scaleDiag,
                        const dfType& df){
    std::size_t n = arg.dim();
    if(scaleDiag.size()!=n){
      throw std::runtime_error("wishartDiagScale_lpdf : scaleDiag must be of dimension of the argument");
    }

    dfType halfNu = 0.5*df;
    typename amtReturnType3<argType,scaleType,dfType>::type ret = 0.0;
    for(std::size_t j=0;j<n;j++){
      ret += expGamma_lpdf(arg.x_.coeff(j),halfNu-0.5*static_cast<double>(j),2.0*scaleDiag.coeff(j));
    }
    std::size_t k = n;
    for(std::size_t j=0;j<n-1;j++){
      for(std::size_t i=j+1;i<n;i++){
        ret += normal_lpdf(arg.x_.coeff(k),0.0,sqrt(scaleDiag.coeff(i)/arg.Lambda(j)));
        k++;
      }
    }
    return(ret);
  }


template <class argType, class scaleType, class dfType>
class wishartDiagScale_ld{
public:
  wishartDiagScale_ld(SPDmatrix<argType>& arg,
                      const Eigen::Matrix<scaleType,Eigen::Dynamic,1>& scaleDiag,
                      const dfType& df){
    std::cout << "wishartDiagScale_ld : this constructor should never be called!!" << std::endl;
  }
};


template <>
class wishartDiagScale_ld<amtVar,amtVar,amtVar>{
  const SPDmatrix<amtVar>* arg_;
  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* scaleDiag_;
  const amtVar* df_;
public:
  wishartDiagScale_ld(const SPDmatrix<amtVar>& arg,
                      const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& scaleDiag,
                      const amtVar& df) : arg_(&arg), scaleDiag_(&scaleDiag), df_(&df) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    stan::math::var ret = 0.0;
    amtVar nuHalf = 0.5*(*df_);
    std::size_t n = arg_->dim();
    for(std::size_t j=0;j<n;j++){
      ret += expGamma_ld(arg_->x_.coeff(j),
                        nuHalf-0.5*static_cast<double>(j),
                        2.0*scaleDiag_->coeff(j)
                           ).operator()(tensor);
    }
    std::size_t k = n;
    for(std::size_t j=0;j<n-1;j++){
      for(std::size_t i=j+1;i<n;i++){
        ret += normal_ld(arg_->x_.coeff(k),
                         0.0,
                         sqrt(scaleDiag_->coeff(i)/arg_->Lambda(j))
                           ).operator()(tensor);
        k++;
      }
    }

    return(ret);
  }

};


template <>
class wishartDiagScale_ld<amtVar,double,double>{
  const SPDmatrix<amtVar>* arg_;
  const Eigen::Matrix<double,Eigen::Dynamic,1>* scaleDiag_;
  const double df_;
public:
  wishartDiagScale_ld(const SPDmatrix<amtVar>& arg,
                      const Eigen::Matrix<double,Eigen::Dynamic,1>& scaleDiag,
                      const double df) : arg_(&arg), scaleDiag_(&scaleDiag), df_(df) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    stan::math::var ret = 0.0;
    double nuHalf = 0.5*df_;
    std::size_t n = arg_->dim();
    for(std::size_t j=0;j<n;j++){
      ret += expGamma_ld(arg_->x_.coeff(j),
                         nuHalf-0.5*static_cast<double>(j),
                         2.0*scaleDiag_->coeff(j)
      ).operator()(tensor);
    }
    std::size_t k = n;
    for(std::size_t j=0;j<n-1;j++){
      for(std::size_t i=j+1;i<n;i++){
        ret += normal_ld(arg_->x_.coeff(k),
                         0.0,
                         cmn::sqrt(scaleDiag_->coeff(i)/arg_->Lambda(j))
        ).operator()(tensor);
        k++;
      }
    }

    return(ret);
  }

};



template <>
class wishartDiagScale_ld<stan::math::var,stan::math::var,stan::math::var>{
  const stan::math::var lpdf_;
public:
  wishartDiagScale_ld(const SPDmatrix<stan::math::var>& arg,
                      const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& scaleDiag,
                      const stan::math::var& df) : lpdf_(wishartDiagScale_lpdf(arg,scaleDiag,df)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    return(lpdf_);
  }

};




template <>
class wishartDiagScale_ld<stan::math::var,double,double>{
  const stan::math::var lpdf_;
public:
  wishartDiagScale_ld(const SPDmatrix<stan::math::var>& arg,
                      const Eigen::Matrix<double,Eigen::Dynamic,1>& scaleDiag,
                      const double df) : lpdf_(wishartDiagScale_lpdf(arg,scaleDiag,df)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    return(lpdf_);
  }

};




} // namespace amt

#endif
