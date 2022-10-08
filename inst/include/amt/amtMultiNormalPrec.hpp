#ifndef _AMTMULTINORMALPREC_HPP_
#define _AMTMULTINORMALPREC_HPP_

#include "sparseVec.hpp"
#include "amtTraits.hpp"
#include <Eigen/Dense>

namespace amt{

/*
 * Fisher information of the SPDmatrix representation of the
 * precision matrix, only relevant for amtVar
 *
 */

void dumpTmp(const size_t i, const size_t j, const stan::math::var val){
  std::cout << "Element " << i << "," << j << " val : " << val.val() << std::endl;
}

template <class argType, class muType, class PType>
stan::math::var
  multi_normal_prec_lpdf_StanVal(const Eigen::Matrix<argType,Eigen::Dynamic,1>& arg,
                                 const Eigen::Matrix<muType,Eigen::Dynamic,1>& mu,
                                 const SPDmatrix<PType>& P){
    if(arg.size()!=P.dim()){
      throw std::runtime_error("multi_normal_prec_lpdf_StanVal : arg incompatible with precision matrix dimension");
    } else if(mu.size()!=P.dim()){
      throw std::runtime_error("multi_normal_prec_lpdf_StanVal : mu incompatible with precision matrix dimension");
    }

    Eigen::Matrix<typename amtReturnType2<argType,muType>::type,Eigen::Dynamic,1> tmp(arg.size());
    for(std::size_t j=0;j<arg.rows();j++) tmp.coeffRef(j) = arg.coeff(j)-mu.coeff(j);
    stan::math::var ret = -0.5*P.quad_form_StanVal(tmp);
    ret += 0.5*(P.logDet_StanVal() - static_cast<double>(arg.size())*1.8378770664093454836);
    return(ret);

  }

template <class tenPtrType>
void multi_normal_argORmuFI(const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* argORmu,
                            const SPDmatrix<amtVar>* P,
                            tenPtrType tensor){
  std::size_t n = P->dim();
  for(std::size_t j=0;j<n;j++){
    sparseVec::syr(argORmu->coeff(j).Jac_,
                   P->coeff_StanVal(j,j),
                   tensor);
    for(std::size_t i=0;i<j;i++){
      sparseVec::syr2(argORmu->coeff(j).Jac_,
                      argORmu->coeff(i).Jac_,
                      P->coeff_StanVal(i,j),
                      tensor);
    }
  }
}

template <class tenPtrType>
void multi_normal_argANDmuFI(const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* arg,
                             const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* mu,
                             const SPDmatrix<amtVar>* P,
                             tenPtrType tensor){
  std::size_t n = P->dim();
  stan::math::var Pelem;
  for(std::size_t j=0;j<n;j++){
    Pelem = P->coeff_StanVal(j,j);
    sparseVec::syr(arg->coeff(j).Jac_,
                   Pelem,
                   tensor);
    sparseVec::syr(mu->coeff(j).Jac_,
                   Pelem,
                   tensor);
    sparseVec::syr2(arg->coeff(j).Jac_,
                    mu->coeff(j).Jac_,
                    -Pelem,
                    tensor);
    for(std::size_t i=0;i<j;i++){
      Pelem = P->coeff_StanVal(i,j);
      sparseVec::syr2(arg->coeff(j).Jac_,
                      arg->coeff(i).Jac_,
                      Pelem,
                      tensor);
      sparseVec::syr2(mu->coeff(j).Jac_,
                      mu->coeff(i).Jac_,
                      Pelem,
                      tensor);
      sparseVec::syr2(arg->coeff(j).Jac_,
                      mu->coeff(i).Jac_,
                      -Pelem,
                      tensor);
      sparseVec::syr2(mu->coeff(j).Jac_,
                      arg->coeff(i).Jac_,
                      -Pelem,
                      tensor);

    }
  }
}


template <class tenPtrType>
void multi_normal_precisionFI(const SPDmatrix<amtVar>* P,
                              tenPtrType tensor,
                              const double obsFac=1.0){
  std::size_t n = P->dim();

  // lambda/lambda part
  for(std::size_t j=0;j<n;j++){
    sparseVec::syr(P->x_.coeff(j).Jac_,
                   0.5*obsFac,
                   tensor);

  }

  if(n>1){
    // V/V-part
    std::size_t xlen = (n*(n+1))/2;
    Eigen::Matrix<stan::math::var,Eigen::Dynamic,Eigen::Dynamic> Sigma(n,n);
    Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> Vp1(n),rho(n);
    Sigma.setZero();
    Sigma.coeffRef(n-1,n-1) = 1.0/(P->Lambda_StanVal(n-1));
    sparseVec::syr(P->x_.coeff(xlen-1).Jac_,
                   obsFac*Sigma.coeff(n-1,n-1)*P->Lambda_StanVal(n-2),
                   tensor);

    std::size_t kapj,kapjp,kapjpp,k,m;
    for(std::size_t j=n-2;j>0;j--){
      kapj = n*j - ((j-1)*j)/2;
      kapjp = n*(j+1) - ((j+1)*j)/2;
      kapjpp = n*(j+2) - ((j+2)*(j+1))/2;
      k = 0;

      for(std::size_t i = kapjp; i<kapjpp;i++) {
        Vp1.coeffRef(k) = P->x_.coeff(i).val_;
        k++;
      }

      rho.head(k) = Sigma.bottomRightCorner(k,k).selfadjointView<Eigen::Upper>()*Vp1.head(k);
      Sigma.block(j,j+1,1,k) = -rho.head(k).transpose();
      Sigma.coeffRef(j,j) = 1.0/(P->Lambda_StanVal(j)) + rho.head(k).dot(Vp1.head(k));



      for(std::size_t i = kapj;i<kapjp;i++){
        sparseVec::syr(P->x_.coeff(i).Jac_,
                       obsFac*P->Lambda_StanVal(j-1)*Sigma.coeff(j+i-kapj,j+i-kapj),
                       tensor);
        for(std::size_t l = i+1;l<kapjp;l++){
          sparseVec::syr2(P->x_.coeff(i).Jac_,
                          P->x_.coeff(l).Jac_,
                          obsFac*P->Lambda_StanVal(j-1)*Sigma.coeff(j+i-kapj,j+l-kapj),
                          tensor);
        }
      }
    }
  }
}


template <class argType, class muType, class precType>
class multi_normal_prec_ld{
public:
  multi_normal_prec_ld(const Eigen::Matrix<argType,Eigen::Dynamic,1>& arg,
                       const Eigen::Matrix<muType,Eigen::Dynamic,1>& mu,
                       const SPDmatrix<precType>& Prec) {}

};

template <>
class multi_normal_prec_ld<double,double,amtVar>{
  const SPDmatrix<amtVar>* Pptr_;
  const stan::math::var lpdf_;
public:
  multi_normal_prec_ld(const Eigen::Matrix<double,Eigen::Dynamic,1>& arg,
                       const Eigen::Matrix<double,Eigen::Dynamic,1>& mu,
                       const SPDmatrix<amtVar>& Prec) : Pptr_(&Prec),
                       lpdf_(multi_normal_prec_lpdf_StanVal<double,double,amtVar>(arg,mu,Prec)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    multi_normal_precisionFI(Pptr_,tensor);
    return(lpdf_);
  }
};

template <>
class multi_normal_prec_ld<amtVar,double,amtVar>{
  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* argPtr_;
  const SPDmatrix<amtVar>* Pptr_;
  const stan::math::var lpdf_;
public:
  multi_normal_prec_ld(const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& arg,
                       const Eigen::Matrix<double,Eigen::Dynamic,1>& mu,
                       const SPDmatrix<amtVar>& Prec) : argPtr_(&arg), Pptr_(&Prec),
                       lpdf_(multi_normal_prec_lpdf_StanVal<amtVar,double,amtVar>(arg,mu,Prec)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    multi_normal_argORmuFI(argPtr_,Pptr_,tensor);
    multi_normal_precisionFI(Pptr_,tensor);
    return(lpdf_);
  }
};




template <>
class multi_normal_prec_ld<amtVar,amtVar,amtVar>{
  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* argPtr_;
  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* muPtr_;
  const SPDmatrix<amtVar>* Pptr_;
  const stan::math::var lpdf_;
public:
  multi_normal_prec_ld(const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& arg,
                       const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& mu,
                       const SPDmatrix<amtVar>& Prec) : argPtr_(&arg), muPtr_(&mu), Pptr_(&Prec),
                       lpdf_(multi_normal_prec_lpdf_StanVal<amtVar,amtVar,amtVar>(arg,mu,Prec)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    multi_normal_argANDmuFI(argPtr_,muPtr_,Pptr_,tensor);
    multi_normal_precisionFI(Pptr_,tensor);
    return(lpdf_);
  }
};



template <>
class multi_normal_prec_ld<double,double,stan::math::var>{
  const stan::math::var lpdf_;
public:
  multi_normal_prec_ld(const Eigen::Ref< const Eigen::Matrix<double,Eigen::Dynamic,1> >& arg,
                       const Eigen::Matrix<double,Eigen::Dynamic,1>& mu,
                       const SPDmatrix< stan::math::var >& Prec) :
  lpdf_(multi_normal_prec_lpdf_StanVal<double,double,stan::math::var>(arg,mu,Prec)){}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    return(lpdf_);
  }
};


template <>
class multi_normal_prec_ld<stan::math::var,double,stan::math::var>{
  const stan::math::var lpdf_;
public:
  multi_normal_prec_ld(const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& arg,
                       const Eigen::Matrix<double,Eigen::Dynamic,1>& mu,
                       const SPDmatrix< stan::math::var >& Prec) :
  lpdf_(multi_normal_prec_lpdf_StanVal<stan::math::var,double,stan::math::var>(arg,mu,Prec)){}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    return(lpdf_);
  }
};

template <>
class multi_normal_prec_ld<stan::math::var,stan::math::var,stan::math::var>{
  const stan::math::var lpdf_;
public:
  multi_normal_prec_ld(const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& arg,
                       const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& mu,
                       const SPDmatrix< stan::math::var >& Prec) :
  lpdf_(multi_normal_prec_lpdf_StanVal<stan::math::var,stan::math::var,stan::math::var>(arg,mu,Prec)){}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    return(lpdf_);
  }
};


/*
 * iid variant where the argument is a matrix, and columns are taken
 * to be iid random vectors
 *
 */


template <class argType, class muType, class PType>
stan::math::var
  iid_multi_normal_prec_lpdf_StanVal(const Eigen::Matrix<argType,Eigen::Dynamic,Eigen::Dynamic>& arg,
                                 const Eigen::Matrix<muType,Eigen::Dynamic,1>& mu,
                                 const SPDmatrix<PType>& P){
    if(arg.rows()!=P.dim()){
      throw std::runtime_error("iid_multi_normal_prec_lpdf_StanVal : arg incompatible with precision matrix dimension");
    } else if(mu.size()!=P.dim()){
      throw std::runtime_error("iid_multi_normal_prec_lpdf_StanVal : mu incompatible with precision matrix dimension");
    }

    Eigen::Matrix<typename amtReturnType2<argType,muType>::type,Eigen::Dynamic,1> tmp(arg.rows());
    stan::math::var ret = 0.0;
    for(std::size_t c=0;c<arg.cols();c++){
    for(std::size_t j=0;j<arg.rows();j++) tmp.coeffRef(j) = arg.coeff(j,c)-mu.coeff(j);
      ret -= 0.5*P.quad_form_StanVal(tmp);
    }
    ret += 0.5*static_cast<double>(arg.cols())*(
      P.logDet_StanVal() - static_cast<double>(arg.rows())*1.8378770664093454836);

    return(ret);

  }

template <class tenPtrType>
void iid_multi_normal_argORmuFI(const Eigen::Matrix<amtVar,Eigen::Dynamic,Eigen::Dynamic>* argORmu,
                            const SPDmatrix<amtVar>* P,
                            tenPtrType tensor){
  std::size_t n = P->dim();
  stan::math::var Pelem;
  for(std::size_t j=0;j<n;j++){
    Pelem = P->coeff_StanVal(j,j);
    for(std::size_t c=0;c<argORmu->cols();c++){
    sparseVec::syr(argORmu->coeff(j,c).Jac_,
                   Pelem,
                   tensor);
    }
    for(std::size_t i=0;i<j;i++){
      Pelem = P->coeff_StanVal(i,j);
      for(std::size_t c=0;c<argORmu->cols();c++){
      sparseVec::syr2(argORmu->coeff(j,c).Jac_,
                      argORmu->coeff(i,c).Jac_,
                      Pelem,
                      tensor);
      }
    }
  }
}


template <class argType, class muType, class precType>
class iid_multi_normal_prec_ld{
public:
  iid_multi_normal_prec_ld(const Eigen::Matrix<argType,Eigen::Dynamic,Eigen::Dynamic>& arg,
                           const Eigen::Matrix<muType,Eigen::Dynamic,1>& mu,
                           const SPDmatrix<precType>& Prec) {}

};

template <>
class iid_multi_normal_prec_ld<amtVar,double,amtVar>{
  const stan::math::var lpdf_;
  const Eigen::Matrix<amtVar,Eigen::Dynamic,Eigen::Dynamic>* argPtr_;
  const SPDmatrix<amtVar>* Pptr_;

  public:
  iid_multi_normal_prec_ld(const Eigen::Matrix<amtVar,Eigen::Dynamic,Eigen::Dynamic>& arg,
                           const Eigen::VectorXd& mu,
                           const SPDmatrix<amtVar>& Prec) : lpdf_(iid_multi_normal_prec_lpdf_StanVal(arg,mu,Prec)),
                           argPtr_(&arg), Pptr_(&Prec)  {}

  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    iid_multi_normal_argORmuFI(argPtr_,Pptr_,tensor);
    multi_normal_precisionFI(Pptr_,tensor,static_cast<double>(argPtr_->cols()));
    return(lpdf_);
  }
};


template <>
class iid_multi_normal_prec_ld<double,double,amtVar>{
  const stan::math::var lpdf_;
  const std::size_t n_;
  const SPDmatrix<amtVar>* Pptr_;

public:
  iid_multi_normal_prec_ld(const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& arg,
                           const Eigen::VectorXd& mu,
                           const SPDmatrix<amtVar>& Prec) : lpdf_(iid_multi_normal_prec_lpdf_StanVal(arg,mu,Prec)),
                           n_(arg.cols()), Pptr_(&Prec)  {}

  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    multi_normal_precisionFI(Pptr_,tensor,static_cast<double>(n_));
    return(lpdf_);
  }
};

template <>
class iid_multi_normal_prec_ld<stan::math::var,double,stan::math::var>{
  const stan::math::var lpdf_;

public:
  iid_multi_normal_prec_ld(const Eigen::Matrix<stan::math::var,Eigen::Dynamic,Eigen::Dynamic>& arg,
                           const Eigen::VectorXd& mu,
                           const SPDmatrix<stan::math::var>& Prec) : lpdf_(iid_multi_normal_prec_lpdf_StanVal(arg,mu,Prec))  {}

  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    return(lpdf_);
  }
};

template <>
class iid_multi_normal_prec_ld<double,double,stan::math::var>{
  const stan::math::var lpdf_;

public:
  iid_multi_normal_prec_ld(const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& arg,
                           const Eigen::VectorXd& mu,
                           const SPDmatrix<stan::math::var>& Prec) : lpdf_(iid_multi_normal_prec_lpdf_StanVal(arg,mu,Prec))  {}

  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    return(lpdf_);
  }
};

} //end namespace amt
#endif
