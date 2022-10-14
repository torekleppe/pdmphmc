#ifndef _AMTMULTINORMALCOV_HPP_
#define _AMTMULTINORMALCOV_HPP_

namespace amt{



template <class argType, class muType, class sigmaType>
inline stan::math::var multi_normal_lpdf_StanVal(const Eigen::Matrix<argType,Eigen::Dynamic,1>& arg,
                                                 const Eigen::Matrix<muType,Eigen::Dynamic,1>& mu,
                                                 const SPDmatrix<sigmaType>& Sigma){
  if(arg.size()!=Sigma.dim()){
    throw std::runtime_error("multi_normal_lpdf_StanVal : arg incompatible with precision matrix dimension");
  } else if(mu.size()!=Sigma.dim()){
    throw std::runtime_error("multi_normal_lpdf_StanVal : mu incompatible with precision matrix dimension");
  }

  Eigen::Matrix<typename amtNumType2<argType,muType>::type,Eigen::Dynamic,1> tmp(arg.size());
  for(std::size_t j=0;j<arg.rows();j++) tmp.coeffRef(j) = asStanVar(arg.coeff(j)) - asStanVar(mu.coeff(j));

  stan::math::var ret = -0.5*Sigma.quad_form_inv_StanVal(tmp,0);

  ret += -0.5*(Sigma.logDet_StanVal() + static_cast<double>(arg.size())*1.8378770664093454836);
  return(ret);
}



template <class argType, class muType, class sigmaType, class tenPtrType>
inline void multi_normal_LGC(const Eigen::Matrix<argType,Eigen::Dynamic,1>* argPtr,
                             const Eigen::Matrix<muType,Eigen::Dynamic,1>* muPtr,
                             const SPDmatrix<sigmaType>* SigmaPtr,
                             tenPtrType tensor){
  std::size_t n = SigmaPtr->dim();
  std::size_t offset;
  packedSym<stan::math::var> prec;
  if constexpr(std::is_same_v<amtVar,argType> || std::is_same_v<amtVar,muType>){
    SigmaPtr->packedInverse_StanVal(prec,0);
    offset=1;
  } else if constexpr(std::is_same_v<amtVar,sigmaType>){
    SigmaPtr->packedInverse_StanVal(prec,1);
    offset=0;
  }

  // either arg or mu or both

  stan::math::var tmp;
  if constexpr(std::is_same_v<amtVar,argType> && std::is_same_v<amtVar,muType>){
    for(size_t j=0;j<n;j++){
      for(size_t i=0;i<j;i++){
        tmp = prec.read(i,j);
        sparseVec::syr2(argPtr->coeff(i).Jac_,
                        argPtr->coeff(j).Jac_,
                        tmp,
                        tensor);
        sparseVec::syr2(argPtr->coeff(i).Jac_,
                        muPtr->coeff(j).Jac_,
                        -tmp,
                        tensor);
        sparseVec::syr2(argPtr->coeff(j).Jac_,
                        muPtr->coeff(i).Jac_,
                        -tmp,
                        tensor);
        sparseVec::syr2(muPtr->coeff(j).Jac_,
                        muPtr->coeff(i).Jac_,
                        tmp,
                        tensor);
      }
      tmp = prec.read(j,j);
      sparseVec::syr(argPtr->coeff(j).Jac_,tmp,tensor);
      sparseVec::syr(muPtr->coeff(j).Jac_,tmp,tensor);
      sparseVec::syr2(argPtr->coeff(j).Jac_,
                      muPtr->coeff(j).Jac_,
                      -tmp,
                      tensor);

    }
  } else if constexpr(std::is_same_v<amtVar,argType>){
    for(size_t j=0;j<n;j++){
      for(size_t i=0;i<j;i++){
        sparseVec::syr2(argPtr->coeff(i).Jac_,
                        argPtr->coeff(j).Jac_,
                        prec.read(i,j),
                        tensor);
      }
      sparseVec::syr(argPtr->coeff(j).Jac_,prec.read(j,j),tensor);
    }


  } else if constexpr(std::is_same_v<amtVar,muType>){
    for(size_t j=0;j<n;j++){
      for(size_t i=0;i<j;i++){
        sparseVec::syr2(muPtr->coeff(i).Jac_,
                        muPtr->coeff(j).Jac_,
                        prec.read(i,j),
                        tensor);
      }
      sparseVec::syr(muPtr->coeff(j).Jac_,prec.read(j,j),tensor);
    }
  }

  // Sigma

  if constexpr(std::is_same_v<amtVar,sigmaType>){
    size_t kap;
    for(size_t i=0;i<n;i++){
      sparseVec::syr(SigmaPtr->x_.coeff(i).Jac_,0.5,tensor);
      if(i<n-1){
        kap = n*(i+1) - (i+1)*i/2;
        for(size_t jj=0;jj<n-i-1;jj++){
          for(size_t ii=0;ii<jj;ii++){
            sparseVec::syr2(SigmaPtr->x_.coeff(kap+ii).Jac_,
                            SigmaPtr->x_.coeff(kap+jj).Jac_,
                            SigmaPtr->Lambda_StanVal(i)*prec.read(i+ii+offset,i+jj+offset),
                            tensor);
          }
          sparseVec::syr(SigmaPtr->x_.coeff(kap+jj).Jac_,
                         SigmaPtr->Lambda_StanVal(i)*prec.read(i+jj+offset,i+jj+offset),
                         tensor);
        }
      }
    }
  }
}




template <class argType, class muType, class SigmaType>
class multi_normal_ld{
public:
  multi_normal_ld(const Eigen::Matrix<argType,Eigen::Dynamic,1>& arg,
                  const Eigen::Matrix<muType,Eigen::Dynamic,1>& mu,
                  const SPDmatrix<SigmaType>& Sigma){
    AMT_NOT_IMPLEMENTED_ERROR__CONTACT_DEVELOPER_3<argType,muType,SigmaType> dummy;
  }
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    return(0.0);
  }

};


template <>
class multi_normal_ld<amtVar,amtVar,amtVar>{
  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* argPtr_;
  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* muPtr_;
  const SPDmatrix<amtVar>* SigmaPtr_;
  stan::math::var lpdf_;
public:
  multi_normal_ld(const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& arg,
                  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& mu,
                  const SPDmatrix<amtVar>& Sigma) : argPtr_(&arg), muPtr_(&mu), SigmaPtr_(&Sigma), lpdf_(multi_normal_lpdf_StanVal(arg,mu,Sigma)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    multi_normal_LGC<amtVar,amtVar,amtVar,tenPtrType>(argPtr_,muPtr_,SigmaPtr_,tensor);
    return(lpdf_);
  }
};


template <>
class multi_normal_ld<amtVar,amtVar,double>{
  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* argPtr_;
  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* muPtr_;
  const SPDmatrix<double>* SigmaPtr_;
  stan::math::var lpdf_;
public:
  multi_normal_ld(const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& arg,
                  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& mu,
                  const SPDmatrix<double>& Sigma) : argPtr_(&arg), muPtr_(&mu), SigmaPtr_(&Sigma), lpdf_(multi_normal_lpdf_StanVal(arg,mu,Sigma)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    multi_normal_LGC<amtVar,amtVar,double,tenPtrType>(argPtr_,muPtr_,SigmaPtr_,tensor);
    return(lpdf_);
  }
};


template <>
class multi_normal_ld<double,amtVar,amtVar>{
  const Eigen::Matrix<double,Eigen::Dynamic,1>* argPtr_;
  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* muPtr_;
  const SPDmatrix<amtVar>* SigmaPtr_;
  stan::math::var lpdf_;
public:
  multi_normal_ld(const Eigen::Matrix<double,Eigen::Dynamic,1>& arg,
                  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& mu,
                  const SPDmatrix<amtVar>& Sigma) : argPtr_(&arg), muPtr_(&mu), SigmaPtr_(&Sigma), lpdf_(multi_normal_lpdf_StanVal(arg,mu,Sigma)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    multi_normal_LGC<double,amtVar,amtVar,tenPtrType>(argPtr_,muPtr_,SigmaPtr_,tensor);
    return(lpdf_);
  }
};

template <>
class multi_normal_ld<double,amtVar,double>{
  const Eigen::Matrix<double,Eigen::Dynamic,1>* argPtr_;
  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* muPtr_;
  const SPDmatrix<double>* SigmaPtr_;
  stan::math::var lpdf_;
public:
  multi_normal_ld(const Eigen::Matrix<double,Eigen::Dynamic,1>& arg,
                  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& mu,
                  const SPDmatrix<double>& Sigma) : argPtr_(&arg), muPtr_(&mu), SigmaPtr_(&Sigma), lpdf_(multi_normal_lpdf_StanVal(arg,mu,Sigma)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    multi_normal_LGC<double,amtVar,double,tenPtrType>(argPtr_,muPtr_,SigmaPtr_,tensor);
    return(lpdf_);
  }
};


template <>
class multi_normal_ld<double,double,amtVar>{
  const Eigen::Matrix<double,Eigen::Dynamic,1>* argPtr_;
  const Eigen::Matrix<double,Eigen::Dynamic,1>* muPtr_;
  const SPDmatrix<amtVar>* SigmaPtr_;
  stan::math::var lpdf_;
public:
  multi_normal_ld(const Eigen::Matrix<double,Eigen::Dynamic,1>& arg,
                  const Eigen::Matrix<double,Eigen::Dynamic,1>& mu,
                  const SPDmatrix<amtVar>& Sigma) : argPtr_(&arg), muPtr_(&mu), SigmaPtr_(&Sigma), lpdf_(multi_normal_lpdf_StanVal(arg,mu,Sigma)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    multi_normal_LGC<double,double,amtVar,tenPtrType>(argPtr_,muPtr_,SigmaPtr_,tensor);
    return(lpdf_);
  }
};



template <>
class multi_normal_ld<stan::math::var,stan::math::var,stan::math::var>{
  stan::math::var lpdf_;
public:
  multi_normal_ld(const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& arg,
                  const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& mu,
                  const SPDmatrix<stan::math::var>& Sigma) : lpdf_(multi_normal_lpdf_StanVal(arg,mu,Sigma)){}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{return(lpdf_);}
};

template <>
class multi_normal_ld<stan::math::var,stan::math::var,double>{
  stan::math::var lpdf_;
public:
  multi_normal_ld(const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& arg,
                  const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& mu,
                  const SPDmatrix<double>& Sigma) : lpdf_(multi_normal_lpdf_StanVal(arg,mu,Sigma)){}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{return(lpdf_);}
};

template <>
class multi_normal_ld<double,stan::math::var,stan::math::var>{
  stan::math::var lpdf_;
public:
  multi_normal_ld(const Eigen::Matrix<double,Eigen::Dynamic,1>& arg,
                  const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& mu,
                  const SPDmatrix<stan::math::var>& Sigma) : lpdf_(multi_normal_lpdf_StanVal(arg,mu,Sigma)){}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{return(lpdf_);}
};

template <>
class multi_normal_ld<double,stan::math::var,double>{
  stan::math::var lpdf_;
public:
  multi_normal_ld(const Eigen::Matrix<double,Eigen::Dynamic,1>& arg,
                  const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& mu,
                  const SPDmatrix<double>& Sigma) : lpdf_(multi_normal_lpdf_StanVal(arg,mu,Sigma)){}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{return(lpdf_);}
};

template <>
class multi_normal_ld<stan::math::var,double,stan::math::var>{
  stan::math::var lpdf_;
public:
  multi_normal_ld(const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& arg,
                  const Eigen::Matrix<double,Eigen::Dynamic,1>& mu,
                  const SPDmatrix<stan::math::var>& Sigma) : lpdf_(multi_normal_lpdf_StanVal(arg,mu,Sigma)){}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{return(lpdf_);}
};

template <>
class multi_normal_ld<double,double,stan::math::var>{
  stan::math::var lpdf_;
public:
  multi_normal_ld(const Eigen::Matrix<double,Eigen::Dynamic,1>& arg,
                  const Eigen::Matrix<double,Eigen::Dynamic,1>& mu,
                  const SPDmatrix<stan::math::var>& Sigma) : lpdf_(multi_normal_lpdf_StanVal(arg,mu,Sigma)){}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{return(lpdf_);}
};

} // namespace

#endif
