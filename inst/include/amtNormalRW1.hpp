#ifndef _AMTRW1_HPP_
#define _AMTRW1_HPP_



namespace amt {



template <class xType,class sigmaType>
inline typename amtReturnType2<xType,sigmaType>::type normalRW1_lpdf(const Eigen::Matrix<xType,Eigen::Dynamic,1>& x,
                                                                     const sigmaType& sigma){

  size_t n = x.size();
  xType ssq = 0.0;
  for(size_t t=1; t<n; t++) ssq += cmn::square(x.coeff(t)-x.coeff(t-1));
  return( -0.5*ssq/cmn::square(sigma) - static_cast<double>(n-1)*(cmn::log(sigma)+0.91893853320467274178));
}

/*
stan::math::var normalRW1_lpdf(const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& x,
                               const double sigma){

  size_t n = x.size();
  stan::math::var ssq = 0.0;
  for(size_t t=1; t<n; t++) ssq += stan::math::square(x.coeff(t)-x.coeff(t-1));
  return( -0.5*ssq/cmn::square(sigma) - static_cast<double>(n-1)*(std::log(sigma)+0.91893853320467274178));
}
stan::math::var normalRW1_lpdf(const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& x,
                               const stan::math::var& sigma){

  size_t n = x.size();
  stan::math::var ssq = 0.0;
  for(size_t t=1; t<n; t++) ssq += stan::math::square(x.coeff(t)-x.coeff(t-1));
  return( -0.5*ssq/stan::math::square(sigma) - static_cast<double>(n-1)*(stan::math::log(sigma)+0.91893853320467274178));
}
*/



template <class xType,class sigmaType>
class normalRW1_ld{
public:
  normalRW1_ld(const Eigen::Matrix<xType,Eigen::Dynamic,1>& x,
               const sigmaType& sigma){}
};

template <>
class normalRW1_ld<double,amtVar>{
  const stan::math::var lpdf_;
  const stan::math::var FI_;
  const amtVar* sigma_;
public:
  normalRW1_ld(const Eigen::VectorXd& x,
               const amtVar& sigma) : lpdf_(normalRW1_lpdf(x,sigma.value())),
               FI_(static_cast<double>(2*(x.size()-1))/stan::math::square(sigma.value())),
               sigma_(&sigma) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    sparseVec::syr(sigma_->Jac_,
                   FI_,
                   tensor);
    return(lpdf_);
  }
};


template <>
class normalRW1_ld<amtVar,double>{
  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* xPtr_;
  const double prec_;
  const size_t T_;
  const double nf_;

public:
  normalRW1_ld(const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& x,
               const double sigma) : xPtr_(&x), prec_(std::pow(sigma,-2)), T_(x.size()),
               nf_(- static_cast<double>(x.size()-1)*(std::log(sigma)+0.91893853320467274178)){
    if(T_<2) throw std::runtime_error("normalRW1_ld : x must be at least of dimension 2");
  }

  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    stan::math::var lpdf = 0.0;
    double twoprec = 2.0*prec_;
    double negaPrec = -prec_;
    sparseVec::syr(xPtr_->coeff(0).Jac_,prec_,tensor);
    for(size_t t=1;t<T_-1;t++){
      lpdf += stan::math::square(xPtr_->coeff(t).val_ - xPtr_->coeff(t-1).val_);
      sparseVec::syr(xPtr_->coeff(t).Jac_,twoprec,tensor);
      sparseVec::syr2(xPtr_->coeff(t-1).Jac_,
                      xPtr_->coeff(t).Jac_,
                      negaPrec,tensor);
    }
    lpdf += stan::math::square(xPtr_->coeff(T_-1).val_ - xPtr_->coeff(T_-2).val_);
    sparseVec::syr(xPtr_->coeff(T_-1).Jac_,prec_,tensor);
    sparseVec::syr2(xPtr_->coeff(T_-2).Jac_,
                    xPtr_->coeff(T_-1).Jac_,
                    negaPrec,tensor);
    lpdf *= -0.5*prec_;
    return(lpdf + nf_);
  }
};

template <>
class normalRW1_ld<amtVar,amtVar>{
  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* xPtr_;
  const amtVar* sigma_;
  const size_t T_;

public:
  normalRW1_ld(const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& x,
               const amtVar& sigma) : xPtr_(&x), sigma_(&sigma), T_(x.size()){
    if(T_<2) throw std::runtime_error("normalRW1_ld : x must be at least of dimension 2");
  }

  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    stan::math::var lpdf = 0.0;
    stan::math::var prec = 1.0/stan::math::square(sigma_->val_);
    stan::math::var twoprec = 2.0*prec;
    stan::math::var negaPrec = -prec;
    sparseVec::syr(xPtr_->coeff(0).Jac_,prec,tensor);
    for(size_t t=1;t<T_-1;t++){
      lpdf += stan::math::square(xPtr_->coeff(t).val_ - xPtr_->coeff(t-1).val_);
      sparseVec::syr(xPtr_->coeff(t).Jac_,twoprec,tensor);
      sparseVec::syr2(xPtr_->coeff(t-1).Jac_,
                      xPtr_->coeff(t).Jac_,
                      negaPrec,tensor);
    }
    lpdf += stan::math::square(xPtr_->coeff(T_-1).val_ - xPtr_->coeff(T_-2).val_);
    sparseVec::syr(xPtr_->coeff(T_-1).Jac_,prec,tensor);
    sparseVec::syr2(xPtr_->coeff(T_-2).Jac_,
                    xPtr_->coeff(T_-1).Jac_,
                    negaPrec,tensor);

    sparseVec::syr(sigma_->Jac_,
                   static_cast<double>(2*(T_-1))*prec,
                   tensor);


    lpdf *= -0.5*prec;
    return(lpdf - static_cast<double>(T_-1)*(log(sigma_->val_)+0.91893853320467274178));
  }
};




template <>
class normalRW1_ld<double,stan::math::var>{
  const stan::math::var lpdf_;
public:
  normalRW1_ld(const Eigen::VectorXd& x,
               const stan::math::var& sigma) : lpdf_(normalRW1_lpdf(x,sigma)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{ return(lpdf_); }
};



template <>
class normalRW1_ld<stan::math::var,double>{
  const stan::math::var lpdf_;
public:
  normalRW1_ld(const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& x,
               const double sigma) : lpdf_(normalRW1_lpdf(x,sigma)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{ return(lpdf_); }
};

template <>
class normalRW1_ld<stan::math::var,stan::math::var>{
  const stan::math::var lpdf_;
public:
  normalRW1_ld(const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& x,
               const stan::math::var& sigma) : lpdf_(normalRW1_lpdf(x,sigma)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{ return(lpdf_); }
};


} // namespace

#endif
