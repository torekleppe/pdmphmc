#ifndef _AMTNORMALLPDF_HPP_
#define _AMTNORMALLPDF_HPP_

#include <functional>

namespace amt{

template <class argType, class meanType, class sdType>
class normal_ld{
public:
  normal_ld(const argType& arg,
            const meanType& mean,
            const sdType& sd){}
};




template<>
class normal_ld<amtVar,double,double>{
  const amtVar* arg_;
  const double mean_;
  const double sd_;
public:
  normal_ld(const amtVar& arg,
            const double mean,
            const double sd) : arg_(&arg), mean_(mean), sd_(sd){}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    sparseVec::syr(arg_->Jac_,
                   std::pow(sd_,-2),
                   tensor);
    return(stan::math::normal_lpdf(arg_->value(),mean_,sd_));
  }
};

template <>
class normal_ld<Eigen::Matrix<amtVar,Eigen::Dynamic,1>,double,double>{
  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* arg_;
  const double mean_;
  const double sd_;
public:
  normal_ld(const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& arg,
            const double mean,
            const double sd) : arg_(&arg), mean_(mean), sd_(sd){}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    double prec = std::pow(sd_,-2);
    for(size_t i=0;i<arg_->size();i++){
      sparseVec::syr(arg_->coeff(i).Jac_,
                     prec,
                     tensor);
    }
    return(stan::math::normal_lpdf(asStanVar(*arg_),mean_,sd_));
  }
};

template<>
class normal_ld<double,amtVar,double>{
  const double arg_;
  const amtVar* mean_;
  const double sd_;
public:
  normal_ld(const double arg,
            const amtVar& mean,
            const double sd) : arg_(arg), mean_(&mean), sd_(sd){}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    sparseVec::syr(mean_->Jac_,
                   std::pow(sd_,-2),
                   tensor);
    return(stan::math::normal_lpdf(arg_,mean_->value(),sd_));
  }
};

template<>
class normal_ld<double,double,amtVar>{
  const double arg_;
  const double mean_;
  const amtVar* sd_;
public:
  normal_ld(const double arg,
            const double mean,
            const amtVar& sd) : arg_(arg), mean_(mean), sd_(&sd){}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    sparseVec::syr(sd_->Jac_,
                   2.0/stan::math::square(sd_->value()),
                   tensor);
    return(stan::math::normal_lpdf(arg_,mean_,sd_->value()));
  }
};


template <>
class normal_ld<amtVar,amtVar,double>{
  const amtVar* arg_;
  const amtVar* mean_;
  const double sd_;
public:
  normal_ld(const amtVar& arg,
            const amtVar& mean,
            const double sd) : arg_(&arg), mean_(&mean), sd_(sd){}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    double prec = std::pow(sd_,-2);
    sparseVec::sym2x2outer(arg_->Jac_,mean_->Jac_,
                           prec,-prec,prec,
                           tensor);
    return(stan::math::normal_lpdf(arg_->value(),mean_->value(),sd_));
  }
};



template <>
class normal_ld<double,amtVar,amtVar>{
  const double arg_;
  const amtVar* mean_;
  const amtVar* sd_;
public:
  normal_ld(const double arg,
            const amtVar& mean,
            const amtVar& sd) : arg_(arg), mean_(&mean), sd_(&sd){}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    stan::math::var prec = 1.0/stan::math::square(sd_->value());
    // note, this works as the basic Fisher info is diagonal
    sparseVec::syr(mean_->Jac_,
                   prec,
                   tensor);
    sparseVec::syr(sd_->Jac_,
                   2.0*prec,
                   tensor);
    return(stan::math::normal_lpdf(arg_,mean_->value(),sd_->value()));
  }
};

template <>
class normal_ld<amtVar,double,amtVar>{
  const amtVar* arg_;
  const double mean_;
  const amtVar* sd_;
public:
  normal_ld(const amtVar& arg,
            const double mean,
            const amtVar& sd) : arg_(&arg), mean_(mean), sd_(&sd){}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    stan::math::var prec = 1.0/stan::math::square(sd_->value());
    sparseVec::syr(arg_->Jac_,
                   prec,
                   tensor);
    sparseVec::syr(sd_->Jac_,
                   2.0*prec,
                   tensor);
    return(stan::math::normal_lpdf(arg_->value(),mean_,sd_->value()));
  }
};


template <>
class normal_ld<amtVar,amtVar,amtVar>{
  const amtVar* arg_;
  const amtVar* mean_;
  const amtVar* sd_;
public:
  normal_ld(const amtVar& arg,
            const amtVar& mean,
            const amtVar& sd) : arg_(&arg), mean_(&mean), sd_(&sd){}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    stan::math::var prec = 1.0/stan::math::square(sd_->value());
    sparseVec::sym2x2outer(arg_->Jac_,mean_->Jac_,
                           prec,-prec,prec,
                           tensor);
    sparseVec::syr(sd_->Jac_,
                   2.0*prec,
                   tensor);
    return(stan::math::normal_lpdf(arg_->value(),mean_->value(),sd_->value()));
  }
};


template <>
class normal_ld<Eigen::VectorXd,amtVar,double>{
  const Eigen::VectorXd* arg_;
  const amtVar* mean_;
  const double sd_;
  const double prec_;
public:
  normal_ld(const Eigen::VectorXd& arg,
            const amtVar& mean,
            const double sd) : arg_(&arg), mean_(&mean), sd_(sd), prec_(static_cast<double>(arg.size())/std::pow(sd,2)){}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    sparseVec::syr(mean_->Jac_,
                   prec_,
                   tensor);
    return(stan::math::normal_lpdf(*arg_,mean_->value(),sd_));
  }
};



template <>
class normal_ld<Eigen::VectorXd,Eigen::Matrix<amtVar,Eigen::Dynamic,1>,double>{
  const Eigen::VectorXd* arg_;
  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* mean_;
  const double sd_;
  const double prec_;
  const size_t N_;
public:
  normal_ld(const Eigen::VectorXd& arg,
            const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& mean,
            const double sd) : arg_(&arg), mean_(&mean), sd_(sd), prec_(std::pow(sd,-2)), N_(arg.size()) {
    if(arg.size() != mean.size()){
      throw std::runtime_error("normal_ld : x and mu must have the same length");
    }
  }
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    stan::math::var lpdf_=0.0;
    Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> mean_vals(N_);
    for(size_t i=0;i<N_;i++){
      mean_vals.coeffRef(i) = (*mean_).coeff(i).value();
      sparseVec::syr((*mean_).coeff(i).Jac_,
                     prec_,
                     tensor);
    }
    return(stan::math::normal_lpdf(*arg_,mean_vals,sd_));
  }
};


template <>
class normal_ld<Eigen::VectorXd,Eigen::Matrix<amtVar,Eigen::Dynamic,1>,amtVar>{
  const Eigen::VectorXd* arg_;
  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* mean_;
  const amtVar* sd_;
  const stan::math::var prec_;
  const size_t N_;
public:
  normal_ld(const Eigen::VectorXd& arg,
            const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& mean,
            const amtVar& sd) : arg_(&arg), mean_(&mean), sd_(&sd), prec_(1.0/stan::math::square(sd.value())), N_(arg.size()) {
    if(arg.size() != mean.size()){
      throw std::runtime_error("normal_ld : x and mu must have the same length");
    }
  }
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    stan::math::var lpdf_=0.0;
    Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> mean_vals(N_);
    for(size_t i=0;i<N_;i++){
      mean_vals.coeffRef(i) = (*mean_).coeff(i).value();
      sparseVec::syr((*mean_).coeff(i).Jac_,
                     prec_,
                     tensor);
    }
    sparseVec::syr(sd_->Jac_,
                   static_cast<double>(2*N_)*prec_,
                   tensor);
    return(stan::math::normal_lpdf(*arg_,mean_vals,sd_->value()));
  }
};


template <>
class normal_ld<Eigen::Matrix<amtVar,Eigen::Dynamic,1>,double,amtVar>{
  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* arg_;
  const double mean_;
  const amtVar* sd_;
public:
  normal_ld(const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& arg,
            const double mean,
            const amtVar& sd) : arg_(&arg), mean_(mean), sd_(&sd){}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    stan::math::var prec = 1.0/stan::math::square(sd_->value());


    for(size_t i=0;i<arg_->size();i++){
      sparseVec::syr(arg_->coeff(i).Jac_,
                     prec,
                     tensor);
    }

    sparseVec::syr(sd_->Jac_,
                   2.0*static_cast<double>(arg_->size())*prec,
                   tensor);

    return(stan::math::normal_lpdf(asStanVar(arg_),
                                   mean_,
                                   sd_->value()));
  }
};





/*
 * Non-amtVar functions
 *
 */





template<>
class normal_ld<stan::math::var,double,double>{
  const stan::math::var lpdf_;
public:
  normal_ld(const stan::math::var& arg,
            const double mean,
            const double sd) : lpdf_(stan::math::normal_lpdf(arg,mean,sd)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const {return(lpdf_);}
};

template<>
class normal_ld<Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>,double,double>{
  const stan::math::var lpdf_;
public:
  normal_ld(const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& arg,
            const double mean,
            const double sd) : lpdf_(stan::math::normal_lpdf(arg,mean,sd)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const {return(lpdf_);}
};


template<>
class normal_ld<double,stan::math::var,double>{
  const stan::math::var lpdf_;
public:
  normal_ld(const double arg,
            const stan::math::var& mean,
            const double sd) : lpdf_(stan::math::normal_lpdf(arg,mean,sd)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const {return(lpdf_);}
};

template<>
class normal_ld<double,double,stan::math::var>{
  const stan::math::var lpdf_;
public:
  normal_ld(const double arg,
            const double mean,
            const stan::math::var& sd) : lpdf_(stan::math::normal_lpdf(arg,mean,sd)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const {return(lpdf_);}
};

template<>
class normal_ld<double,stan::math::var,stan::math::var>{
  const stan::math::var lpdf_;
public:
  normal_ld(const double arg,
            const stan::math::var& mean,
            const stan::math::var& sd) : lpdf_(stan::math::normal_lpdf(arg,mean,sd)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const {return(lpdf_);}
};

template<>
class normal_ld<stan::math::var,double,stan::math::var>{
  const stan::math::var lpdf_;
public:
  normal_ld(const stan::math::var& arg,
            const double mean,
            const stan::math::var& sd) : lpdf_(stan::math::normal_lpdf(arg,mean,sd)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const {return(lpdf_);}
};


template<>
class normal_ld<stan::math::var,stan::math::var,double>{
  const stan::math::var lpdf_;
public:
  normal_ld(const stan::math::var& arg,
            const stan::math::var& mean,
            const  double sd) : lpdf_(stan::math::normal_lpdf(arg,mean,sd)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const {return(lpdf_);}
};


template<>
class normal_ld<stan::math::var,stan::math::var,stan::math::var>{
  const stan::math::var lpdf_;
public:
  normal_ld(const stan::math::var& arg,
            const stan::math::var& mean,
            const stan::math::var& sd) : lpdf_(stan::math::normal_lpdf(arg,mean,sd)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const {return(lpdf_);}
};


template<>
class normal_ld<Eigen::VectorXd,stan::math::var,double>{
  const stan::math::var lpdf_;
public:
  normal_ld(const Eigen::VectorXd& arg,
            const stan::math::var& mean,
            const double& sd) : lpdf_(stan::math::normal_lpdf(arg,mean,sd)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const {return(lpdf_);}
};

template<>
class normal_ld<Eigen::VectorXd,Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>,double>{
  const stan::math::var lpdf_;
public:
  normal_ld(const Eigen::VectorXd& arg,
            const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& mean,
            const double& sd) : lpdf_(stan::math::normal_lpdf(arg,mean,sd)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const {return(lpdf_);}
};

template<>
class normal_ld<Eigen::VectorXd,Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>,stan::math::var>{
  const stan::math::var lpdf_;
public:
  normal_ld(const Eigen::VectorXd& arg,
            const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& mean,
            const stan::math::var& sd) : lpdf_(stan::math::normal_lpdf(arg,mean,sd)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const {return(lpdf_);}
};

template<>
class normal_ld<Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>,double,stan::math::var>{
  const stan::math::var lpdf_;
public:
  normal_ld(const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& arg,
            const double mean,
            const stan::math::var& sd) : lpdf_(stan::math::normal_lpdf(arg,mean,sd)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const {return(lpdf_);}
};


}
#endif
