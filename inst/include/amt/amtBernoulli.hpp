#ifndef _AMTBERNOULLI_HPP_
#define _AMTBERNOULLI_HPP_


namespace amt{

/*
 * Log of probability mass function of a Bernoulli variate y,
 * with P(y=1) = inv_logit(alpha), see Stan functions bernoulli_logit_x()
 *
 */

void __bernoulli_logit_lm_y_check(const Eigen::Matrix<int,Eigen::Dynamic,1>& y){

    if(!(y.maxCoeff()<=1 && y.minCoeff()>=0)){
      throw std::runtime_error("bernoulli_logit_lm : bad value in y");
    }

}


template <class dataType,class alphaType>
void __bernoulli_logit_lm_dimension_check(const Eigen::Matrix<dataType,Eigen::Dynamic,1>& y,
                                     const Eigen::Matrix<alphaType,Eigen::Dynamic,1>& alpha){
  if(y.size() != alpha.size()){
    throw std::runtime_error("bernoulli_logit_lm : dimensions of arguments not equal !");
  }
}


template <class dataType,class alphaType>
class bernoulli_logit_lm{
public:
  bernoulli_logit_lm(const Eigen::Matrix<dataType,Eigen::Dynamic,1>& y,
                            const Eigen::Matrix<alphaType,Eigen::Dynamic,1>& alpha){}

};

template <>
class bernoulli_logit_lm<int,amtVar>{
  const Eigen::Matrix<int,Eigen::Dynamic,1>* yPtr_;
  const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* alphaPtr_;
public:
  bernoulli_logit_lm(const Eigen::Matrix<int,Eigen::Dynamic,1>& y,
                     const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& alpha) : yPtr_(&y), alphaPtr_(&alpha) {
    __bernoulli_logit_lm_dimension_check(y,alpha);
    __bernoulli_logit_lm_y_check(y);
  }
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    stan::math::var lpdf_ = 0.0;
    stan::math::var expa,expaP1;
    for(size_t i = 0;i<yPtr_->size();i++){
      if(asDouble(alphaPtr_->coeff(i).val_)>0.0){
        expa = stan::math::exp(- alphaPtr_->coeff(i).val_);
        if(yPtr_->coeff(i)==0) lpdf_ -= alphaPtr_->coeff(i).val_;
      } else {
        expa = stan::math::exp(alphaPtr_->coeff(i).val_);
        if(yPtr_->coeff(i)==1) lpdf_ += alphaPtr_->coeff(i).val_;
      }
      expaP1 = 1.0+expa;
      lpdf_ -= stan::math::log(expaP1);
      sparseVec::syr(alphaPtr_->coeff(i).Jac_,
                     expa/stan::math::square(expaP1),
                     tensor);
    }
    return lpdf_;
  }
};

template <>
class bernoulli_logit_lm<int,stan::math::var>{
  stan::math::var lpdf_;
public:
  bernoulli_logit_lm(const Eigen::Matrix<int,Eigen::Dynamic,1>& y,
                     const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& alpha) : lpdf_(
                         stan::math::bernoulli_logit_lpmf(y,alpha)
                     ) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{return lpdf_;}
};



} //namespace
#endif
