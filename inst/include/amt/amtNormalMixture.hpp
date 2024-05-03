#ifndef _AMTNORMALMIXTURE_HPP_
#define _AMTNORMALMIXTURE_HPP_

namespace amt{


/*
 * Normal mixture
 *
 * \sum_j w_j N(x_i|mu_j,sd_j)
 */


// handcoded gradient of univariate normal mixture using vectorized double numerics
double normalMixture_core(const Eigen::VectorXd& arg,
                          const Eigen::VectorXd& mean,
                          const Eigen::VectorXd& sd,
                          const Eigen::VectorXd& wts,
                          Eigen::VectorXd& dmean,
                          Eigen::VectorXd& dsd,
                          Eigen::VectorXd& dwts){

  std::size_t n = arg.size();
  std::size_t nc = mean.size();
  if(nc != sd.size() || nc != wts.size()){
    throw std::runtime_error("normalMixture_core : mean, sd and wts must have same size");
  }

  Eigen::MatrixXd lps(n,nc);
  Eigen::MatrixXd dlpm(n,nc);
  Eigen::MatrixXd dlps(n,nc);
  Eigen::VectorXd dev(n),sdev(n);
  Eigen::VectorXd swts = (wts.array()+1.0e-14).matrix();
  Eigen::VectorXd logwts = log(swts.array()).matrix();
  Eigen::VectorXd vars = square(sd.array()).matrix();
  Eigen::VectorXd logsd = log(sd.array()+1.0e-14).matrix();



  for(std::size_t c=0;c<nc;c++){
    dev = (arg.array()-mean.coeff(c)).matrix();
    sdev = (1.0/vars.coeff(c))*square(dev.array()).matrix();
    lps.col(c) = (-0.5*sdev.array() - logsd.coeff(c) - 0.9189385332046729 + logwts.coeff(c)).matrix();
    dlpm.col(c) = (1.0/vars.coeff(c))*dev;
    dlps.col(c) = (1.0/sd.coeff(c))*(sdev.array()-1.0).matrix();
  }


  Eigen::VectorXd maxlps = lps.array().rowwise().maxCoeff().matrix();

  lps -= maxlps.replicate(1,nc);

  Eigen::MatrixXd denWts = lps.array().exp().matrix();
  Eigen::VectorXd denWtsSum = denWts.array().rowwise().sum().matrix();

  denWts.array() /= denWtsSum.replicate(1,nc).array();

  dmean = (dlpm.array()*denWts.array()).colwise().sum().matrix();
  dsd = (dlps.array()*denWts.array()).colwise().sum().matrix();
  dwts = (denWts.array()/swts.transpose().replicate(n,1).array()).colwise().sum().matrix();

  return(maxlps.sum() + denWtsSum.array().log().sum());

}



template <class argType, class meanType, class sdType, class wtsType>
class normalMixture_ld{
  normalMixture_ld(const argType& arg,
                   const meanType& mean,
                   const sdType& sd,
                   const wtsType& wts){}
};


template <>
class normalMixture_ld<Eigen::VectorXd,Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>,Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>,Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> >{

  stan::math::var lpdf_;
public:
  normalMixture_ld(const Eigen::VectorXd& arg,
                   const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& mean,
                   const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& sd,
                   const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& wts){
    if(mean.size() != sd.size() || mean.size() != wts.size()){
      throw std::runtime_error("normalMixture_ld : mean, sd and wts must have same size");
    }
    std::size_t ncomp = mean.size();
    Eigen::VectorXd doubMean(ncomp),dmean(ncomp);
    Eigen::VectorXd doubSd(ncomp),dsd(ncomp);
    Eigen::VectorXd doubWts(ncomp),dwts(ncomp);

    for(std::size_t c=0;c<ncomp;c++){
      doubMean.coeffRef(c) = mean.coeff(c).val();
      doubSd.coeffRef(c) = sd.coeff(c).val();
      doubWts.coeffRef(c) = wts.coeff(c).val();
    }

    double lp = normalMixture_core(arg,doubMean,doubSd,doubWts,dmean,dsd,dwts);

    lpdf_ = lp;
    for(std::size_t c=0;c<ncomp;c++){
      lpdf_ += dmean.coeff(c)*(mean.coeff(c)-doubMean.coeff(c));
      lpdf_ += dsd.coeff(c)*(sd.coeff(c)-doubSd.coeff(c));
      lpdf_ += dwts.coeff(c)*(wts.coeff(c)-doubWts.coeff(c));
    }

  }
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const {return(lpdf_);}
};
}







#endif
