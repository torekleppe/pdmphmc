#ifndef _AMTSPDMATRIX_HPP_
#define _AMTSPDMATRIX_HPP_

#include <Eigen/Dense>
namespace amt{

/*
 * Interface for interpreting a vector x of length d*(d+1)/2 as a
 * d x d symmetric positive definite matrix.
 *
 * Based on the representation A = L * D * L^T where D is diagonal
 * with diagonal elements exp(x[1:d]). L is unit (ones on the diagonal)
 * lower triangular with (below diagonal) columns
 * (starting from the left-most) corresponding to the remaining elements of
 * x. E.g. d=3 : D = diag(x[1:3]),
 *      [1    0    0]
 *L=    [x[4] 1    0]
 *      [x[5] x[6] 1]
 */

template <class xType>
class SPDmatrix{
public:
  // main storage kept public
  Eigen::Matrix<xType,Eigen::Dynamic,1> x_;
private:
  std::size_t n_;
  Eigen::Matrix<xType,Eigen::Dynamic,1> Lam_;
  /*
   * Values of the lower triangular part of V are x_(VlinInd(i,j))
   */

  inline size_t VlinInd(const size_t i, const size_t j) const {
    if(i<=j) throw std::runtime_error("SPDmatrix::VlinInd : bad arguments");
    return(n_*(j+1) - ((j+1)*j)/2 + i - j - 1);
  }
  inline void computeLam(){
    if(Lam_.size()!=n_){
      Lam_.resize(n_);

      for(std::size_t i = 0;i<n_;i++) {
        Lam_.coeffRef(i) = cmn::exp(x_.coeff(i));
      }
    }
  }


  // solves L[leading_dim..n-1,leading_dim..n-1]*x = b for x
  // where x and b have dimensions n-leading_dim
  void Lsolve_StanVal(
      const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& b,
      Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& x,
      const std::size_t leading_dim=0) const {

    x.coeffRef(0) = b.coeff(0);
    std::size_t c = 1;
    std::size_t r;
    for(std::size_t i=leading_dim+1;i<n_;i++){
      x.coeffRef(c) = b.coeff(c);
      r = 0;
      for(std::size_t j = leading_dim;j<i;j++) {
        x.coeffRef(c) -= x.coeff(r)*asStanVar(x_.coeff(VlinInd(i,j)));
        r++;
      }
      c++;
    }
  }

public:

  // constructors
  SPDmatrix(const size_t n,
            const Eigen::Matrix<xType,Eigen::Dynamic,1> &x) : n_(n), x_(x) {
    //std::cout << "constructor: x-dim: " << x.size() << std::endl;
    if(x_.size()!=n_*(n_+1)/2){
      throw std::runtime_error("SPDmatrix : x must be of length n*(n+1)/2");
    }
    computeLam();
  }
  // dimension
  inline std::size_t dim() const {return n_;}
  inline xType Lambda(const size_t i) const {return(Lam_.coeff(i));}

  /* access removing amt-AD information: *Val()-functions */
  inline stan::math::var Lambda_StanVal(const size_t i) const {
    if constexpr(std::is_same_v<amtVar,xType>){
      return(Lam_.coeff(i).val_);
    } else {
      return(Lam_.coeff(i));
    }
  }
  inline stan::math::var x_StanVal(const std::size_t i) const {
    if constexpr(std::is_same_v<amtVar,xType>){
      return(x_.coeff(i).val_);
    } else {
      return(x_.coeff(i));
    }
  }

  /* compute indiviual elements of represented matrix */
  inline xType coeff(const std::size_t i, const std::size_t j) const {
    xType ret = 0.0;
    if(i==j){
      for(std::size_t k=0;k<i;k++){
        ret+=cmn::square(x_.coeff(i-k-1+n_*(k+1)-((k+1)*k)/2))*Lam_.coeff(k);
      }
      ret+=Lam_.coeff(i);
    } else {
      std::size_t kapj;
      std::size_t indMin = std::min(i,j);
      for(std::size_t k=0;k<indMin;k++){
        kapj = n_*(k+1)-((k+1)*k)/2;
        ret+=x_.coeff(i-k-1+kapj)*x_.coeff(j-k-1+kapj)*Lam_.coeff(k);
      }
      kapj = n_*(indMin+1)-((indMin+1)*indMin)/2;
      ret+=x_.coeff(std::max(i,j)-indMin-1+kapj)*Lam_.coeff(indMin);
    }
    return(ret);
  }

  /* compute double value of indiviual elements of represented matrix */
  inline double coeff_double(const std::size_t i, const std::size_t j) const {
    double ret = 0.0;
    if(i==j){
      for(std::size_t k=0;k<i;k++){
        ret+=cmn::square(asDouble(x_.coeff(i-k-1+n_*(k+1)-((k+1)*k)/2)))*asDouble(Lam_.coeff(k));
      }
      ret+=asDouble(Lam_.coeff(i));
    } else {
      std::size_t kapj;
      std::size_t indMin = std::min(i,j);
      for(std::size_t k=0;k<indMin;k++){
        kapj = n_*(k+1)-((k+1)*k)/2;
        ret+=asDouble(x_.coeff(i-k-1+kapj))*asDouble(x_.coeff(j-k-1+kapj))*asDouble(Lam_.coeff(k));
      }
      kapj = n_*(indMin+1)-((indMin+1)*indMin)/2;
      ret+=asDouble(x_.coeff(std::max(i,j)-indMin-1+kapj))*asDouble(Lam_.coeff(indMin));
    }
    return(ret);
  }

  inline stan::math::var coeff_StanVal(const std::size_t i, const std::size_t j) const {
    stan::math::var ret = 0.0;
    if(i==j){
      for(std::size_t k=0;k<i;k++){
        ret+=stan::math::square(asStanVar(x_.coeff(i-k-1+n_*(k+1)-((k+1)*k)/2)))*asStanVar(Lam_.coeff(k));
      }
      ret+=asStanVar(Lam_.coeff(i));
    } else {
      std::size_t kapj;
      std::size_t indMin = std::min(i,j);
      for(std::size_t k=0;k<indMin;k++){
        kapj = n_*(k+1)-((k+1)*k)/2;
        ret+=asStanVar(x_.coeff(i-k-1+kapj))*asStanVar(x_.coeff(j-k-1+kapj))*asStanVar(Lam_.coeff(k));
      }
      kapj = n_*(indMin+1)-((indMin+1)*indMin)/2;
      ret+=asStanVar(x_.coeff(std::max(i,j)-indMin-1+kapj))*asStanVar(Lam_.coeff(indMin));
    }
    return(ret);
  }

  /* Basic linear algebra operations */
  xType logDet() const {return(x_.head(n_).sum());}
  stan::math::var logDet_StanVal() const {
    stan::math::var ret = 0.0;
    for(std::size_t i=0;i<n_;i++) ret+=asStanVar(x_.coeff(i));
    return(ret);
  }

  // computes b^T * P * b where *this = P and b is a vector of dimension n_
  template <class bType>
  typename amtReturnType2<xType,bType>::type quad_form(const Eigen::Matrix<bType,Eigen::Dynamic,1>& b) const {
    if(n_ != b.size()){
      throw std::runtime_error("SPDmatrix::quad_form : b incompatible with SPDmatrix");
    }
    typename amtReturnType2<xType,bType>::type ret,tmp;
    ret = 0.0;
    std::size_t k=n_;
    for(std::size_t j=0;j<n_;j++){
      tmp = b.coeff(j);
      for(std::size_t i=j+1;i<n_;i++) {
        tmp += b.coeff(i)*x_.coeff(k);
        k++;
      }
      ret += square(tmp)*Lam_.coeff(j);
    }
    return(ret);
  }
  template <class bType>
  inline stan::math::var quad_form_StanVal(const Eigen::Matrix<bType,Eigen::Dynamic,1>& b) const {
    if(n_ != b.size()){
      throw std::runtime_error("SPDmatrix::quad_formVal : b incompatible with SPDmatrix");
    }
    stan::math::var ret,tmp;
    ret = 0.0;
    std::size_t k=n_;
    for(std::size_t j=0;j<n_;j++){
      tmp = asStanVar(b.coeff(j));
      for(std::size_t i=j+1;i<n_;i++) {
        tmp += asStanVar(b.coeff(i))*asStanVar(x_.coeff(k));
        k++;
      }
      ret += cmn::square(tmp)*asStanVar(Lam_.coeff(j));
    }
    return(ret);
  }

  // computes b^T*P[leading_dim..n-1,leading_dim..n-1]^{-1}*b
  template <class bType>
  inline stan::math::var quad_form_inv_StanVal(const Eigen::Matrix<bType,Eigen::Dynamic,1>& b,
                                               const std::size_t leading_dim) const {
    Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> tmp(n_-leading_dim);
    Lsolve_StanVal(b,tmp,leading_dim);
    stan::math::var ret = 0.0;
    for(std::size_t i=leading_dim;i<n_;i++) ret+=cmn::square(tmp.coeff(i-leading_dim))/asStanVar(Lam_.coeff(i));
    return(ret);
  }




  // Todo: symmetric triangular products




  /*
   * computes a dense symmetric representation of the inverse of
   * P[leading_dim..n-1,leading_dim..n-1] where P=*this
   *
   */

  void packedInverse_StanVal(packedSym<stan::math::var>& inv,
                             const size_t leading_dim=0) const {
    std::size_t invn = n_-leading_dim;
    inv.allocate(invn);
    inv.write(invn-1,invn-1,stan::math::inv(asStanVar(Lam_.coeff(n_-1))));

    stan::math::var rhoTmp,diagTmp;
    size_t kap_p,kap_pp;
    size_t rhoDim = 1;
    size_t i = n_-2;




    for(size_t iter = 1; iter < invn;iter++){

      kap_p = n_ + i*n_ - (i*(i+1))/2;
      //std::cout  << kap_p << std::endl;
      //std::cout << i << std::endl;
      diagTmp = stan::math::inv(asStanVar(Lam_.coeff(i)));
      for(size_t j=0;j<rhoDim;j++){
        rhoTmp = 0.0;
        for(size_t k=0;k<rhoDim;k++){
          rhoTmp += inv.read(i+1+j-leading_dim,i+1+k-leading_dim)*asStanVar(x_.coeff(kap_p+k));
        }
        diagTmp += rhoTmp*asStanVar(x_.coeff(kap_p+j));
        //std::cout << rhoTmp << std::endl;
        inv.write(i-leading_dim,i+j+1-leading_dim,-rhoTmp);
      }
      inv.write(i-leading_dim,i-leading_dim,diagTmp);
      rhoDim++;
      i--;
    }


  }






  /*
   * Only for debugging purposes
   */
  Eigen::MatrixXd asDoubleMatrix(bool print=true) const {

    Eigen::MatrixXd V(n_,n_);
    Eigen::MatrixXd Lam(n_,n_);
    V.setZero();
    Lam.setZero();
    for(size_t j=0;j<n_;j++){
      V(j,j) = 1.0;
      Lam(j,j) = std::exp(asDouble(x_.coeff(j)));
      if(j<n_-1){
        for(size_t i=j+1;i<n_;i++) V(i,j) = asDouble(x_.coeff(VlinInd(i,j)));
      }
    }
    Eigen::MatrixXd P = V*Lam*V.transpose();
    if(print){
      std::cout << "dump of SPDmatrix object:" << std::endl;
      std::cout << "V =\n" << V << std::endl;
      std::cout << "Lambda =\n" << Lam << std::endl;
      std::cout << "P =\n" << P << std::endl;
    }
    return(P);
  }




}; // class

}// namespace
#endif
