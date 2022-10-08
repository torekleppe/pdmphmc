#ifndef _AMTTRIDIAGCHOL_HPP_
#define _AMTTRIDIAGCHOL_HPP_

namespace amt{

template <class numType>
class triDiagChol{
  std::size_t n_;
  Eigen::Matrix<numType,Eigen::Dynamic,1> L_;
public:



  triDiagChol(const Eigen::Matrix<numType,Eigen::Dynamic,1>& diag,
              const Eigen::Matrix<numType,Eigen::Dynamic,1>& odiag)  : n_(diag.size()),L_(2*diag.size()) {
    if(odiag.size()!=n_-1){
      throw std::runtime_error("triDiagChol : odiag has wrong dimension");
    }
    L_.coeffRef(0) = cmn::sqrt(diag.coeff(0));
    numType LlogDet = cmn::log(L_.coeff(0));
    for(int i=1;i<n_;i++){
      L_.coeffRef(n_+i-1) = odiag.coeff(i-1)/L_.coeff(i-1);
      L_.coeffRef(i) = cmn::sqrt(diag.coeff(i)-cmn::square(L_.coeff(n_+i-1)));
      LlogDet += cmn::log(L_.coeff(i));
    }
    L_.coeffRef(2*n_-1) = LlogDet;
  }

 triDiagChol(const std::size_t n,
             const numType& firstLastDiag,
              const numType& remainingDiag,
            const numType& offDiagonal) : n_(n),L_(2*n) {

    L_.coeffRef(0) = cmn::sqrt(firstLastDiag);
    numType LlogDet = cmn::log(L_.coeff(0));
    for( size_t t = 1; t < n_-1; t++){
      L_.coeffRef(n_+t-1) = offDiagonal/L_.coeff(t-1);
      L_.coeffRef(t) = cmn::sqrt(remainingDiag - cmn::square(L_.coeff(n_+t-1)));
      LlogDet += cmn::log(L_.coeff(t));
    }
    L_.coeffRef(2*(n_-1)) = offDiagonal/L_.coeff(n_-2);
    L_.coeffRef(n_-1) = cmn::sqrt(firstLastDiag-cmn::square(L_.coeff(2*(n_-1))));
    L_.coeffRef(2*n_-1) = LlogDet + cmn::log(L_.coeff(n_-1));
  }

  /*
   * solves (L^T)*x = b for x where L=*this
   */

  void LT_solve(const Eigen::Ref<const Eigen::Matrix<numType,Eigen::Dynamic,1> >& b,
                Eigen::Matrix<numType,Eigen::Dynamic,1>& x) const {
    if(x.size()!=n_) x.resize(n_);
    if(b.size()!=n_){
      throw std::runtime_error("triDiagChol::LT_solve() : b has wrong dimension");
    }
    x.coeffRef(n_-1) = b.coeff(n_-1)/L_.coeff(n_-1);
    for(int i = n_-2;i>=0;i--){
      x.coeffRef(i) = (b.coeff(i) - x.coeff(i+1)*L_.coeff(n_+i))/L_.coeff(i);
    }
  }

  void LT_solve_inplace(Eigen::Matrix<numType,Eigen::Dynamic,1>& bx) const {

    if(bx.size()!=n_){
      throw std::runtime_error("triDiagChol::LT_solve_inplace() : bx has wrong dimension");
    }
    bx.coeffRef(n_-1)/=L_.coeff(n_-1);
    for(int i = n_-2;i>=0;i--){
      bx.coeffRef(i) -= bx.coeff(i+1)*L_.coeff(n_+i);
      bx.coeffRef(i) /= L_.coeff(i);
    }
  }

  /*
   * solves L*x = b for x where L=*this
   */

  void L_solve(const Eigen::Ref<const Eigen::Matrix<numType,Eigen::Dynamic,1> >& b,
               Eigen::Matrix<numType,Eigen::Dynamic,1>& x) const {
    if(x.size()!=n_) x.resize(n_);
    if(b.size()!=n_){
      throw std::runtime_error("triDiagChol::L_solve() : b has wrong dimension");
    }
    x.coeffRef(0) = b.coeff(0)/L_.coeff(0);
    for(int i=1;i<n_;i++){
      x.coeffRef(i) = (b.coeff(i)-x.coeff(i-1)*L_.coeff(n_+i-1))/L_.coeff(i);
    }
  }

  void L_solve_inplace(Eigen::Matrix<numType,Eigen::Dynamic,1>& bx) const {

    if(bx.size()!=n_){
      throw std::runtime_error("triDiagChol::L_solve_inplace() : bx has wrong dimension");
    }
    bx.coeffRef(0)/=L_.coeff(0);
    for(int i=1;i<n_;i++){
      bx.coeffRef(i) -= bx.coeff(i-1)*L_.coeff(n_+i-1);
      bx.coeffRef(i) /= L_.coeff(i);
    }
  }




  numType LlogDet() const {return L_.coeff(2*n_-1);}

  void dump() const {
    Eigen::MatrixXd tmp(n_,n_);
    for(size_t i=0;i<n_;i++) tmp(i,i) = asDouble(L_(i));
    for(size_t i=0;i<n_-1;i++) tmp(i+1,i) = asDouble(L_(n_+i));
    std::cout << "L = \n " << tmp << std::endl;
    std::cout << "log-determinant : " << asDouble(L_(2*n_-1)) << std::endl;

  }

}; // class

}//namespace


#endif

