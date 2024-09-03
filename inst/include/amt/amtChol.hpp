#ifndef _AMT_CHOL_HPP_
#define _AMT_CHOL_HPP_

/*
 * Simple templated cholesky factorization for which may be used both for
 * stan::math::var and amt::amtVar variables.
 *
 */

namespace amt{

template <class numericType_>
class chol{
  Eigen::Matrix<numericType_,Eigen::Dynamic,Eigen::Dynamic> L_;

  void inplaceLsolve(Eigen::Ref<Eigen::Matrix<numericType_,Eigen::Dynamic,1> > bx){
    bx.coeffRef(0) /= L_.coeff(0,0);
    for(std::size_t i=1;i<L_.rows();i++){
      bx.coeffRef(i) = (bx.coeff(i)-bx.head(i).dot(L_.row(i).head(i)))/L_.coeff(i,i);
    }
  }

  void inplaceLTsolve(Eigen::Ref<Eigen::Matrix<numericType_,Eigen::Dynamic,1> > bx){
    int d = L_.rows();
    bx.coeffRef(d-1) /= L_.coeff(d-1,d-1);
    for(int i=d-2;i>=0;i--){
      bx.coeffRef(i) = (bx.coeff(i)-bx.tail(d-i-1).dot(L_.col(i).tail(d-i-1)))/L_.coeff(i,i);
    }
  }

public:
  chol(const Eigen::Matrix<numericType_,Eigen::Dynamic,Eigen::Dynamic>& A){
    std::size_t d = A.rows();
    if(A.cols() != d){
      throw std::runtime_error("chol: input matrix not square");
    }
    L_.resize(d,d);
    L_.setZero();
    numericType_ s;
    for(std::size_t i=0;i<d;i++){
      for(std::size_t j=0;j<=i;j++){
        s = L_.row(i).head(j).dot(L_.row(j).head(j));
        if(i==j){
          L_.coeffRef(i,j) = cmn::sqrt(A.coeff(i,i)-s);
        } else {
          L_.coeffRef(i,j) = (A.coeff(i,j)-s)/L_.coeff(j,j);
        }
      }
    }
  }


  template <class btype>
  Eigen::Matrix<numericType_,Eigen::Dynamic,1> solve(Eigen::Matrix<btype,Eigen::Dynamic,1>& b){
    if(b.size()!=L_.rows()){
      throw std::runtime_error("chol::solve(vector) wrong dimension of b");
    }
    Eigen::Matrix<numericType_,Eigen::Dynamic,1> ret(L_.rows());
    for(std::size_t i=0;i<L_.rows();i++) ret.coeffRef(i) = b.coeff(i); //loop in case of mixed types
    inplaceLsolve(ret);
    inplaceLTsolve(ret);
    return(ret);
  }

  template <class btype>
  Eigen::Matrix<numericType_,Eigen::Dynamic,1> Lsolve(Eigen::Matrix<btype,Eigen::Dynamic,1>& b){
    if(b.size()!=L_.rows()){
      throw std::runtime_error("chol::solve(vector) wrong dimension of b");
    }
    Eigen::Matrix<numericType_,Eigen::Dynamic,1> ret(L_.rows());
    for(std::size_t i=0;i<L_.rows();i++) ret.coeffRef(i) = b.coeff(i); //loop in case of mixed types
    inplaceLsolve(ret);
    return(ret);
  }

  template <class btype>
  Eigen::Matrix<numericType_,Eigen::Dynamic,Eigen::Dynamic> solve(Eigen::Matrix<btype,Eigen::Dynamic,Eigen::Dynamic>& b){
    if(b.rows()!=L_.rows()){
      throw std::runtime_error("chol::solve(matrix) wrong dimension of b");
    }
    Eigen::Matrix<numericType_,Eigen::Dynamic,Eigen::Dynamic> ret(L_.rows(),b.cols());
    for(std::size_t j=0;j<b.cols();j++){
      for(std::size_t i=0;i<L_.rows();i++){
        ret.coeffRef(i,j) = b.coeff(i,j); //loop in case of mixed types
      }
      inplaceLsolve(ret.col(j));
      inplaceLTsolve(ret.col(j));
    }
    return(ret);
  }

  template <class btype>
  Eigen::Matrix<numericType_,Eigen::Dynamic,Eigen::Dynamic> Lsolve(Eigen::Matrix<btype,Eigen::Dynamic,Eigen::Dynamic>& b){
    if(b.rows()!=L_.rows()){
      throw std::runtime_error("chol::solve(matrix) wrong dimension of b");
    }
    Eigen::Matrix<numericType_,Eigen::Dynamic,Eigen::Dynamic> ret(L_.rows(),b.cols());
    for(std::size_t j=0;j<b.cols();j++){
      for(std::size_t i=0;i<L_.rows();i++){
        ret.coeffRef(i,j) = b.coeff(i,j); //loop in case of mixed types
      }
      inplaceLsolve(ret.col(j));
    }
    return(ret);
  }





  Eigen::Matrix<numericType_,Eigen::Dynamic,Eigen::Dynamic> L() const {return L_;}


};




} // namespace

#endif
