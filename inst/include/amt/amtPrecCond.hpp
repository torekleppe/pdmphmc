#ifndef _AMTPRECOND_HPP_
#define _AMTPRECOND_HPP_

namespace amt{


class linearPreconditioner{
  Eigen::MatrixXd precCholFac_;
  Eigen::MatrixXd varCholFac_;
  Eigen::VectorXd mean_;
  bool init_;
public:
  linearPreconditioner() : init_(false) {}
  void setPrec(const Eigen::MatrixXd& prec){
    if(! init_){
      precCholFac_ = stan::math::cholesky_decompose(prec);
      init_ = true;
    } else {
      std::cout << "linearPreconditioner already initialized, ignored" << std::endl;
    }
  }
  void setVar(const Eigen::MatrixXd& var){
    if(! init_){
      varCholFac_ = stan::math::cholesky_decompose(var);
      init_ = true;
    } else {
      std::cout << "linearPreconditioner already initialized, ignored" << std::endl;
    }
  }
  void setRegressionPar(const Eigen::VectorXd& y,
                        const Eigen::MatrixXd& x){
    if(! init_){
      if(y.size()!=x.rows()){
        std::cout << "bad dimensions in linearPreconditioner::setRegressionPar" << std::endl;
        throw 123;
      }

      Eigen::LLT<Eigen::MatrixXd> llt(x.transpose()*x);
      mean_ = llt.solve(x.transpose()*y);
      precCholFac_ = llt.matrixL();
      double ssq = (y-x*mean_).squaredNorm();
      double fac = std::max(1.0,static_cast<double>(y.size() - x.cols()));
      precCholFac_ *= std::sqrt(fac/ssq);
      init_ = true;
    } else {
      std::cout << "linearPreconditioner already initialized, ignored" << std::endl;
    }
  }
  template <class varType>
  void toPar(const Eigen::Matrix<varType,Eigen::Dynamic,1>& stdPar,
             Eigen::Matrix<varType,Eigen::Dynamic,1>& par) const {
    if(precCholFac_.rows()>0){
      int d = stdPar.size();
      if(d != precCholFac_.cols()){
        std::cout << "bad dimension in linearPreconditioner::toPar" << std::endl;
        throw 345345;
      }
      if(par.size() != d) par.resize(d);
      par.coeffRef(d-1) = stdPar.coeff(d-1)/precCholFac_.coeff(d-1,d-1);
      for(int j=d-2;j>=0;j--){
        par.coeffRef(j) = stdPar.coeff(j);
        for(int k=j+1;k<d;k++) par.coeffRef(j) -= precCholFac_.coeff(k,j)*par.coeff(k);
        par.coeffRef(j) /= precCholFac_.coeff(j,j);
      }

      //std::cout << "original :\n " << stdPar.transpose() << std::endl;
      //std::cout << "test:\n" << (precCholFac_.transpose()*par).transpose() << std::endl;


    } else if(varCholFac_.rows()>0) {
      int d = stdPar.size();
      if(d != varCholFac_.cols()){
        std::cout << "bad dimension in linearPreconditioner::toPar" << std::endl;
        throw 345345;
      }
      par = stdPar.coeff(0)*varCholFac_.col(0); // implicitly resizes par
      for(int j=1;j<d;j++){
        for(int k=1;k<=j;k++) par.coeffRef(j) +=  varCholFac_.coeff(j,k)*stdPar.coeff(k);
      }
    } else {
      std::cout << "linearPreconditioner::toPar : not initialized with setVar() or setPrec()" << std::endl;
      throw 3453457;
    }
    if(mean_.size()>0){
      for(int j=0;j<par.size();j++) par.coeffRef(j) += mean_.coeff(j);
    }
  }

  template <class varType>
  Eigen::Matrix<varType,Eigen::Dynamic,1> operator()(
    const Eigen::Matrix<varType,Eigen::Dynamic,1>& stdPar) const {
    Eigen::Matrix<varType,Eigen::Dynamic,1> ret;
    toPar(stdPar,ret);
    return(ret);
  }



};


} // namespace
#endif

