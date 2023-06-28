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
      std::cout << "linearPreconditioner::toPar : not initialized" << std::endl;
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

  Eigen::VectorXd toStdPar(const Eigen::VectorXd& par){
    Eigen::VectorXd dev = par;
    if(mean_.size()>0) dev-=mean_;
    if(precCholFac_.rows()>0){
      return(precCholFac_.transpose()*dev);
    } else if(varCholFac_.rows()>0){
      return(varCholFac_.triangularView<Eigen::Lower>().solve(dev));
    } else {
      std::cout << "linearPreconditioner::toStdPar : not initialized" << std::endl;
      throw 3453457;
    }
  }

  void printSummary(){
    if(init_){
      int dim = std::max(precCholFac_.rows(),varCholFac_.rows());
      std::cout << "linearPreconditioner, dimension : " << dim << std::endl;
      if(mean_.size()>0){
        std::cout << "Mean: \n" << mean_ << std::endl;
      } else {
        std::cout << "zero mean," << std::endl;
      }
      std::cout << "Jacobian: \n" << std::endl;
      Eigen::MatrixXd jac;
      if(precCholFac_.rows()>0){
        Eigen::MatrixXd id;
        id.setIdentity(dim,dim);
        jac = precCholFac_.transpose().triangularView<Eigen::Upper>().solve(id);
      } else {
        jac = varCholFac_;
      }
      std::cout << "Jacobian: \n" << jac << std::endl;
      std::cout << "Covariance matrix: \n" << jac*jac.transpose() << std::endl;

    } else {
      std::cout << "linearPreconditioner, not initialized" << std::endl;
    }
  }
};


class wishartPreconditioner{
  Eigen::MatrixXd scaleL_;
  double nu_;
  int n_;
  int xn_;
  std::vector<Eigen::VectorXd> gmeans_;
  Eigen::VectorXd gamMeans_;
  Eigen::VectorXd gamStds_;
public:
  wishartPreconditioner(){}
  int vecDim() const {return xn_;}
  void setPars(const double nu,
                        const Eigen::MatrixXd& scaleMat){
    nu_ = nu;
    n_ = scaleMat.rows();
    xn_ = n_*(n_+1)/2;
    scaleL_ = Eigen::LLT<Eigen::MatrixXd>(scaleMat).matrixL();
    int n = scaleL_.rows();
    for(size_t i=0;i<n-1;i++){
      gmeans_.emplace_back(scaleL_.col(i).tail(n-i-1)/scaleL_.coeff(i,i));
    }
    gamMeans_.resize(n);
    gamStds_.resize(n);
    double a,b;
    for(size_t i=0;i<n;i++){
      a = 0.5*(nu_-static_cast<double>(i));
      b = 2.0*std::pow(scaleL_.coeff(i,i),2);
      gamMeans_(i) = std::log(b) + stan::math::digamma(a);
      gamStds_(i) = std::sqrt(stan::math::trigamma(a));
    }
  }

  template <class modelType, class varType>
  void toPar(modelType& model,
             const Eigen::Matrix<varType,Eigen::Dynamic,1>& stdPar,
             Eigen::Matrix<varType,Eigen::Dynamic,1>& par) const {
    if(stdPar.size()!=xn_){
      std::cout << "bad dimension in wishartPreconditioner" << std::endl;
      throw 345345;
    }
    if(par.size()!=stdPar.size()) par.resize(stdPar.size());


    for(size_t i=0;i<n_;i++) par.coeffRef(i) = gamMeans_.coeff(i) + gamStds_.coeff(i)*stdPar.coeff(i);

    size_t offset = n_;
    size_t blockSize;
    varType efac;
    varType ldet = 0.0;
    for(size_t j=0;j<n_-1;j++){
      efac = exp(-0.5*par.coeff(j));
      ldet -= (0.5*static_cast<double>(n_-j-1))*par.coeff(j);
      blockSize = n_ - j - 1;
      for(size_t i=0;i<blockSize;i++){
        par.coeffRef(i+offset) = 0.0;
        for(size_t jj=0;jj<=i;jj++){
          par.coeffRef(i+offset) += scaleL_.coeff(i+j+1,jj+j+1)*stdPar(offset+jj);
        }
        par.coeffRef(i+offset) *= efac;
        par.coeffRef(i+offset) += gmeans_[j].coeff(i);
      }
      offset += blockSize;
    }
    // add log-Jacobian determinant to target
    model+=ldet;
  } // toPar

  template <class modelType, class varType>
  SPDmatrix<varType> operator()(modelType& model,
                              const Eigen::Matrix<varType,Eigen::Dynamic,1>& stdPar) const {
    Eigen::Matrix<varType,Eigen::Dynamic,1> par;
    toPar(model,stdPar,par);
    return(SPDmatrix<varType>(n_,par));
  }

  void printSummary(){
    std::cout << "wishartPreconditioner, dimension: " << n_ << std::endl;
    std::cout << "nu (degrees of freedom): " << nu_ << std::endl;
    std::cout << "scale matrix:\n" << scaleL_*scaleL_.transpose() << std::endl;
  }
}; // wishartPreconditioner
} // namespace
#endif

