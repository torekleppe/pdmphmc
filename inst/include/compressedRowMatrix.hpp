#ifndef _COMPRESSEDROWMATRIX_HPP_
#define _COMPRESSEDROWMATRIX_HPP_

#ifndef _COMPRESSEDROWMATRIX_BLOCKSIZE_
#define _COMPRESSEDROWMATRIX_BLOCKSIZE_ 1000
#endif

template <class T>
class compressedRowMatrix{
  Eigen::Matrix<T,Eigen::Dynamic,1> vals_;
  Eigen::Matrix<std::size_t,Eigen::Dynamic,1> cinds_;
  Eigen::Matrix<std::size_t,Eigen::Dynamic,1> rinds_;
  Eigen::Matrix<std::size_t,Eigen::Dynamic,1> preImNz_;
  std::size_t cols_;
  std::size_t rows_;
  std::size_t ele_count_;

  void updatePreImNz(){
    Eigen::Matrix<int,Eigen::Dynamic,1> colNz(cols_);
    colNz.setZero();
    for(size_t i=0;i<ele_count_;i++) colNz(cinds_.coeff(i)) = 1;
    //std::cout << "colNz\n" << colNz.transpose() << std::endl;
    int nz = colNz.sum();
    if(preImNz_.size()!=nz) preImNz_.resize(nz);
    //std::cout << "preImNz before \n" << preImNz_.transpose() << std::endl;
    int c = 0;
    for(size_t i=0;i<colNz.size();i++){
      if(colNz(i)==1){
        preImNz_.coeffRef(c) = i;
        c++;
      }
    }
    //std::cout << "preImNz after \n" << preImNz_.transpose() << std::endl;
    //std::cout << "colNz \n" << colNz << std::endl;
    //std::cout << "nz : " << nz << std::endl;
    //std::cout << "preImNz \n" << preImNz_ << std::endl;
  }


public:
  compressedRowMatrix() : rows_(0),cols_(0),ele_count_(0) {}
  compressedRowMatrix(const size_t cols) : rows_(0), cols_(cols), ele_count_(0) {}
  inline void setCols(const size_t cols){ // conservative resize of columns
    if(cols>=cols_){
      cols_ = cols;
    } else {
      std::cout << "error in compressedRowMatrix::setCols" << std::endl;
      throw(878);
    }
  }
  inline size_t rows() const {return rows_;}
  inline size_t rowNz(const size_t whichRow) const {return rinds_(whichRow+1)-rinds_(whichRow);}
  inline size_t getColIndex(const size_t whichRow, const size_t offset) const {return cinds_.coeff(rinds_.coeff(whichRow)+offset);}
  inline double getValColIndex(const size_t whichRow, const size_t offset) const {return vals_.coeff(rinds_.coeff(whichRow)+offset);}
  inline bool isAllFinite() const {return vals_.head(ele_count_).array().isFinite().all();}
  inline bool isEqualTo(const compressedRowMatrix<double>& other){
    if(rows_ != other.rows_ || cols_ != other.cols_ || ele_count_ != other.ele_count_) return(false);

    return(((vals_.head(ele_count_)-other.vals_.head(ele_count_)).array().abs() <
      1.0e-14*(vals_.head(ele_count_).array().abs().max(1.0))).all());
  }
  void cpValsFrom(const compressedRowMatrix<T>& from){
    if(ele_count_==from.ele_count_){
      vals_.head(ele_count_) = from.vals_.head(ele_count_);
    } else {
      std::cout << "error in compressedRowMatrix::cpValsFrom" << std::endl;
    }
  }

  void inline toTriplet(Eigen::VectorXi& i,
                        Eigen::VectorXi& j,
                        Eigen::VectorXd& val) const {
    if(i.size()!=ele_count_) i.resize(ele_count_);
    if(j.size()!=ele_count_) j.resize(ele_count_);
    if(val.size()!=ele_count_) val.resize(ele_count_);
    for(size_t ii=0; ii<rows_;ii++){
      for(size_t jj=rinds_(ii);jj<rinds_(ii+1);jj++){
        i.coeffRef(jj) = static_cast<int>(ii)+1;
        j.coeffRef(jj) = static_cast<int>(cinds_(jj))+1;
        val.coeffRef(jj) = vals_.coeff(jj);
      }
    }
  }

  inline void pushDenseRow(const Eigen::Matrix<T,Eigen::Dynamic,1> row){
    if(row.size()>cols_){
      std::cout << "bad vector into compressedRowMatrix::pushDenseRow" << std::endl;
      throw(879);
    }
    rows_++;
    if(rinds_.size()<rows_+1){
      rinds_.conservativeResize(rinds_.size()+_COMPRESSEDROWMATRIX_BLOCKSIZE_);
    }
    if(rows_==1) rinds_.coeffRef(0) = 0;
    // count number of new elements
    Eigen::Array<bool,Eigen::Dynamic,1> nonz = row.array().abs()>1.0e-14;
    ele_count_ += nonz.count();
    if(vals_.size()<ele_count_){
      vals_.conservativeResize(vals_.size()+_COMPRESSEDROWMATRIX_BLOCKSIZE_);
      cinds_.conservativeResize(cinds_.size()+_COMPRESSEDROWMATRIX_BLOCKSIZE_);
    }
    rinds_.coeffRef(rows_) = ele_count_;
    size_t c = rinds_.coeff(rows_-1);
    for(std::size_t i=0;i<row.size();i++){
      if(nonz.coeff(i)){
        vals_.coeffRef(c) = row.coeff(i);
        cinds_.coeffRef(c) = i;
        c++;
      }
    }
    updatePreImNz();
    //std::cout << preImNz_ << std::endl;
  }


  void rightMultiplyDiag(const Eigen::Matrix<T,Eigen::Dynamic,1>& diag){
    size_t t;
    for(size_t i=0; i<rows_;i++){
      t = rinds_.coeff(i+1)-rinds_.coeff(i);
      vals_.segment(rinds_.coeff(i),t).array() *= cinds_.segment(rinds_.coeff(i),t).unaryExpr(diag).array();
    }
  }

  void rightMultiplyVec(const Eigen::Matrix<T,Eigen::Dynamic,1>& rhs,
                        Eigen::Matrix<T,Eigen::Dynamic,1>& result) const {
    if(result.size()!=rows_) result.resize(rows_);
    size_t t;
    for(size_t i = 0; i<rows_;i++){
      t = rinds_.coeff(i+1)-rinds_.coeff(i);
      result.coeffRef(i) = vals_.segment(rinds_.coeff(i),t).dot(
        cinds_.segment(rinds_.coeff(i),t).unaryExpr(rhs)
      );
    }
  }

  Eigen::Matrix<T,Eigen::Dynamic,1> operator*(const Eigen::Matrix<T,Eigen::Dynamic,1>& rhs) const {
    Eigen::Matrix<T,Eigen::Dynamic,1> ret;
    rightMultiplyVec(rhs,ret);
    return(ret);
  }

  // same as Eigen operation (*this).transpose()*rhs
  void transposedRightMultiplyVec(const Eigen::Matrix<T,Eigen::Dynamic,1>& rhs,
                                  Eigen::Matrix<T,Eigen::Dynamic,1>& result) const {
    if(result.size()!=cols_) result.resize(cols_);
    result.setZero();
    size_t t;
    for(size_t i=0;i<rows_;i++){
      for(size_t j=rinds_.coeff(i);j<rinds_.coeff(i+1);j++){
        result.coeffRef(cinds_.coeff(j)) += rhs.coeff(i)*vals_.coeff(j);
      }
    }
    //std::cout << "dense:\n" << (asDense().transpose()*rhs).transpose() << std::endl;
    //std::cout << "this :\n" << result.transpose() << std::endl;
  }



  // same as Eigen operation (*this).row(whichRow).dot(rhs)
  // where it is assumed that rhs.size() >= cols_
  inline T rowDot(const int whichRow,
                  const Eigen::VectorXd& rhs) const {
    T ret = 0.0;
    for(size_t j=rinds_.coeff(whichRow);j<rinds_.coeff(whichRow+1);j++){
      ret += vals_.coeff(j)*rhs.coeff(cinds_.coeff(j));
    }
    return(ret);
  }

  // same as Eigen operation (*this).row(whichRow).head(rhs.size()).dot(rhs)
  // note that rhs.size()
  inline T rowHeadDot(const int whichRow,
                      const Eigen::VectorXd& rhs) const {
    T ret = 0.0;
    size_t rhsSize = rhs.size();
    for(size_t j=rinds_.coeff(whichRow);j<rinds_.coeff(whichRow+1);j++){
      if(cinds_.coeff(j)<rhsSize){
        ret+= rhs.coeff(cinds_.coeff(j))*vals_.coeff(j);
      } else {
        break;
      }
    }
    return(ret);
  }

  inline T rowSquaredNorm(const int whichRow) const {
    return vals_.segment(rinds_.coeff(whichRow),rinds_.coeff(whichRow+1)-rinds_.coeff(whichRow)).squaredNorm();
  }


  //
  // computes rhs += alpha*(*this).row(whichRow).head(lhs.size())
  inline void scaledRowHeadIncrement(const int whichRow,
                                     const double alpha,
                                     Eigen::Ref<Eigen::VectorXd> lhs) const {
    size_t lhsSize = lhs.size();
    for(size_t j=rinds_.coeff(whichRow);j<rinds_.coeff(whichRow+1);j++){
      if(cinds_.coeff(j)<lhsSize){
        lhs.coeffRef(cinds_.coeff(j)) += alpha*vals_.coeff(j);
      } else {
        break;
      }
    }
  }


  /*
   * assuming that the dim=momentum.size() left-most columns of
   * *this represent A*S and that we have a constraint on the form c(y)
   * where y = A*S*q' + (A*m+b), where \nabla_y c(y) = rawGradient,
   *
   * this function updates the standardized momentum p' as
   *
   * momentum -= fac*n
   *
   * where
   * n = ((A*S).leftCols(dim))^T*rawGradient
   *
   * fac = 2.0*(n.dot(momentum))/(n.dot(n))
   *
   * and returns fac.
   *
   * The routine takes advantage of any sparsity in n
   */
  Eigen::VectorXd nvec_;
  inline double splinStandardizedCollisionMomentumUpdate(const Eigen::VectorXd& rawGradient,
                                                         Eigen::Ref<Eigen::VectorXd> momentum,
                                                         const double reversalThresh=0.0)  {
    //std::cout << preImNz_ << std::endl;
    if(momentum.size()<preImNz_.coeff(preImNz_.size()-1)+1){
      std::cout << "error in compressedRowMatrix::splinStandardizedCollisionMomentumUpdate" << std::endl;
      throw(890);
    }

    if(nvec_.size()!=momentum.size()) nvec_.resize(momentum.size());
    nvec_.setZero();
    size_t t;
    for(size_t i=0;i<rows_;i++){
      for(size_t j=rinds_.coeff(i);j<rinds_.coeff(i+1);j++){
        nvec_.coeffRef(cinds_.coeff(j)) += rawGradient.coeff(i)*vals_.coeff(j);
      }
    }
    //std::cout << "before fac" << std::endl;

    double nvecSnorm = (preImNz_.unaryExpr(nvec_)).squaredNorm();
    double fac = 2.0*preImNz_.unaryExpr(momentum).dot(preImNz_.unaryExpr(nvec_))/nvecSnorm;
    double phi;
    if(reversalThresh>0.0){
      phi = 0.5*fac*sqrt(nvecSnorm)/preImNz_.unaryExpr(momentum).norm();
    } else {
      phi = 2.0;
    }

    //std::cout << "abs(phi) : " << std::fabs(phi) << std::endl;
    if(std::fabs(phi)<1.0e-3) throw 32;

    if(fac<0.0){
      if(std::fabs(phi)>reversalThresh){
        for(size_t i=0;i<preImNz_.size();i++) momentum.coeffRef(preImNz_.coeff(i)) -= fac*nvec_.coeff(preImNz_.coeff(i));
      } else {
        for(size_t i=0;i<preImNz_.size();i++) momentum.coeffRef(preImNz_.coeff(i)) = -momentum.coeffRef(preImNz_.coeff(i));
      }
    }
    return(fac);
  }


  /*
   * assuming that the dim=momentum.size() left-most columns of
   * *this represent A*S and that we have a constraint on the form c(y)
   * where y = A*S*q' + (A*m+b), where \nabla_y c(y) = rawGradient,
   *
   * this function updates the standardized momentum p' as
   *
   * momentum(preImNz) = z - (<z+momentum(preImNz),n(preImNz)>/<n(preImNz),n(preImNz)>)*n(preImNz)
   *
   * where
   *
   * z \sim N(0,I) \in preImNz.size()
   *
   * n = ((A*S).leftCols(dim))^T*rawGradient
   *
   * fac = 2.0*(n.dot(momentum))/(n.dot(n))
   *
   * and returns fac.
   *
   * The routine takes advantage of any sparsity in n
   */
  Eigen::VectorXd boundary_z_;
  inline double splinStandardizedRandomizedMomentumUpdate(
      rng& r, // used
      const Eigen::VectorXd& rawGradient,
      Eigen::Ref<Eigen::VectorXd> momentum){

    if(nvec_.size()!=momentum.size()) nvec_.resize(momentum.size());
    if(boundary_z_.size()!=preImNz_.size()) boundary_z_.resize(preImNz_.size());
    nvec_.setZero();
    r.rnorm(boundary_z_);
    size_t t;
    for(size_t i=0;i<rows_;i++){
      for(size_t j=rinds_.coeff(i);j<rinds_.coeff(i+1);j++){
        nvec_.coeffRef(cinds_.coeff(j)) += rawGradient.coeff(i)*vals_.coeff(j);
      }
    }

    double nvecSnorm = (preImNz_.unaryExpr(nvec_)).squaredNorm();
    double halffac = preImNz_.unaryExpr(momentum).dot(preImNz_.unaryExpr(nvec_))/nvecSnorm;
    double fac = 2.0*halffac;
    double scal = halffac + boundary_z_.dot(preImNz_.unaryExpr(nvec_))/nvecSnorm;

    double phi = 0.5*fac*sqrt(nvecSnorm)/preImNz_.unaryExpr(momentum).norm();
    //std::cout << "phi: " << phi << std::endl;
    if(fabs(phi)<0.0001){
      std::cout << "momentum refresh at boundary" << std::endl;
      if(preImNz_.unaryExpr(nvec_).dot(boundary_z_)>0.0){
        for(size_t i=0;i<preImNz_.size();i++){
          momentum.coeffRef(preImNz_.coeff(i)) = boundary_z_.coeff(i);
        }
      } else {
        for(size_t i=0;i<preImNz_.size();i++){
          momentum.coeffRef(preImNz_.coeff(i)) = -boundary_z_.coeff(i);
        }
      }
    } else {
    double tmp1;
    if(fac<0.0){
        for(size_t i=0;i<preImNz_.size();i++){
          tmp1 = nvec_.coeff(preImNz_.coeff(i));
#ifndef _RANDBOUNCE_NO_SPARSITY_CHECK_
          if(std::fabs(tmp1)>1.0e-14) momentum.coeffRef(preImNz_.coeff(i)) = boundary_z_.coeff(i)-scal*tmp1;
#else
          momentum.coeffRef(preImNz_.coeff(i)) = boundary_z_.coeff(i)-scal*tmp1;
#endif
        }
      }
    }
    return(fac);
  }



  Eigen::MatrixXd asDense() const {
    Eigen::MatrixXd dense(rows_,cols_);
    dense.setZero();
    for(size_t i =0;i<rows_;i++){
      for(size_t j=rinds_(i);j<rinds_(i+1);j++){
        dense(i,cinds_(j)) = vals_(j);
      }
    }
    return(dense);
  }

  void dump(){
    Eigen::MatrixXd dense(rows_,cols_);
    dense.setZero();
    for(size_t i =0;i<rows_;i++){
      for(size_t j=rinds_(i);j<rinds_(i+1);j++){
        dense(i,cinds_(j)) = vals_(j);
      }
    }
    std::cout << "compressedRowMatrix: # rows: " << rows_ << ", # cols: " << cols_ << std::endl;
    std::cout << "dense representation:\n" << dense << std::endl;
    std::cout << "rinds:\n" << rinds_.head(rows_+1).transpose() << std::endl;
  }

  friend std::ostream& operator<< (std::ostream& out, const compressedRowMatrix<T>& obj);
};

std::ostream& operator<< (std::ostream& out, const compressedRowMatrix<double>& obj){

  out << "compressedRowMatrix: #rows:" << obj.rows_ << ", #cols:" << obj.cols_ << ", #nz:" << obj.ele_count_ <<  std::endl;

  if(obj.rows_<20 && obj.rows_>0 && obj.cols_<20 && obj.cols_>0){
    Eigen::MatrixXd dense(obj.rows_,obj.cols_);
    dense.setZero();
    for(size_t i =0;i<obj.rows_;i++){
      for(size_t j=obj.rinds_(i);j<obj.rinds_(i+1);j++){
        dense(i,obj.cinds_(j)) = obj.vals_(j);
      }
    }
    out << "dense representation:\n" << dense << std::endl;
  }
  return out;
}


#endif
