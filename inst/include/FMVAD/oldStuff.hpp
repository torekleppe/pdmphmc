#ifndef _FMVAD_HPP_
#define _FMVAD_HPP_
#include <complex>
#include <Eigen/Dense>

namespace FMVAD {

static size_t _FMVAD_global_dim_ = 0;

class FMVADvar {
  Eigen::VectorXd grad_;
  double val_;
public:
  FMVADvar() : grad_(_FMVAD_global_dim_), val_(0.0) { grad_.setZero();}
  FMVADvar(const double val) : grad_(_FMVAD_global_dim_), val_(val) {grad_.setZero();}
  FMVADvar(const double val, const Eigen::VectorXd& grad) : val_(val), grad_(grad) {}
  FMVADvar(const FMVADvar& rhs) : grad_(rhs.grad_), val_(rhs.val_) {}
  inline double val() const {return val_;}
  inline double grad(const int which) const {return grad_.coeff(which);}
  inline void __setGrad(const int which, const double val) {grad_.coeffRef(which)=val;}
  inline void operator+=(const FMVADvar& rhs){
    val_+=rhs.val_;
    grad_+=rhs.grad_;
  }
  inline FMVADvar operator+(const FMVADvar& rhs) const {
    FMVADvar ret = *this;
    ret+=rhs;
    return(ret);
  }
  inline FMVADvar operator+(const double rhs) const {
    FMVADvar ret = *this;
    ret.val_+=rhs;
    return(ret);
  }
  inline void operator-=(const FMVADvar& rhs){
    val_-=rhs.val_;
    grad_-=rhs.grad_;
  }
  inline FMVADvar operator-(const FMVADvar& rhs) const {
    FMVADvar ret = *this;
    ret-=rhs;
    return(ret);
  }
  inline FMVADvar operator-(const double rhs) const {
    FMVADvar ret = *this;
    ret.val_-=rhs;
    return(ret);
  }
  inline FMVADvar operator-() const {return(FMVADvar(-val_,-grad_));}
  inline void operator*=(const FMVADvar& rhs){
    grad_ *= rhs.val_;
    grad_ += val_*rhs.grad_;
    val_ *= rhs.val_;
  }
  inline FMVADvar operator*(const FMVADvar& rhs) const {
    FMVADvar ret = *this;
    ret*=rhs;
    return(ret);
  }
  inline void operator*=(const double rhs){
    val_*=rhs;
    grad_*=rhs;
  }
  inline FMVADvar operator*(const double rhs) const {
    FMVADvar ret = *this;
    ret*=rhs;
    return(ret);
  }

  inline void operator/=(const FMVADvar& rhs){
    grad_/=rhs.val_;
    val_/=rhs.val_;
    grad_-= (val_/rhs.val_)*rhs.grad_;
  }
  inline FMVADvar operator/(const FMVADvar& rhs) const {
    FMVADvar ret = *this;
    ret/=rhs;
    return(ret);
  }
  inline FMVADvar operator/(const double rhs) const {
    return(FMVADvar(val_/rhs,grad_/rhs));
  }


  inline void Sqrt(){
    val_ = sqrt(val_);
    grad_ *= 0.5/val_;
  }
  inline void Square(){
    grad_ *= 2.0*val_;
    val_ *= val_;
  }

  inline void Inverse(){
    val_ = 1.0/val_;
    grad_ *= -std::pow(val_,2);
  }

  friend Eigen::Matrix<FMVADvar,Eigen::Dynamic,1> independent(const Eigen::VectorXd x);
  friend std::ostream& operator<< (std::ostream& out, const FMVADvar& obj);
  friend bool operator==(const FMVADvar& lhs, const FMVADvar& rhs);
};
} // stop namespace early to specialize Eigen NumTraits
/*
 * Traits to make the variable work well with Eigen
 */
namespace Eigen {
template<> struct NumTraits<FMVAD::FMVADvar >
  : NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
  typedef FMVAD::FMVADvar Real;
  typedef FMVAD::FMVADvar NonInteger;
  typedef FMVAD::FMVADvar Nested;

  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    ReadCost = 1,
    AddCost = 3,
    MulCost = 3
  };
};
} // end namespace Eigen
namespace FMVAD{ // then restart namespace
/*
 * Friend functions
 *
 */
Eigen::Matrix<FMVADvar,Eigen::Dynamic,1> independent(const Eigen::VectorXd x){
  _FMVAD_global_dim_ = x.size();
  Eigen::Matrix<FMVADvar,Eigen::Dynamic,1> out(x.size());
  for(int i=0;i<x.size();i++){
    out.coeffRef(i).val_ = x.coeff(i);
    out.coeffRef(i).grad_.setConstant(x.size(),0.0);
    out.coeffRef(i).grad_.coeffRef(i) = 1.0;
  }
  return(out);
}

std::ostream& operator<< (std::ostream& out, const FMVADvar& obj){
  out << "(val: " << obj.val_ << " grad: " << obj.grad_.transpose() << ")";
  return(out);
}

bool operator==(const FMVADvar& lhs, const FMVADvar& rhs){
  return(lhs.val_==rhs.val_);
}

/*
 * operations where lhs is double
 */
inline FMVADvar operator*(const double lhs,const FMVADvar& rhs){return rhs*lhs;}
inline FMVADvar operator+(const double lhs,const FMVADvar& rhs){return rhs+lhs;}
inline FMVADvar operator-(const double lhs,const FMVADvar& rhs){return lhs+(-rhs);}
inline FMVADvar operator/(const double lhs,const FMVADvar& rhs){
  FMVADvar ret(rhs);
  ret.Inverse();
  ret*=lhs;
  return(ret);
}
/*
 * Overloads of common mathematical functions
 *
 */

inline FMVADvar sqrt(const FMVADvar& arg){
  FMVADvar ret(arg);
  ret.Sqrt();
  return(ret);
}
inline FMVADvar square(const FMVADvar& arg){
  FMVADvar ret(arg);
  ret.Square();
  return(ret);
}

/*
 * Type defs mixing Eigen and this type
 *
 */

typedef Eigen::Matrix<FMVADvar,Eigen::Dynamic,1> VectorXa;
typedef Eigen::Matrix<FMVADvar,Eigen::Dynamic,Eigen::Dynamic> MatrixXa;

/*
 * various utilities
 */
inline double asDouble(const FMVADvar& v){return v.val();}
inline Eigen::VectorXd asDouble(const VectorXa& m){
  return m.unaryExpr([](const FMVADvar& v){return v.val();});
}
inline Eigen::MatrixXd asDouble(const MatrixXa& m){
  return m.unaryExpr([](const FMVADvar& v){return v.val();});
}



} // namespace FMVAD





double spectralRadius(const Eigen::MatrixXd& A){
  Eigen::EigenSolver<Eigen::MatrixXd> sp(A);
  return(sp.eigenvalues().array().abs().maxCoeff());
}
FMVAD::FMVADvar spectralRadius(const FMVAD::MatrixXa& A){
  Eigen::EigenSolver<Eigen::MatrixXd> sp(asDouble(A));
  Eigen::PartialPivLU< Eigen::MatrixXcd > lu(sp.eigenvectors());

  std::cout << "eigenvalues : \n" << sp.eigenvalues() << std::endl;

  int j;
  double rho = sp.eigenvalues().array().abs().maxCoeff(&j);
  std::complex<double> maxev = sp.eigenvalues().coeff(j);
  FMVAD::FMVADvar ret(rho);
  std::cout << "selected at index " << j << std::endl;
  Eigen::MatrixXcd deigtmp(1,1);
  for(size_t i=0;i<FMVAD::_FMVAD_global_dim_;i++){
    deigtmp = lu.solve(
      A.unaryExpr([&i](const FMVAD::FMVADvar& v){return std::complex<double>(v.grad(i),0.0);})).row(j)
    *sp.eigenvectors().col(j);
    ret.__setGrad(i,(maxev.real()*deigtmp.coeff(0.0).real() + maxev.imag()*deigtmp.coeff(0,0).imag())/rho);
  }
  return(ret);
}



#endif
