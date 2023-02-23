#ifndef _FMVAD_HPP_
#define _FMVAD_HPP_

namespace FMVAD{
static std::size_t _FMVAD_GRAD_DIM_ = 0;

template <class T>
class FMVADExpression{
public:
  inline const T& cast() const {return static_cast<const T&>(*this);}
  inline double val() const {return cast().val();}
  inline const Eigen::VectorXd& __grad() const {return cast().__grad();}
};

class FMVADvar : public FMVADExpression<FMVADvar> {
  double val_;
  Eigen::VectorXd grad_;

public:
  FMVADvar() : val_(0.0) {grad_.setConstant(_FMVAD_GRAD_DIM_,0.0);}
  FMVADvar(const double v) : val_(v) {grad_.setConstant(_FMVAD_GRAD_DIM_,0.0);}
  template <class T>
  FMVADvar(const FMVADExpression<T>& rhs) : val_(rhs.val()), grad_(rhs.__grad()) {}
  inline double val() const {return val_;}
  inline const Eigen::VectorXd& __grad() const {return grad_;}
  inline double grad(const std::size_t i) const {return grad_.coeff(i);}
  inline void __setIndep(const double v, const std::size_t which){
    val_=v;
    grad_.setZero();
    grad_.coeffRef(which) = 1.0;
  }
  inline void __setGrad(const size_t which, const double gradVal){
    grad_.coeffRef(which) = gradVal;
  }
  inline void __setVal(const double val){val_ = val;}
  FMVADvar& operator=(const double rhs){
    val_ = rhs;
    grad_.setConstant(_FMVAD_GRAD_DIM_,0.0);
    return *this;
  }
  FMVADvar& operator=(const FMVADvar& rhs){
    //std::cout << "operator= basic" << std::endl;
    grad_ = rhs.grad_;
    val_ = rhs.val_;
    return *this;
  }
  template <class T>
  FMVADvar& operator=(const FMVADExpression<T>& rhs){
    //std::cout << "operator= expression" << std::endl;
    grad_ = rhs.__grad();
    val_ = rhs.val();
    return *this;
  }
  inline void operator*=(const double rhs){
    grad_*=rhs;
    val_*=rhs;
  }

  void print() const {std::cout << "val: " << val_ << " grad: " << grad_.transpose() << std::endl;}
  friend std::ostream& operator<< (std::ostream& out, const FMVADvar& obj);
};
std::ostream& operator<< (std::ostream& out, const FMVADvar& obj){
  out << "FMVADvar: val: " << obj.val_ << " grad: " << obj.grad_.transpose();
  return(out);
}


} // end namespace
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


namespace FMVAD{



/*
 * The common math operations
 *
 */

template <class T>
class PlusDouble : public FMVADExpression<PlusDouble<T> > {
  double val_;
  Eigen::VectorXd grad_;
public:
  PlusDouble(const double arg1,
               const FMVADExpression<T>& arg2) : val_(arg1+arg2.val()), grad_(arg2.__grad()) {}
  inline double val() const {return val_;}
  inline const Eigen::VectorXd& __grad() const {return grad_;}
};

template <class T>
class UnaryMinus : public FMVADExpression<UnaryMinus<T> > {
  double val_;
  Eigen::VectorXd grad_;
public:
  UnaryMinus(const FMVADExpression<T>& arg) : val_(-arg.val()), grad_(-arg.__grad()) {}
  inline double val() const {return val_;}
  inline const Eigen::VectorXd& __grad() const {return grad_;}
};


template<class T>
inline PlusDouble<T> operator+(const double lhs,
                                        const FMVADExpression<T>& rhs){
  return PlusDouble<T>(lhs,rhs);
}
template<class T>
inline PlusDouble<T> operator+(const FMVADExpression<T>& lhs,
                               const double rhs){
  return PlusDouble<T>(rhs,lhs);
}

template <class T>
inline PlusDouble<T> operator-(const FMVADExpression<T>& lhs,
                               const double rhs){
  return PlusDouble<T>(-rhs,lhs);
}

template<class T>
inline PlusDouble<UnaryMinus<T> > operator-(const double lhs,
                                            const FMVADExpression<T>& rhs){
  return PlusDouble<UnaryMinus<T> >(lhs,UnaryMinus(rhs));
}





/*
template<>
inline PlusDouble<FMVADvar> operator+(const double lhs,
                                 const FMVADExpression<FMVADvar>& rhs){
  std::cout << "mixed operator+" << std::endl;
  return PlusDouble<FMVADvar>(lhs,rhs);
}
*/


template <class T1,class T2>
class Plus : public FMVADExpression<Plus<T1,T2> > {
  double val_;
  Eigen::VectorXd grad_;
public:
  Plus(const FMVADExpression<T1>& arg1,
       const FMVADExpression<T2>& arg2) : val_(arg1.val()+arg2.val()), grad_(arg1.__grad()+arg2.__grad()) {}
  inline double val() const {return val_;}
  inline const Eigen::VectorXd& __grad() const {return grad_;}
};

template <class T1,class T2>
inline Plus<T1,T2> operator+(const FMVADExpression<T1>& arg1,
                             const FMVADExpression<T2>& arg2 ){return Plus<T1,T2>(arg1,arg2);}
/*
template <class T1>
inline Plus<T1,FMVADvar> operator+(const FMVADExpression<T1>& arg1,
                                 const FMVADvar& arg2 ){return Plus<T1,FMVADvar>(arg1,arg2);}

template <class T2>
inline Plus<FMVADvar,T2> operator+(const FMVADvar& arg1,
                                 const FMVADExpression<T2>& arg2){return Plus<FMVADvar,T2>(arg1,arg2);}

*/
/*
inline FMVADvar operator+(const FMVADvar& arg1,
                        const FMVADvar& arg2 ){
  std::cout << "v-v-plus" << std::endl;
  std::cout << "arg 1: " << arg1 << std::endl;
  return FMVADvar(Plus<FMVADvar,FMVADvar>(arg1,arg2));
}
*/

template <class T1,class T2>
class Minus : public FMVADExpression<Minus<T1,T2> > {
  double val_;
  Eigen::VectorXd grad_;
public:
  Minus(const FMVADExpression<T1>& arg1,
       const FMVADExpression<T2>& arg2) : val_(arg1.val()-arg2.val()), grad_(arg1.__grad()-arg2.__grad()) {}
  inline double val() const {return val_;}
  inline const Eigen::VectorXd& __grad() const {return grad_;}
};

template <class T1,class T2>
inline Minus<T1,T2> operator-(const FMVADExpression<T1>& arg1,
                             const FMVADExpression<T2>& arg2 ){return Minus<T1,T2>(arg1,arg2);}
/*
template <class T1>
inline Minus<T1,FMVADvar> operator-(const FMVADExpression<T1>& arg1,
                                   const FMVADvar& arg2 ){return Minus<T1,FMVADvar>(arg1,arg2);}

template <class T2>
inline Minus<FMVADvar,T2> operator-(const FMVADvar& arg1,
                                   const FMVADExpression<T2>& arg2){return Minus<FMVADvar,T2>(arg1,arg2);}

inline FMVADvar operator-(const FMVADvar& arg1,
                          const FMVADvar& arg2 ){
  return FMVADvar(Minus<FMVADvar,FMVADvar>(arg1,arg2));
}



*/


template <class T1,class T2>
class Multiply : public FMVADExpression<Multiply<T1,T2> > {
  double val_;
  Eigen::VectorXd grad_;
public:
  Multiply(const FMVADExpression<T1>& arg1,
           const FMVADExpression<T2>& arg2) : val_(arg1.val()*arg2.val()), grad_(arg2.val()*arg1.__grad()+arg1.val()*arg2.__grad()) {}
  inline double val() const {return val_;}
  inline const Eigen::VectorXd& __grad() const {return grad_;}
};

template <class T1,class T2>
class Divide : public FMVADExpression<Divide<T1,T2> > {
  double inv2_;
  double val_;
  Eigen::VectorXd grad_;
public:
  Divide(const FMVADExpression<T1>& arg1,
           const FMVADExpression<T2>& arg2) : inv2_(1.0/arg2.val()),val_(inv2_*arg1.val()), grad_(inv2_*arg1.__grad() - (inv2_*val_)*arg2.__grad()) {}
  inline double val() const {return val_;}
  inline const Eigen::VectorXd& __grad() const {return grad_;}
};




template <class T1,class T2>
inline Multiply<T1,T2> operator*(const FMVADExpression<T1>& arg1,
                                 const FMVADExpression<T2>& arg2 ){
  return Multiply<T1,T2>(arg1,arg2);}

template <class T1,class T2>
inline Divide<T1,T2> operator/(const FMVADExpression<T1>& arg1,
                               const FMVADExpression<T2>& arg2 ){
  return Divide<T1,T2>(arg1,arg2);
}



/*
template <class T1>
inline Multiply<T1,FMVADvar> operator*(const FMVADExpression<T1>& arg1,
                                     const FMVADvar& arg2 ){
  return Multiply<T1,FMVADvar>(arg1,arg2);}


template <class T2>
inline Multiply<FMVADvar,T2> operator*(const FMVADvar& arg1,
                                     const FMVADExpression<T2>& arg2){
  return Multiply<FMVADvar,T2>(arg1,arg2);
}
inline FMVADvar operator*(const FMVADvar& arg1,
                        const FMVADvar& arg2 ){
  return FMVADvar(Multiply<FMVADvar,FMVADvar>(arg1,arg2));
}
*/

template <class T>
class MultiplyDouble : public FMVADExpression<T>{
  double val_;
  Eigen::VectorXd grad_;
public:
  MultiplyDouble(const FMVADExpression<T>& arg1,
                 const double arg2) : val_(arg1.val()*arg2), grad_(arg2*arg1.__grad()) {}
  inline double val() const {return val_;}
  inline const Eigen::VectorXd& __grad() const {return grad_;}
};


// computes arg1/arg2 when arg1 is double/inactive
template <class T>
class DivideDouble : public FMVADExpression<DivideDouble<T> > {
  double val_;
  Eigen::VectorXd grad_;
public:
  DivideDouble(const double arg1,
                const FMVADExpression<T>& arg2) : val_(arg1/arg2.val()), grad_((-val_/arg2.val())*arg2.__grad()) {}
  inline double val() const {return val_;}
  inline const Eigen::VectorXd& __grad() const {return grad_;}
};

template <class T>
inline MultiplyDouble<T> operator*(const FMVADExpression<T>& arg1,
                                   const double arg2){
  return MultiplyDouble<T>(arg1,arg2);
}
template <class T>
inline MultiplyDouble<T> operator*(const double arg1,
                                   const FMVADExpression<T>& arg2){
  return MultiplyDouble<T>(arg2,arg1);
}

template <class T>
inline DivideDouble<T> operator/(const double arg1,
                                 const FMVADExpression<T>& arg2){
  return DivideDouble<T>(arg1,arg2);
}

template <class T>
inline MultiplyDouble<T> operator/(const FMVADExpression<T>& arg1,
                                   const double arg2){
  return MultiplyDouble<T>(arg1,1.0/arg2);
}

/*
 * Unary special functions
 *
 */


template <class T>
class Exp : public FMVADExpression<Exp<T> > {
  double val_;
  Eigen::VectorXd grad_;
public:
  Exp(const FMVADExpression<T>& arg) : val_(exp(arg.val())), grad_(val_*arg.__grad()) {}
  inline double val() const {return val_;}
  inline const Eigen::VectorXd& __grad() const {return grad_;}
};
template<class A>
inline Exp<A> exp(const FMVADExpression<A>& arg){return Exp<A>(arg);}

template <class T>
class Log : public FMVADExpression<Log<T> > {
  double val_;
  Eigen::VectorXd grad_;
public:
  Log(const FMVADExpression<T>& arg) : val_(log(arg.val())), grad_((1.0/arg.val())*arg.__grad()) {}
  inline double val() const {return val_;}
  inline const Eigen::VectorXd& __grad() const {return grad_;}
};
template<class A>
inline Log<A> log(const FMVADExpression<A>& arg){return Log<A>(arg);}


template <class T>
class Sqrt : public FMVADExpression<Sqrt<T> > {
  double val_;
  Eigen::VectorXd grad_;
public:
  Sqrt(const FMVADExpression<T>& arg) : val_(sqrt(arg.val())), grad_((0.5/val_)*arg.__grad()) {}
  inline double val() const {return val_;}
  inline const Eigen::VectorXd& __grad() const {return grad_;}
};
template<class A>
inline Sqrt<A> sqrt(const FMVADExpression<A>& arg){return Sqrt<A>(arg);}


template <class T>
class Square : public FMVADExpression<Square<T> > {
  double val_;
  Eigen::VectorXd grad_;
public:
  Square(const FMVADExpression<T>& arg) : val_(pow(arg.val(),2)), grad_((2.0*arg.val())*arg.__grad()) {}
  inline double val() const {return val_;}
  inline const Eigen::VectorXd& __grad() const {return grad_;}
};

inline double square(const double arg){return std::pow(arg,2);}

template <class A>
inline Square<A> square(const FMVADExpression<A>& arg){return Square<A>(arg);}

/*
 * furhter typedefs
 *
 */


typedef Eigen::Matrix<FMVADvar,Eigen::Dynamic,1> VectorXa;
typedef Eigen::Matrix<FMVADvar,Eigen::Dynamic,Eigen::Dynamic> MatrixXa;

/*
 * Utility for starting AD calculations, and extracting gradient information:
 *
 */

VectorXa independent(const Eigen::VectorXd& x_double){
  _FMVAD_GRAD_DIM_ = x_double.size();
  VectorXa ret(_FMVAD_GRAD_DIM_);
  for(size_t i=0;i<_FMVAD_GRAD_DIM_;i++) ret.coeffRef(i).__setIndep(x_double.coeff(i),i);
  return(ret);
}

template<class A>
Eigen::VectorXd gradient(const FMVADExpression<A>& dependent){
  Eigen::VectorXd ret = dependent.__grad();
  return(ret);
}


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

/*
 * overload of common Eigen functions in order to allow mixed type operations
 */

inline VectorXa operator*(const VectorXa& vec, const double scal){
  VectorXa ret = vec;
  for(size_t i=0;i<vec.size();i++) ret.coeffRef(i)*=scal;
  return(ret);
}
inline VectorXa operator*(const double scal, const VectorXa& vec){return vec*scal;}





} // end namespace

#endif
