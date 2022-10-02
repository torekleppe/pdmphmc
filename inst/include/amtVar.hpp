#ifndef _AMTVAR_HPP_
#define _AMTVAR_HPP_

#include <Eigen/Dense>
#include "sparseVec.hpp"
#include "amt.hpp"


namespace amt{



class amtVar{
private:
  inline double __dblVal(const double arg) const {return arg;}
  inline double __dblVal(const stan::math::var& arg) const {return arg.val();}
public:
  stan::math::var val_;
  sparseVec::sparseVec<stan::math::var> Jac_;

  amtVar(){}
  amtVar(const double fixedVal) : val_(fixedVal) {}
  amtVar(const stan::math::var& val,
         const size_t ind) : val_(val), Jac_(ind) {}
  amtVar(const amtVar& cpFrom) : val_(cpFrom.val_), Jac_(cpFrom.Jac_) {}

  amtVar& operator=(const amtVar& rhs){
    val_ = rhs.val_;
    Jac_ = rhs.Jac_;
    return *this;
  }


  void independent(const stan::math::var& value, const size_t ind){
    val_ = value;
    Jac_.setUnit(ind);
  }

  inline stan::math::var value() const {return val_;}
  //inline double val() const {return val_.val();}
  inline size_t firstInd() const {return Jac_.firstInd();}
  inline bool isBasic() const{ return Jac_.isUnit();}
  inline const sparseVec::sparseVec<stan::math::var>& JacRef() const {return Jac_;}


  amtVar& operator=(const double rhs){
    val_ = rhs;
    Jac_.clear();
    return *this;
  }


  inline amtVar operator-() const {
    amtVar ret(*this);
    ret.val_ = -ret.val_;
    ret.Jac_ = -ret.Jac_;
    return(ret);
  }

  inline void operator+=(const amtVar& rhs){
    val_+= rhs.val_;
    Jac_ += rhs.Jac_;
  }
  inline void operator-=(const amtVar& rhs){
    val_ -= rhs.val_;
    Jac_ -= rhs.Jac_;
  }
  inline void operator*=(const amtVar& rhs){
    Jac_.scal(rhs.val_);
    Jac_.axpy(val_,rhs.Jac_);
    val_ *= rhs.val_;
  }
  inline void operator/=(const amtVar& rhs){
    Jac_.scal(1.0/rhs.val_);
    Jac_.axpy(-val_/pow(rhs.val_,2),rhs.Jac_);
    val_ /= rhs.val_;
  }
  inline void operator+=(const double rhs){
    val_ += rhs;
  }
  inline void operator-=(const double rhs){
    val_ -= rhs;
  }
  inline void operator*=(const double rhs){
    Jac_.scal(rhs);
    val_ *= rhs;
  }
  inline void operator/=(const double rhs){
    Jac_.scal(1.0/rhs);
    val_ /= rhs;
  }

  inline void Inverse(){
    Jac_.scal(-1.0/pow(val_,2));
    val_ = 1.0/val_;
  }
  inline void Exp(){
    val_ = exp(val_);
    Jac_.scal(val_);
  }

  inline void Log(){
    Jac_.scal(inv(val_));
    val_ = log(val_);
  }


  inline void Sqrt(){
    val_ = sqrt(val_);
    Jac_.scal(0.5/val_);
  }

  inline void Square(){
    Jac_.scal(2.0*val_);
    val_ *= val_;
  }

  inline void Logit(){
    Jac_.scal(inv(val_*(1.0-val_)));
    val_ = logit(val_);
  }

  inline void Inv_logit(){
    stan::math::var t;
    if(__dblVal(val_)>0.0){
      t = exp(-val_);
      val_ = 1.0/(1.0+t);
      Jac_.scal(square(val_)*t);
    } else {
      t = exp(val_);
      val_ = t/(1.0+t);
      Jac_.scal(val_/(1.0+t));
    }
  }

  inline void dump() const {
    std::cout << "value : " << val_ << std::endl;
    std::cout << "Jacobian : " << std::endl;
    Jac_.dump();
  }
  friend std::ostream& operator<< (std::ostream& out, const amtVar& obj);
};

/*
 * Out of place operations
 */

inline amtVar operator+(const amtVar& lhs,
                        const amtVar& rhs){
  amtVar ret(lhs);
  ret+=rhs;
  return(ret);
}

inline amtVar operator+(const amtVar& lhs,
                        const double rhs){
  amtVar ret(lhs);
  ret+=rhs;
  return(ret);
}

inline amtVar operator+(const double lhs,
                        const amtVar& rhs){
  amtVar ret(rhs);
  ret+=lhs;
  return(ret);
}


inline amtVar operator-(const amtVar& lhs,
                        const amtVar& rhs){
  amtVar ret(lhs);
  ret-=rhs;
  return(ret);
}

inline amtVar operator-(const amtVar& lhs,
                        const double rhs){
  amtVar ret(lhs);
  ret-=rhs;
  return(ret);
}

inline amtVar operator-(const double lhs,
                        const amtVar& rhs){
  amtVar ret(-rhs);
  ret+=lhs;
  return(ret);
}


inline amtVar operator*(const amtVar& lhs,
                        const amtVar& rhs){
  amtVar ret(lhs);
  ret*=rhs;
  return(ret);
}

inline amtVar operator*(const amtVar& lhs,
                        const double rhs){
  amtVar ret(lhs);
  ret*=rhs;
  return(ret);
}

inline amtVar operator*(const double lhs,
                        const amtVar& rhs){
  amtVar ret(rhs);
  ret*=lhs;
  return(ret);
}


inline amtVar operator/(const amtVar& lhs,
                        const amtVar& rhs){
  amtVar ret(lhs);
  ret/=rhs;
  return(ret);
}

inline amtVar operator/(const amtVar& lhs,
                        const double rhs){
  amtVar ret(lhs);
  ret/=rhs;
  return(ret);
}

inline amtVar operator/(const double lhs,
                        const amtVar& rhs){
  amtVar ret(rhs);
  ret.Inverse();
  ret*=lhs;
  return(ret);
}

/*
 * General single argument functions
 */


inline amtVar exp(const amtVar& arg){
  amtVar ret(arg);
  ret.Exp();
  return(ret);
}

inline amtVar log(const amtVar& arg){
  amtVar ret(arg);
  ret.Log();
  return(ret);
}

inline amtVar sqrt(const amtVar& arg){
  amtVar ret(arg);
  ret.Sqrt();
  return(ret);
}


inline amtVar square(const amtVar& arg){
  amtVar ret(arg);
  ret.Square();
  return(ret);
}

inline amtVar logit(const amtVar& arg){
  amtVar ret(arg);
  ret.Logit();
  return(ret);
}

inline amtVar inv_logit(const amtVar& arg){
  amtVar ret(arg);
  ret.Inv_logit();
  return(ret);
}

// multiargument basic functions


// lazy implementation, possibly improve!
inline amtVar pow(const amtVar& x, const amtVar& y){
  return(exp(y*log(x)));
}
inline amtVar pow(const amtVar& x, const double y){
  return(exp(y*log(x)));
}
inline amtVar pow(const amtVar& x, const int y){
  return(exp(static_cast<double>(y)*log(x)));
}
inline amtVar pow(const double x, const amtVar& y){
  return(exp(y*log(x)));
}



std::ostream& operator<< (std::ostream& out, const amtVar& obj){
  out << "amtVar, val: " << obj.val_.val() << " jac:" << obj.Jac_;
  return out;
}

/*inline Eigen::VectorXd amtDoubleValue(const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& x){
 // size_t n = x.size();
  Eigen::VectorXd ret;
  //for(size_t i=0;i<x.size();i++) ret(i) = ::doubleValue(x.coeff(i).val_);
  return(ret);
}*/

/*
 * overloads to get value from ADtypes
 */
/*
inline double doubleValue(const double var){return var;}
inline double doubleValue(const stan::math::var& var){return var.val();}
inline Eigen::VectorXd doubleValue(const Eigen::Ref<const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> >& var){
  Eigen::VectorXd ret(var.size());
  for(size_t i=0;i<var.size();i++) ret.coeffRef(i) = var.coeff(i).val();
  return(ret);
}
*/

}// namespace

namespace Eigen {

template<> struct NumTraits<amt::amtVar >
  : NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
  typedef amt::amtVar Real;
  typedef amt::amtVar NonInteger;
  typedef amt::amtVar Nested;

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

}
/*
//inline double doubleValue(const amt::amtVar& var){return amt::doubleValue(var);}
inline Eigen::VectorXd doubleValue( const Eigen::Matrix<amt::amtVar,Eigen::Dynamic,1>& var){
  Eigen::VectorXd ret(var.size());
  for(size_t i=0;i<var.size();i++) ret.coeffRef(i) = amt::doubleValue(var.coeff(i));
  return(ret);
}
*/
#endif
