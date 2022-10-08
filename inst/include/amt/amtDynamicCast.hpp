#ifndef _AMTDYNAMICCAST_HPP_
#define _AMTDYNAMICCAST_HPP_

namespace amt {


inline double asDouble(const amtVar& x){return x.val_.val();}
inline double asDouble(const stan::math::var& x){return x.val();}
inline double asDouble(const double x){return x;}

inline Eigen::VectorXd asDouble(const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& x){
  return(x.unaryExpr([](const amtVar& x){return x.val_.val();}));
}
inline Eigen::VectorXd asDouble(const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& x){
  return(x.unaryExpr([](const stan::math::var& x){return x.val();}));
}
inline Eigen::VectorXd asDouble(const Eigen::VectorXd& x){
  return(x);
}

inline stan::math::var asStanVar(const amtVar& x){return x.val_;}
inline stan::math::var asStanVar(const stan::math::var& x){return x;}
inline double asStanVar(const double x){return x;}

inline Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>
asStanVar(const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& x){
  return(x.unaryExpr([](const amtVar& x){return x.val_;}));
}
inline Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>
asStanVar(const Eigen::Matrix<amtVar,Eigen::Dynamic,1>* x){
  return(x->unaryExpr([](const amtVar& x){return x.val_;}));
}



/*
template <class T>
class asDouble{
public:
  asDouble(const T& arg){
    throw std::runtime_error("asDouble: unknown type");
  }
};

template<>
class asDouble<double>{
  double ret_;
public:
  asDouble(const double arg) : ret_(arg) {}
  operator double() const {return ret_;}
};
*/
}
#endif
