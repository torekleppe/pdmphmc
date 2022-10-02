#ifndef _AD_OVERLOADS_HPP_
#define _AD_OVERLOADS_HPP_

#include <Eigen/Dense>



/*
 * overloads to get value from ADtypes
 */


inline double doubleValue(const double var){return var;}
inline double doubleValue(const stan::math::var& var){return var.val();}
//inline double doubleValue(const amt::amtVar& var){}

inline Eigen::VectorXd doubleValue(const Eigen::VectorXd &var){return var;}
inline Eigen::VectorXd doubleValue(const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> &var){
  Eigen::VectorXd ret(var.size());
  for(int i=0;i<var.size();i++) ret.coeffRef(i) = var.coeff(i).val();
  return ret;
}
inline Eigen::MatrixXd doubleValue(const Eigen::MatrixXd &var){return var;}
inline Eigen::MatrixXd doubleValue(const Eigen::Matrix<stan::math::var,Eigen::Dynamic,Eigen::Dynamic> &var){
  Eigen::MatrixXd ret(var.rows(),var.cols());
  for(int j=0;j<var.cols();j++) for(int i=0;i<var.rows();i++) ret.coeffRef(i,j) = var.coeff(i,j).val();
  return ret;
}

#endif
