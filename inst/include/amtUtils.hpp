#ifndef _AMTUTILS_HPP_
#define _AMTUTILS_HPP_
#include <limits>
namespace amt{

Eigen::Matrix<size_t,Eigen::Dynamic,1> basicInds(const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& x){
    Eigen::Matrix<size_t,Eigen::Dynamic,1> binds(x.size());
    for(size_t i=0;i<x.size();i++){
      binds.coeffRef(i) = (x.coeff(i).isBasic()) ? x.coeff(i).firstInd() : std::numeric_limits<size_t>::max();
    }
    return(binds);
  }
} //namespace
#endif
