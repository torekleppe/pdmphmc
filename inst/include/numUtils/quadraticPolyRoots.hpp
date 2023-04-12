#ifndef _QUADRATICPOLYROOTS_HPP_
#define _QUADRATICPOLYROOTS_HPP_
#include <cmath>

namespace numUtils{

#define _QUADRATICPOLY_TOL_ 1.0e-14

// returns the real roots in x of a*x^2 + b*x + c
Eigen::VectorXd quadraticPolyRoots(const double a,
                                   const double b,
                                   const double c){
  if(std::fabs(a)<_QUADRATICPOLY_TOL_){
    // at most linear
    if(std::fabs(b)<_QUADRATICPOLY_TOL_){
      // constant
      Eigen::VectorXd ret(0);
      return(ret);
    } else {
      // linear
      Eigen::VectorXd ret(1);
      ret.coeffRef(0) = -c/b;
      return(ret);
    }
  } else {
    // quadratic
    double t = b*b - 4.0*a*c;
    if(t<0.0){ // no real roots
      Eigen::VectorXd ret(0);
      return(ret);
    }
    double q = -0.5*((b>=0.0) ? b + sqrt(t) : b-sqrt(t));
    Eigen::VectorXd ret(2);
    ret.coeffRef(0) = q/a;
    ret.coeffRef(1) = c/q;
    return(ret);
  }

}


}
#endif
