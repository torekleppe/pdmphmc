#ifndef _CUBICPOLYROOTSININTERVAL_HPP_
#define _CUBICPOLYROOTSININTERVAL_HPP_
#include "cubicPolyRoots.hpp"

namespace numUtils{

// returns smallest real roots of the polynomial a*x^3 + b*x^2 + c*x + d in
// the interval [lb,ub]
// if no such real root exists, an arbitrary number > ub is returned

inline double smallestCubicPolyRootsInInterval(const double lb,
                                               const double ub,
                                               const double a,
                                               const double b,
                                               const double c,
                                               const double d){
  Eigen::VectorXd allRoots = cubicPolyRoots(a,b,c,d);
  double ret = ub + 1.0;
  double x,dev;
  double numFac = std::max(1.0,std::max(std::fabs(lb),std::abs(ub)));
  bool converged;
  for(size_t i=0;i<allRoots.size();i++){
    if(allRoots.coeff(i)>0.99*lb && allRoots.coeff(i)<1.01*ub){
      // check the root
      x = allRoots.coeff(i);
      converged = false;
      for(size_t i=0;i<10;i++){
        dev = d + (c + (a * x + b) * x) * x;
        //std::cout << " dev : " << dev << std::endl;
        if(std::fabs(dev)<1.0e-15*numFac){
          converged = true;
          break;
        }
        x -= dev/(c + (3.0 * a * x + 2.0 * b) * x);
      }
      if(!converged){
        std::cout << "failed convergence in smallestCubicPolyRootsInInterval, dev :" << dev << std::endl;
      }
      if(x>=lb && x<=ub && x<ret) ret = x;
    }
  }
  return(ret);
}


}//namespace
#endif
