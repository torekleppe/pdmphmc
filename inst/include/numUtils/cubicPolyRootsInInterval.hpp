#ifndef _CUBICPOLYROOTSININTERVAL_HPP_
#define _CUBICPOLYROOTSININTERVAL_HPP_
#include "cubicPolyRoots.hpp"

namespace numUtils{

// returns smallest real roots of the polynomial a*x^3 + b*x^2 + c*x + d in
// the interval [lb,ub]
// if no such real root exists, an arbitrary number > ub is returned
// if excludeZeroRoot=true, it is implicitly assumed that d=0.0 and that the root x=0 is disregarded
inline double smallestCubicPolyRootsInInterval(const double lb,
                                               const double ub,
                                               const double a,
                                               const double b,
                                               const double c,
                                               const double d,
                                               bool excludeZeroRoot=false){
  //std::cout << a << " " << b << " " << c << " " << d << std::endl;
  Eigen::VectorXd allRoots;
  if(excludeZeroRoot){
    allRoots = quadraticPolyRoots(a,b,c);
  } else {
   allRoots = cubicPolyRoots(a,b,c,d);
  }
  //std::cout << allRoots << std::endl;
  double ret = ub + 1.0;
  double x,dev;
  double numFac = std::max(1.0,std::max(std::fabs(lb),std::abs(ub)));
  bool converged;
  for(size_t i=0;i<allRoots.size();i++){
    if(allRoots.coeff(i)>lb-0.01*numFac && allRoots.coeff(i)<ub+0.01*numFac){
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




// returns all real roots of the polynomials a*x^3 + b*x^2 + c*x + d in
// the interval [lb,ub]

inline int sequenceOfCubicPolyRootsInInterval(const double lb,
                                              const double ub,
                                              const Eigen::VectorXd& a,
                                              const Eigen::VectorXd& b,
                                              const Eigen::VectorXd& c,
                                              const Eigen::VectorXd& d,
                                              Eigen::VectorXd& rootsOut,
                                              Eigen::VectorXi& whichDimOut){
  Eigen::VectorXd rii;
  Eigen::VectorXd roots(3*a.size());
  Eigen::VectorXi whichDim(3*a.size());
  double x,dev;
  double numFac = std::max(1.0,std::max(std::fabs(lb),std::abs(ub)));
  bool converged = false;
  int rootCount = 0;
  for(size_t i=0;i<a.size();i++){
    rii = cubicPolyRoots(a.coeff(i),b.coeff(i),c.coeff(i),d.coeff(i));
    for(size_t j=0;j<rii.size();j++){
      if(rii.coeff(j)>lb-0.01*numFac && rii.coeff(j)<ub+0.01*numFac){
        x = rii.coeff(j);
        //std::cout << "x orig " << x << std::endl;
        converged = false;
        for(size_t ii=0;ii<10;ii++){
          dev = d.coeff(i) + (c.coeff(i) + (a.coeff(i) * x + b.coeff(i)) * x) * x;
          //std::cout << dev << std::endl;
          if(std::fabs(dev)<1.0e-15*numFac){
            converged = true;
            break;
          }
          x -= dev/(c.coeff(i) + (3.0 * a.coeff(i) * x + 2.0 * b.coeff(i)) * x);
        }
        if(!converged){
          std::cout << "failed convergence in sequenceOfCubicPolyRootsInInterval, dev :" << dev << std::endl;
        }
        if(x>=lb && x<=ub){
          roots.coeffRef(rootCount) = x;
          whichDim.coeffRef(rootCount) = i;
          rootCount++;
        }
      }
    }
  }

  if(rootsOut.size()!=rootCount) rootsOut.resize(rootCount);
  if(whichDimOut.size()!=rootCount) whichDimOut.resize(rootCount);

  if(rootCount>1){
    // sort in the case of muliple roots
    Eigen::VectorXi perm(rootCount);
    perm.setLinSpaced(rootCount,0,rootCount-1);
    std::sort(perm.data(),perm.data()+perm.size(),
              [&](const int& a, const int& b){
                return(roots.coeff(a)<roots.coeff(b));
              });
    for(size_t i=0;i<rootCount;i++){
      rootsOut.coeffRef(i) = roots.coeff(perm.coeff(i));
      whichDimOut.coeffRef(i) = whichDim.coeff(perm.coeff(i));
    }
  } else if(rootCount==1){
    rootsOut.coeffRef(0) = roots.coeff(0);
    whichDimOut.coeffRef(0) = whichDim.coeff(0);
  }
  return(rootCount);
}




}//namespace
#endif
