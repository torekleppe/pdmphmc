#ifndef _CUBICPOLYROOTS_HPP_
#define _CUBICPOLYROOTS_HPP_
#include <cmath>


namespace numUtils{


#ifndef CUBICPOLYROOTS_TOL
#define CUBICPOLYROOTS_TOL 1.0e-12
#endif

//#define CUBICPOLYROOTS_TEST
void _test_cubicpolyroots_(const double a,
                           const double b,
                           const double c,
                           const double d,
                           const Eigen::VectorXd roots){
  std::cout << a << "*x^3 + " << b << "*x^2 + " << c << "*x  + " << d << std::endl;
  std::cout << "# (real) roots : " << roots.size() << std::endl;
  double x,dev;
  for(size_t i=0;i<roots.size();i++){
    x = roots.coeff(i);
    dev = a*x*x*x + b*x*x + c*x + d;
    std::cout << "root # " << i+1 << " : x = " << x << " dev : " << dev << std::endl;
  }

}
/*
inline double _refine_cubicPolyRoots(const double x0,
                                     const double a,
                                    const double b,
                                    const double c,
                                    const double d){
  double x = x0;
  double dev = d + (c + (a * x + b) * x) * x;
  if(std::fabs(dev)>1.0e-13*std::max(std::fabs(x),1.0)){
  x = x - dev/(c + (3 * a * x + 2 * b) * x);
  dev = d + (c + (a * x + b) * x) * x;
  x = x - dev/(c + (3 * a * x + 2 * b) * x);
  dev = d + (c + (a * x + b) * x) * x;

  if(std::fabs(dev)>1.0e-14){
    std::cout << a << " " << b << " " << c << " " << d << std::endl;
    std::cout << "refine: dev = " << dev << " x = " << x << std::endl;
  }
  }
  return x;
}
*/
// roots of the polynomial a*x^3 + b*x^2 + c*x + d
inline Eigen::VectorXd cubicPolyRoots(const double a,
                                      const double b,
                                      const double c,
                                      const double d) {
  if(!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c) || !std::isfinite(d)){
    std::cout << "bad arguments into cubicPolyRoots" << std::endl;
    return(Eigen::VectorXd(0));
  }

  if(std::fabs(a)<CUBICPOLYROOTS_TOL){
    // polynomial is at most quadratic
    if(std::fabs(b)<CUBICPOLYROOTS_TOL){
      // polynomial is at most linear
      if(std::fabs(c)<CUBICPOLYROOTS_TOL){
        // constant polynomial
        Eigen::VectorXd ret(0);
#ifdef CUBICPOLYROOTS_TEST
        _test_cubicpolyroots_(a,b,c,d,ret);
#endif
        return(ret);
      } else{
        // linear polynomial
        Eigen::VectorXd ret(1);
        ret.coeffRef(0) = -d/c;
#ifdef CUBICPOLYROOTS_TEST
        _test_cubicpolyroots_(a,b,c,d,ret);
#endif
        return(ret);
      }
    } else {
      // quadratic polynomial
      double dev = std::pow(c,2) - 4.0*b*d;
      if(dev>=0.0){
        double q = -0.5*(c+((b>=0.0) ? 1.0 : -1.0 )*std::sqrt(dev));
        Eigen::VectorXd ret(2);
        ret.coeffRef(0) = q/b;
        ret.coeffRef(1) = d/q;
#ifdef CUBICPOLYROOTS_TEST
        _test_cubicpolyroots_(a,b,c,d,ret);
#endif
        return(ret);
      } else {
        Eigen::VectorXd ret(0);
#ifdef CUBICPOLYROOTS_TEST
        _test_cubicpolyroots_(a,b,c,d,ret);
#endif
        return(ret);
      }
    }



  } else {
    // cubic polynomial
    double aa = b/a;
    double bb = c/a;
    double cc = d/a;
    double Q = (std::pow(aa,2) - 3.0*bb)/9.0;
    double R = (2.0*std::pow(aa,3) - 9.0*aa*bb + 27.0*cc)/54.0;
    double Rsq = std::pow(R,2);
    double Qcube = std::pow(Q,3);
    if(Rsq<Qcube){
      // three distinct roots
      double T = std::acos(R/std::sqrt(Qcube))/3.0;
      double fac = -2.0*std::sqrt(Q);
      double t1 = aa/3.0;
      Eigen::VectorXd ret(3);
      ret.coeffRef(0) = fac*std::cos(T) - t1;
      ret.coeffRef(1) = fac*std::cos(T+2.09439510239319549)-t1;
      ret.coeffRef(2) = fac*std::cos(T-2.09439510239319549)-t1;
#ifdef CUBICPOLYROOTS_TEST
      _test_cubicpolyroots_(a,b,c,d,ret);
#endif
      return(ret);
    } else {
      // single root
      double A = ((R>=0.0) ? -1.0 : 1.0)*std::cbrt(std::fabs(R) + std::sqrt(Rsq-Qcube));
      double B = (std::fabs(A)>CUBICPOLYROOTS_TOL) ? Q/A : 0.0;
      Eigen::VectorXd ret(1);
      ret.coeffRef(0) = A+B-aa/3.0;
#ifdef CUBICPOLYROOTS_TEST
      _test_cubicpolyroots_(a,b,c,d,ret);
#endif
      return(ret);
    }


  }
}

}

#endif
