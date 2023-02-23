#ifndef _FMVAD_SPEC_FUNS_HPP_
#define _FMVAD_SPEC_FUNS_HPP_

#include <Eigen/Dense>
#include "FMVAD.hpp"


namespace FMVAD{

/*
 * Compute the spectral radius of a matrix (based on the EigenSolver in eigen)
 *
 */


double spectralRadius(const Eigen::MatrixXd& A){
  Eigen::EigenSolver<Eigen::MatrixXd> sp(A);
  return(sp.eigenvalues().array().abs().maxCoeff());
}
FMVAD::FMVADvar spectralRadius(const FMVAD::MatrixXa& A){
  Eigen::EigenSolver<Eigen::MatrixXd> sp(asDouble(A));
  Eigen::PartialPivLU< Eigen::MatrixXcd > lu(sp.eigenvectors());
  //std::cout << "eigenvalues : \n" << sp.eigenvalues() << std::endl;
  int j;
  double rho = sp.eigenvalues().array().abs().maxCoeff(&j);
  std::complex<double> maxev = sp.eigenvalues().coeff(j);
  FMVAD::FMVADvar ret(rho);
  //std::cout << "selected at index " << j << std::endl;
  Eigen::MatrixXcd deigtmp(1,1);
  for(size_t i=0;i<_FMVAD_GRAD_DIM_;i++){
    deigtmp = lu.solve(
      A.unaryExpr([&i](const FMVAD::FMVADvar& v){return std::complex<double>(v.grad(i),0.0);})).row(j)
    *sp.eigenvectors().col(j);
    ret.__setGrad(i,(maxev.real()*deigtmp.coeff(0.0).real() + maxev.imag()*deigtmp.coeff(0,0).imag())/rho);
  }
  return(ret);
}


} // end namespace


#endif
