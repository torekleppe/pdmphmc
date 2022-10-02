#ifndef _STABILITYMONITOR_HPP_
#define _STABILITYMONITOR_HPP_

#include <vector>
#include <Eigen/Dense>

class stabilityMonitor{
  size_t n_;
  std::vector<double> vals_;
  Eigen::VectorXd regGrid_;
  double xssq_;


public:
  stabilityMonitor(const size_t n) : n_(n) {vals_.empty();}
  void push(const double val){vals_.push_back(val);}
  bool hasSufficientData(){return(vals_.size()>=n_);}
  bool isStable_regression(const bool print=false){
    if(regGrid_.size()<n_){
      regGrid_.setLinSpaced(n_,-1.0,1.0);
      xssq_ = regGrid_.array().pow(2).sum();
    }
    double beta = 0.0;
    double alpha = 0.0;
    double ssq = 0.0;
    double tobs;
    size_t first = vals_.size()-n_;

    for(size_t i = 0; i<n_;i++) {
      beta += regGrid_.coeff(i)*vals_[first+i];
      alpha += vals_[first+i];
      if(print) std::cout << vals_[first+i] << std::endl;
    }
    beta /= xssq_;
    alpha /= static_cast<double>(n_);


    for(size_t i = 0; i<n_;i++) ssq += std::pow(vals_[first+i]-alpha-beta*regGrid_.coeff(i),2);
    ssq /= static_cast<double>(n_-2);
    tobs = beta/std::max(1.0e-14,std::sqrt(ssq/xssq_));

    if(print){
    std::cout << "alpha : " << alpha << std::endl;
    std::cout << "beta : " << beta << std::endl;
    std::cout << "tobs : " << tobs << std::endl;
    }
    return(std::fabs(tobs)<2.0);
  }


};

#endif

