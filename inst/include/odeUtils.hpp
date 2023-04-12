#ifndef _ODEUTILS_HPP_
#define _ODEUTILS_HPP_

#include "BaseClasses.hpp"

/*
 * Unified specification of special root functions (linear, L1 etc)
 *
 */
struct specialRootSpec{
  bool allowRepeatedRoots_;
  // dense linear root function linRootJac_*y + linRootConst_ = 0 where y is the ode state
  Eigen::MatrixXd linRootJac_;
  Eigen::VectorXd linRootConst_;

  // sparse linear root function
  compressedRowMatrix<double> spLinRootJac_;
  Eigen::VectorXd spLinRootConst_;

  // linear L1-norm
  std::vector<compressedRowMatrix<double> > spLinL1RootJac_;
  std::vector<Eigen::VectorXd> spLinL1RootConst_;
  std::vector<double> spLinL1RootRhs_;

  // linear L2-norm
  std::vector<compressedRowMatrix<double> > spLinL2RootJac_;
  std::vector<Eigen::VectorXd> spLinL2RootConst_;
  std::vector<double> spLinL2RootRhs_;

  // linear - generic function
  std::vector<compressedRowMatrix<double> > spLinFRootJac_;
  std::vector<Eigen::VectorXd> spLinFRootConst_;
  std::vector<constraintFunctor*> spLinFRootFun_;

  specialRootSpec() : allowRepeatedRoots_(false) {}
  inline bool nonTrivial() const {return linRootJac_.rows()>0 || spLinRootJac_.rows()>0 || spLinL1RootJac_.size()>0 || spLinL2RootJac_.size()>0 || spLinFRootJac_.size()>0;}
  friend std::ostream& operator<< (std::ostream& out, const specialRootSpec& obj);
};
std::ostream& operator<< (std::ostream& out, const specialRootSpec& obj){
  out << "specialRootSpec,\n# linRoots: " << obj.linRootJac_.rows() << "\n";
  out << "# sparseLinRoots:" << obj.spLinRootJac_.rows() << "\n";
  out << "Lin Jacobian: \n" << obj.linRootJac_ << "\nConstant\n" << obj.linRootConst_ << std::endl;
  out << "SparseLin Jacobian: \n" << obj.spLinRootJac_ << "\nConstant\n" << obj.spLinRootConst_ << std::endl;
  out << "# sparse linear L1 constraints: " << obj.spLinL1RootConst_.size() << std::endl;
  out << "# sparse linear L2 constraints: " << obj.spLinL2RootConst_.size() << std::endl;
  out << "# sparse linear Fun constraints: " << obj.spLinFRootConst_.size() << std::endl;
  return out;
}



/*
 * Unified output for the rootSolver methods of the step
 *
 */
struct rootInfo{
  double rootTime_;
  int rootType_; // 0=non-linear, 1=linear, 2=linear sparse, 3=linear sparse L1, 4=linear sparse L2, 5= linear sparse + fun
  int rootDim_;
  Eigen::VectorXd auxInfo_;
  rootInfo() : rootTime_(1.0e100),rootType_(-1), rootDim_(-1) {}
  rootInfo(const double rootTime, const int rootType, const int rootDim) : rootTime_(rootTime), rootType_(rootType), rootDim_(rootDim) {}
  rootInfo(const double rootTime, const int rootType, const int rootDim, const Eigen::VectorXd& aux) : rootTime_(rootTime), rootType_(rootType), rootDim_(rootDim), auxInfo_(aux)  {}
  void earliest(const rootInfo& other){
    if(other.rootTime_<rootTime_){
      rootTime_ = other.rootTime_;
      rootType_ = other.rootType_;
      rootDim_ = other.rootDim_;
      auxInfo_ = other.auxInfo_;
    }
  }
  friend std::ostream& operator<< (std::ostream& out, const rootInfo& obj);

};

std::ostream& operator<< (std::ostream& out, const rootInfo& obj){
  out << "rootInfo, time: " << obj.rootTime_ << " type: " << obj.rootType_ << " dim: " << obj.rootDim_;
  return out;
}




/*
 * representing the state of an ode-system, unifying interface
 * for both first and second order ode-solvers
 */
class odeState{
public:
  Eigen::VectorXd y;
  Eigen::VectorXd ydot;
  Eigen::VectorXd M;
  odeState(){}
  odeState(const Eigen::VectorXd &ode1_y) : y(ode1_y) {}
  odeState(const Eigen::VectorXd &ode2_y,
           const Eigen::VectorXd &ode2_ydot) : y(ode2_y), ydot(ode2_ydot) {}
  odeState(const Eigen::VectorXd &ode2_y,
           const Eigen::VectorXd &ode2_ydot,
           const Eigen::VectorXd &ode2_M) : y(ode2_y), ydot(ode2_ydot), M(ode2_M) {}
  void copyTo(odeState &to) const { // copy to existing object
    to.y=y;
    to.ydot=ydot;
    to.M = M;
  }
};



#endif
