#ifndef _ODEUTILS_HPP_
#define _ODEUTILS_HPP_


/*
 * Unified specification of special root functions (linear, L1 etc)
 *
 */
struct specialRootSpec{
  // dense linear root function linRootJac_*y + linRootConst_ = 0 where y is the ode state
  Eigen::MatrixXd linRootJac_;
  Eigen::VectorXd linRootConst_;

  // sparse linear root function (todo)


  // linear L1-norm (todo)

  specialRootSpec(){}
  inline bool nonTrivial() const {return linRootJac_.rows()>0;}
  friend std::ostream& operator<< (std::ostream& out, const specialRootSpec& obj);
};
std::ostream& operator<< (std::ostream& out, const specialRootSpec& obj){
  out << "specialRootSpec,\n# linRoots: " << obj.linRootJac_.rows() << "\n";
  out << "Jacobian: \n" << obj.linRootJac_ << "\nConstant\n" << obj.linRootConst_ << std::endl;
  return out;
}



/*
 * Unified output for the rootSolver methods of the step
 *
 */
struct rootInfo{
  double rootTime_;
  int rootType_; // 0=non-linear, 1=linear
  int rootDim_;
  rootInfo() : rootTime_(1.0e100),rootDim_(-1) {}
  rootInfo(double rootTime, int rootType, int rootDim) : rootTime_(rootTime), rootType_(rootType), rootDim_(rootDim) {}
  void earliest(const rootInfo& other){
    if(other.rootTime_<rootTime_){
      rootTime_ = other.rootTime_;
      rootType_ = other.rootType_;
      rootDim_ = other.rootDim_;
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
