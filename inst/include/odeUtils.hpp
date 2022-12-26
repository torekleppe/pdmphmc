#ifndef _ODEUTILS_HPP_
#define _ODEUTILS_HPP_


struct rootInfo{
  int rootType_;
  int rootDim_;
};


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
