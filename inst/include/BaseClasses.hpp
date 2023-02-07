#ifndef _AMTBASECLASSES_HPP_
#define _AMTBASECLASSES_HPP_



class constraintFunctor{
public:
  virtual double operator()(const Eigen::VectorXd& arg) const = 0;
  virtual double operator()(const Eigen::VectorXd& arg,
                          Eigen::VectorXd& grad) const  = 0;
  virtual inline std::string name() const {return "unnamed constraint functor";}
};


#endif
