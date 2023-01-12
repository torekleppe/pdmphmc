#ifndef _AMTCONSTRAINTINFO_HPP_
#define _AMTCONSTRAINTINFO_HPP_

#include "../rng.hpp"

namespace amt {

struct constraintInfo{
  int numNonLin_;
  int numLin_;
  Eigen::MatrixXd linJac_;
  Eigen::VectorXd linConst_;


  constraintInfo() : numNonLin_(0), numLin_(0) {}

  template<class targetType,class tensorType>
  bool compute(targetType& t,
               const Eigen::VectorXd q0){
    tensorType mtd;
    amt::amtModel< stan::math::var, tensorType, false> model(mtd);

    model.setIndependent(q0);
    t(model);

    numLin_ = model.numLinConstr();
    numNonLin_ = model.numNonlinConstr();

    if(numLin_>0){
      // compute representation of linear constraints as linJac_*q + linConst_
      model.linConstraintJacobian(linJac_);
      linConst_ = model.getLinConstraint() - linJac_*q0;

      if(!(linJac_.array().isFinite().all() && linConst_.array().isFinite().all())){
        std::cout << "numerical problems with linear constraints at initial point" << std::endl;
        return(false);
      }


      // check if the linear constraints are indeed linear
      Eigen::VectorXd rdir(q0.size()),tq(q0.size()),dev,predConstr;
      rng rr(123233443);
      rr.rnorm(rdir);
      double tval,alpha = 1.0;
      bool iterationDone = false;
      for(size_t j=0;j<20;j++){
        tq = q0 + alpha*rdir;
        model.setIndependent(tq);
        t(model);
        tval = model.getTargetDouble();
        dev = model.getLinConstraint();
        if(std::isfinite(tval) && dev.array().isFinite().all()){
          std::cout << "new eval OK, alpha = " << alpha << std::endl;
          iterationDone = true;
          break;
        }
        alpha*=0.5;
      }

      if(!iterationDone){
        std::cout << "could not evaluate close to initial point, please provide another initial point" << std::endl;
        return(false);
      }


      // first check
      predConstr = linJac_*tq + linConst_;
      dev -= predConstr;
      for(size_t i=0;i<dev.size();i++){
        if(std::fabs(dev.coeff(i))>1.0e-14*std::max(1.0,predConstr.coeff(i))){
          std::cout << "prediction test:" << std::endl;
          std::cout << "linConstrain() # " << i+1 << " does not appear to be linear in PARAMETERs."
                    << " Consider nonlinConstraint() instead." << std::endl;
          return(false);
        }
      }
      // second check: recompute Jacobian
      Eigen::MatrixXd refJac;
      model.linConstraintJacobian(refJac);
      refJac -= linJac_;

      for(size_t i=0;i<linJac_.rows();i++){
        for(size_t j=0;j<linJac_.cols();j++){
          if(std::fabs(refJac.coeff(i,j))>1.0e-14*std::max(1.0,linJac_.coeff(i,j))){
            std::cout << "Jacobian test:" << std::endl;
            std::cout << "linConstrain() # " << i+1 << " does not appear to be linear in PARAMETERs."
                      << " Consider nonlinConstraint() instead." << std::endl;
            return(false);
          }
        }
      }
    } // end linear constraints compute and check

    return true;
  }
  inline bool nonTrivial() const {return numNonLin_>0 || numLin_>0;}
  inline bool nonTrivialSpecial() const {return numLin_>0;}

  void toJSON(jsonOut &outf) const {
    outf.push("linConstraintJac",linJac_);
    outf.push("linConstraintConst",linConst_);
  }

  friend std::ostream& operator<< (std::ostream& out, const constraintInfo& obj);
};
std::ostream& operator<< (std::ostream& out, const constraintInfo& obj){
  out << "constraintInfo, numNonLin: " << obj.numNonLin_ << " numLin: " << obj.numLin_ << "\n";
  out << "linJac:\n" << obj.linJac_ << "\n" << "linConst:\n" << obj.linConst_;
  return out;
}


}




#endif
