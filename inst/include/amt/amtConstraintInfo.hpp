#ifndef _AMTCONSTRAINTINFO_HPP_
#define _AMTCONSTRAINTINFO_HPP_

#include "../rng.hpp"
#include "../compressedRowMatrix.hpp"
#include "../BaseClasses.hpp"

#ifndef _CONSTRAINT_FINDIFF_THRESH_
#define _CONSTRAINT_FINDIFF_THRESH_ 1.0e-4
#endif


namespace amt {

struct constraintInfo{
  int numNonLin_;
  int numLin_;
  int numspLin_;
  int numspLinL1_;
  int numspLinL2_;
  int numspLinF_;
  Eigen::MatrixXd linJac_;
  Eigen::VectorXd linConst_;
  compressedRowMatrix<double> splinJac_;
  Eigen::VectorXd splinConst_;

  std::vector<compressedRowMatrix<double> > splinL1Jac_;
  std::vector<Eigen::VectorXd> splinL1Const_;
  Eigen::VectorXd splinL1Rhs_;

  std::vector<compressedRowMatrix<double> > splinL2Jac_;
  std::vector<Eigen::VectorXd> splinL2Const_;
  Eigen::VectorXd splinL2Rhs_;

  std::vector<compressedRowMatrix<double> > splinFJac_;
  std::vector<Eigen::VectorXd> splinFConst_;
  std::vector<constraintFunctor*> splinFfun_;


  constraintInfo() : numNonLin_(0), numLin_(0), numspLin_(0), numspLinL1_(0), numspLinL2_(0), numspLinF_(0) {}

  template <class modelType>
  bool functorFDtest(const modelType& mdl){
    bool ret = true;
    bool retSingle;
    Eigen::VectorXd evalPoint,evalH,fTorGrad,fdGrad;
    double h,valEval,valhp,valhm,dev;
    for(size_t i=0;i<splinFfun_.size();i++){
      retSingle = true;
      evalPoint = mdl.getSpLinFLhs(i);
      fTorGrad.resize(evalPoint.size());
      fdGrad.resize(evalPoint.size());
      valEval = (*splinFfun_[i])(evalPoint,fTorGrad);
      if(fTorGrad.size()!=evalPoint.size()){
        std::cout << "gradient of functor " << (*splinFfun_[i]).name() << ", used in sparseLinFunConstraint # "  << i+1 << " has incorrect dimension " << std::endl;
        throw(600);
      }

      for(size_t j=0;j<evalPoint.size();j++){
        h = 1.0e-6*std::max(std::abs(evalPoint.coeff(j)),1.0);
        evalH = evalPoint;
        evalH.coeffRef(j)+=h;
        valhp = (*splinFfun_[i])(evalH);
        evalH = evalPoint;
        evalH.coeffRef(j)-=h;
        valhm = (*splinFfun_[i])(evalH);
        fdGrad.coeffRef(j) = (0.5*(valhp-valhm))/h;
        dev = std::fabs(fdGrad.coeff(j)-fTorGrad.coeff(j))/
          (_CONSTRAINT_FINDIFF_THRESH_*std::max(1.0,std::fabs(fTorGrad.coeff(j))));
        //std::cout << "dev: " << dev << std::endl;
        if(dev>1.0) retSingle=false;
      }
      if(!retSingle){
        ret = false;
        std::cout << "gradient of functor " << (*splinFfun_[i]).name() << ", used in sparseLinFunConstraint # " << i+1 << " seems incorrect" << std::endl;
        std::cout << "finite difference gradient:\n" << fdGrad << std::endl;
        std::cout << "gradient from functor:\n" << fTorGrad << std::endl;
      }
    }
    return(ret);
  }

  template<class targetType,class tensorType>
  bool compute(targetType& t,
               const Eigen::VectorXd q0){
    tensorType mtd;
    amt::amtModel< stan::math::var, tensorType, false> model(mtd);

    model.setIndependent(q0);
    t(model);

    numLin_ = model.numLinConstr();
    numNonLin_ = model.numNonlinConstr();
    numspLin_ = model.numSpLinConstr();
    numspLinL1_ = model.numSpLinL1Constr();
    numspLinL2_ = model.numSpLinL2Constr();
    numspLinF_ = model.numSpLinFConstr();


    if(numLin_>0 || numspLin_>0 || numspLinL1_>0 || numspLinL2_>0 || numspLinF_>0){
      // compute representation of linear constraints as linJac_*q + linConst_
      if(numLin_>0){
        model.linConstraintJacobian(linJac_);
        linConst_ = model.getLinConstraint() - linJac_*q0;
        if(!(linJac_.array().isFinite().all() && linConst_.array().isFinite().all())){
          std::cout << "numerical problems with linear constraints at initial point" << std::endl;
          return(false);
        }
      }
      if(numspLin_>0){
        splinJac_.setCols(q0.size());
        model.splinConstraintJacobian(splinJac_);
        splinConst_ = model.getSpLinConstraint() - splinJac_*q0;
        if(!(splinJac_.isAllFinite() && splinConst_.array().isFinite().all())){
          std::cout << "numerical problems with sparse linear constraints at initial point" << std::endl;
          return(false);
        }
      }

      if(numspLinL1_>0){
        for(int i=0;i<numspLinL1_;i++){
          splinL1Jac_.emplace_back();
          splinL1Const_.emplace_back();
          splinL1Jac_[i].setCols(q0.size());
          model.splinRep(splinL1Jac_[i],splinL1Const_[i],i,0);
          splinL1Rhs_ = model.getSpLinL1Rhs();
          if(!(splinL1Jac_[i].isAllFinite() && splinL1Const_[i].array().isFinite().all())){
            std::cout << "numerical problems with sparse linear L1 constraints at initial point" << std::endl;
            return(false);
          }
        }
      }

      if(numspLinL2_>0){
        for(int i=0;i<numspLinL2_;i++){
          splinL2Jac_.emplace_back();
          splinL2Const_.emplace_back();
          splinL2Jac_[i].setCols(q0.size());
          model.splinRep(splinL2Jac_[i],splinL2Const_[i],i,1);
          splinL2Rhs_ = model.getSpLinL2Rhs();
          if(!(splinL2Jac_[i].isAllFinite() && splinL2Const_[i].array().isFinite().all())){
            std::cout << "numerical problems with sparse linear L2 constraints at initial point" << std::endl;
            return(false);
          }
        }
      }

      if(numspLinF_>0){
        splinFfun_ = model.splinFfun();
        for(int i=0;i<numspLinF_;i++){
          splinFJac_.emplace_back();
          splinFConst_.emplace_back();
          splinFJac_[i].setCols(q0.size());
          model.splinRep(splinFJac_[i],splinFConst_[i],i,2);

          if(!(splinFJac_[i].isAllFinite() && splinFConst_[i].array().isFinite().all())){
            std::cout << "numerical problems with sparse linear Fun constraints at initial point" << std::endl;
            return(false);
          }
        }
        std::cout << "checking gradients provided in functors: ";
        bool FDout = functorFDtest(model);

        if(! FDout){
          std::cout << "failed!!!" << std::endl;
          return false;
        } else {
          std::cout << "OK" << std::endl;
        }

      }






      // check if the linear constraints are indeed linear
      Eigen::VectorXd rdir(q0.size()),tq(q0.size()),dev,spdev,sptmp,predConstr;
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
        spdev = model.getSpLinConstraint();

        if(std::isfinite(tval) && dev.array().isFinite().all() && spdev.array().isFinite().all()){
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
      if(numLin_>0){
        predConstr = linJac_*tq + linConst_;
        dev -= predConstr;
        for(size_t i=0;i<dev.size();i++){
          if(std::fabs(dev.coeff(i))>1.0e-14*std::max(1.0,predConstr.coeff(i))){
            std::cout << "prediction test:" << std::endl;
            std::cout << "linConstraint() # " << i+1 << " does not appear to be linear in PARAMETERs."
                      << " Consider nonlinConstraint() instead." << std::endl;
            return(false);
          }
        }
      }
      if(numspLin_>0){
        predConstr = splinJac_*tq + splinConst_;
        spdev -= predConstr;
        for(size_t i=0;i<spdev.size();i++){
          if(std::fabs(spdev.coeff(i))>1.0e-14*std::max(1.0,predConstr.coeff(i))){
            std::cout << "prediction test:" << std::endl;
            std::cout << "sparseLinConstraint() # " << i+1 << " does not appear to be linear in PARAMETERs."
                      << " Consider nonlinConstraint() instead." << std::endl;
            return(false);
          }
        }
      }



      if(numspLinL1_>0){
        for(size_t i=0;i<splinL1Jac_.size();i++){
          predConstr = splinL1Jac_[i]*tq + splinL1Const_[i];
          sptmp = predConstr - model.getSpLinL1Lhs(i);
          for(size_t j=0;j<sptmp.size();j++){
          if(std::fabs(sptmp.coeff(j))>1.0e-14*std::max(1.0,predConstr.coeff(j))){
            std::cout << "prediction test:" << std::endl;
            std::cout << "first argument of sparseLinL1Constraint() # " << i+1 << " does not appear to be linear in PARAMETERs."
                      << " Consider nonlinConstraint() instead." << std::endl;
            return(false);
          }
          }
        }
      }

      if(numspLinL2_>0){
        for(size_t i=0;i<splinL2Jac_.size();i++){
          predConstr = splinL2Jac_[i]*tq + splinL2Const_[i];
          sptmp = predConstr - model.getSpLinL2Lhs(i);
          for(size_t j=0;j<sptmp.size();j++){
            if(std::fabs(sptmp.coeff(j))>1.0e-14*std::max(1.0,predConstr.coeff(j))){
              std::cout << "prediction test:" << std::endl;
              std::cout << "first argument of sparseLinL2Constraint() # " << i+1 << " does not appear to be linear in PARAMETERs."
                        << " Consider nonlinConstraint() instead." << std::endl;
              return(false);
            }
          }
        }
      }

      if(numspLinF_>0){
        for(size_t i=0;i<splinFJac_.size();i++){
          predConstr = splinFJac_[i]*tq + splinFConst_[i];
          sptmp = predConstr - model.getSpLinFLhs(i);
          for(size_t j=0;j<sptmp.size();j++){
            if(std::fabs(sptmp.coeff(j))>1.0e-14*std::max(1.0,predConstr.coeff(j))){
              std::cout << "prediction test:" << std::endl;
              std::cout << "first argument of sparseLinFunConstraint() # " << i+1 << " does not appear to be linear in PARAMETERs."
                        << " Consider nonlinConstraint() instead." << std::endl;
              return(false);
            }
          }
        }
        std::cout << "checking gradients provided in functors once more: ";
        bool FDout = functorFDtest(model);

        if(! FDout){
          std::cout << "failed!!!" << std::endl;
          return false;
        } else {
          std::cout << "OK" << std::endl;
        }
      }
      // second check: recompute Jacobian (only done for basic cases)
      if(numLin_>0){
        Eigen::MatrixXd refJac;
        model.linConstraintJacobian(refJac);
        refJac -= linJac_;

        for(size_t i=0;i<linJac_.rows();i++){
          for(size_t j=0;j<linJac_.cols();j++){
            if(std::fabs(refJac.coeff(i,j))>1.0e-14*std::max(1.0,linJac_.coeff(i,j))){
              std::cout << "Jacobian test:" << std::endl;
              std::cout << "linConstraint() # " << i+1 << " does not appear to be linear in PARAMETERs."
                        << " Consider nonlinConstraint() instead." << std::endl;
              return(false);
            }
          }
        }
      }
      if(numspLin_>0){
        compressedRowMatrix<double> refSpJac;
        refSpJac.setCols(q0.size());
        model.splinConstraintJacobian(refSpJac);
        if(! splinJac_.isEqualTo(refSpJac)){
          std::cout << "Jacobian test:" << std::endl;
          std::cout << "at least one of the sparseLinConstraint()s does not appear to be linear in PARAMETERs."
                    << " Consider nonlinConstraint() instead." << std::endl;
          return(false);
        }
      }
    } // end linear constraints compute and check







    return true;
  }
  inline bool nonTrivial() const {return numNonLin_>0 || numLin_>0 || numspLin_>0;}
  inline bool nonTrivialSpecial() const {return numLin_>0 || numspLin_>0;}

  void toJSON(jsonOut &outf) const {
    outf.push("numNonLinConstr",numNonLin_);
    outf.push("numLinConstr",numLin_);
    outf.push("numSparseLinConstr",numspLin_);
    outf.push("linConstraintJac",linJac_);
    outf.push("linConstraintConst",linConst_);
    Eigen::VectorXi i,j;
    Eigen::VectorXd val;
    splinJac_.toTriplet(i,j,val);
    outf.push("sparseLinConstrJac_i",i);
    outf.push("sparseLinConstrJac_j",j);
    outf.push("sparseLinConstrJac_val",val);
    outf.push("sparseLinConstraintConst",splinConst_);
  }

  friend std::ostream& operator<< (std::ostream& out, const constraintInfo& obj);
};
std::ostream& operator<< (std::ostream& out, const constraintInfo& obj){
  out << "constraintInfo, numNonLin:" << obj.numNonLin_ << " numLin:" << obj.numLin_ << " numSpLin:" << obj.numspLin_ << "\n";
  if(obj.numLin_>0){
  out << "linJac:\n" << obj.linJac_ << "\n" << "linConst:\n" << obj.linConst_ << std::endl;
  }
  if(obj.numspLin_>0){
  out << "splinJac:\n" << obj.splinJac_ << "splinConst:\n" << obj.splinConst_ << std::endl;
  }
  if(obj.numspLinL1_>0){
  for(size_t i=0;i<obj.splinL1Jac_.size();i++){
    out << "splinL1Jac # " << i << " :\n" << obj.splinL1Jac_[i] << "splinL1Const \n" << obj.splinL1Const_[i] << std::endl;
  }
  }
  if(obj.numspLinL2_>0){
  for(size_t i=0;i<obj.splinL2Jac_.size();i++){
    out << "splinL2Jac # " << i << " :\n" << obj.splinL2Jac_[i] << "splinL2Const \n" << obj.splinL2Const_[i] << std::endl;
  }
  }
  if(obj.numspLinF_){
    for(size_t i=0;i<obj.splinFfun_.size();i++){
      out << "splinF # " << i << " functor name: " << (*obj.splinFfun_[i]).name() << std::endl;
      out << "Jac:\n"  << obj.splinFJac_[i] << "Const:\n" << obj.splinFConst_[i] << std::endl;
    }
  }
  return out;
}


} // namespace




#endif
