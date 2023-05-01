#ifndef _AMTMODEL_HPP_
#define _AMTMODEL_HPP_
#include <vector>
#include <string>
#include <iostream>
namespace amt{

#define _SELF_PREC_WARNING_ std::cout << "not basic variable, cannot add self precision" << std::endl;

template <class varType>
void dumpVar(const varType& arg){
  if constexpr (std::is_same_v<double, varType>){
    std::cout << " double, value = " << arg << std::endl;
  } else if constexpr (std::is_same_v<stan::math::var, varType>){
    std::cout << " stan::math::var, value = " << arg.val() << std::endl;
  } else if constexpr (std::is_same_v<amtVar, varType>){
    std::cout << "amtVar " << std::endl;
    arg.dump();
  }
}

inline double numericValue(const double arg){return arg;}
inline stan::math::var numericValue(const stan::math::var& arg){return arg;}
inline stan::math::var numericValue(const amtVar& arg){return arg.value();}






template <class varType, class tensorType, bool storeNames>
class amtModel{
  std::vector<std::string> parNames_;
  std::vector<size_t> parFrom_;
  std::vector<size_t> parTo_;
  size_t parCount_;

  std::vector<size_t> parType_; //(0=scalar,1=vector,2=matrix)
  std::vector<std::vector<size_t> > parDims_;
  const std::vector<std::string> parTypeNames_ = {"scalar","vector","matrix"};

  tensorType* tenPtr_;
  Eigen::Matrix<varType,Eigen::Dynamic,1> par_;


  // typename amtTargetType<varType>::type target_;
  stan::math::var target_;

  std::vector<double> defaultVals_;
  std::vector<double> defaultGens_;
  std::vector<std::string> generatedNames_;
  std::vector<size_t> genFrom_;
  std::vector<size_t> genTo_;
  Eigen::VectorXd generated_;
  size_t generatedCount_;
  size_t emptyNameCount_;

  size_t linConstrCount_;
  size_t splinConstrCount_;
  size_t splinL1ConstrCount_;
  size_t splinL2ConstrCount_;
  size_t splinFConstrCount_;
  size_t nonLinConstrCount_;
  Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> linConstr_;

  Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> splinConstr_;

  std::vector<Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> > splinL1lhs_;
  Eigen::VectorXd splinL1rhs_;

  std::vector<Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> > splinL2lhs_;
  Eigen::VectorXd splinL2rhs_;

  std::vector<Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> > splinFlhs_;
  std::vector<constraintFunctor*> splinFfun_;

  Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> nonLinConstr_;

  inline void reset(){
    if constexpr (storeNames){
      parNames_.clear();
      parFrom_.clear();
      parTo_.clear();
      parType_.clear();
      emptyNameCount_ = 0;
    }
    target_ = 0.0;
    parCount_ = 0;
    generatedCount_ = 0;
    linConstrCount_ = 0;
    splinConstrCount_ = 0;
    splinL1ConstrCount_ = 0;
    splinL2ConstrCount_ = 0;
    splinFConstrCount_ = 0;
    nonLinConstrCount_ = 0;
  }
  void expandParNames(){
    epn_.clear();
    for(size_t i=0;i<parNames_.size();i++){
      if(parType_[i]==0){
        epn_.push_back(parNames_[i]);
      } else if(parType_[i]==1) {
        size_t c = 1;
        for(size_t j=parFrom_[i];j<=parTo_[i];j++){
          epn_.push_back(parNames_[i]+"["+std::to_string(c)+"]");
          c++;
        }
      } else if(parType_[i]==2){
        for(size_t jj=1;jj<=parDims_[i][1];jj++){
          for(size_t ii=1;ii<=parDims_[i][0];ii++){
            epn_.push_back(parNames_[i]+"["+std::to_string(ii)+","+std::to_string(jj)+"]");
          }
        }
      } else {
        throw std::runtime_error("amt::amtModel: bad parameter storage type");
      }
    }
  }
  void expandGeneratedNames(){
    egn_.clear();
    for(size_t i=0;i<generatedNames_.size();i++){
      if(genFrom_[i]==genTo_[i]){
        egn_.push_back(generatedNames_[i]);
      } else {
        size_t c = 1;
        for(size_t j=genFrom_[i];j<=genTo_[i];j++){
          egn_.push_back(generatedNames_[i]+"["+std::to_string(c)+"]");
          c++;
        }
      }
    }
  }

public:

  // expanded parameter/generated names
  std::vector<std::string> epn_;
  std::vector<std::string> egn_;


  amtModel() :  parCount_(0), tenPtr_(NULL), target_(0.0), generatedCount_(0), emptyNameCount_(0), linConstrCount_(0), splinConstrCount_(0), splinL1ConstrCount_(0), splinL2ConstrCount_(0), splinFConstrCount_(0), nonLinConstrCount_(0) {}
  amtModel(tensorType& tensor) : parCount_(0), tenPtr_(&tensor), target_(0.0), generatedCount_(0), emptyNameCount_(0), linConstrCount_(0), splinConstrCount_(0), splinL1ConstrCount_(0), splinL2ConstrCount_(0), splinFConstrCount_(0), nonLinConstrCount_(0) {}

  inline size_t dim() const {return parCount_;}
  inline size_t dimGen() const {return generatedCount_;}
  inline size_t numLinConstr() const {return linConstrCount_;}
  inline size_t numSpLinConstr() const {return splinConstrCount_;}
  inline size_t numSpLinL1Constr() const {return splinL1ConstrCount_;}
  inline size_t numSpLinL2Constr() const {return splinL2ConstrCount_;}
  inline size_t numSpLinFConstr() const {return splinFConstrCount_;}
  inline size_t numNonlinConstr() const {return nonLinConstrCount_;}
  inline void finalize(){
    expandParNames();
    expandGeneratedNames();
  }
  //inline constraintInfo constrInfo() const {return constraintInfo(nonLinConstrCount_,linConstrCount_);}


  std::vector<std::string> storeColNames(const Eigen::VectorXi& storePars){
    std::vector<std::string> ret;
    for(size_t i=0;i<storePars.size();i++) ret.push_back(epn_[storePars.coeff(i)]);
    ret.insert(ret.end(),egn_.begin(),egn_.end());
    return(ret);
  }

  inline void setTenPtr(tensorType& tensor){ tenPtr_ = &tensor;}
  void setIndependent(const Eigen::VectorXd &dblPar,
                      const size_t generatedSize=0){
    reset();
    if(par_.size()!=dblPar.size()) par_.resize(dblPar.size());
    if(generated_.size()!=generatedSize) generated_.resize(generatedSize);
    if constexpr (std::is_same_v<double, varType>) {
      par_ = dblPar;
    } else if constexpr (std::is_same_v<stan::math::var, varType>){
      for(size_t i=0;i<par_.size();i++) par_.coeffRef(i) = dblPar.coeff(i);
    } else if constexpr (std::is_same_v<amtVar, varType>){
      for(size_t i=0;i<par_.size();i++) par_.coeffRef(i).independent(dblPar.coeff(i),i);
    }
  }

  void setIndependent(const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& adPar,
                      const size_t generatedSize=0){
    reset();
    if(par_.size()!=adPar.size()) par_.resize(adPar.size());
    if(generated_.size()!=generatedSize) generated_.resize(generatedSize);
    if constexpr (std::is_same_v<double, varType>) {
      std::cout << "setIndependent - not implemented yet" << std::endl;
      throw(1);
    } else if constexpr (std::is_same_v<stan::math::var, varType>){
      par_= adPar;
    } else if constexpr (std::is_same_v<amtVar, varType>){
      for(size_t i=0;i<par_.size();i++) par_.coeffRef(i).independent(adPar.coeff(i),i);
    }
  }


  void getMiDiag(json_wrap& ctrl,
                 Eigen::VectorXd& MiDiag){

    MiDiag.setConstant(dim(),-1.0);
    std::vector<std::string> allNames;
    Eigen::VectorXd tmp;
    int parI;
    ctrl.getAllNames("fixedMiDiag",allNames);
    //std::cout << "Mi diag : " << std::endl;

    for(int i=0;i<allNames.size();i++){
      auto r = std::find(parNames_.begin(),parNames_.end(),allNames[i]);

      if(r!= parNames_.end()){
        parI = r-parNames_.begin();

        //std::cout << allNames[i] << std::endl;
        //std::cout << parNames_[parI] << std::endl;

        ctrl.getNumeric("fixedMiDiag",allNames[i],tmp);
        if(tmp.size()==1){
          for(size_t j=parFrom_[parI];j<=parTo_[parI];j++) MiDiag.coeffRef(j) = tmp.coeff(0);
        } else if(tmp.size()==parTo_[parI]-parFrom_[parI]+1){
          for(size_t j=parFrom_[parI];j<=parTo_[parI];j++) MiDiag.coeffRef(j) = tmp.coeff(j-parFrom_[parI]);
        } else {
          std::cout << "fixedMiDiag : bad format of field " << allNames[i] << ", ignored " << std::endl;
        }
      } else {
        std::cout << "fixedMiDiag : " << allNames[i] << " not in model, ignored " << std::endl;
      }
    }
    //std::cout << MiDiag << std::endl;
  }


  void getStorePars(const std::vector<std::string>& storeParsNames,
                    Eigen::VectorXi& storePars){
    std::vector<int> spt;
    bool found;
    for(int i=0;i<storeParsNames.size();i++){
      found = false;
      // try first the basic parameter names
      for(int j=0;j<parNames_.size();j++){
        if(storeParsNames[i].compare(parNames_[j])==0){
          for(int k=parFrom_[j];k<=parTo_[j];k++) spt.push_back(k);
          found = true;
          break;
        }
      }
      // if not a basic name, try the expanded names
      if(!found){
        for(int j=0;j<epn_.size();j++){
          if(storeParsNames[i].compare(epn_[j])==0){
            spt.push_back(j);
            found = true;
            break;
          }
        }
      }
      if(!found){
        std::cout << "WARNING: store.pars name " << storeParsNames[i] << " not a model parameter, ignored" << std::endl;
      }
    }

    // copy indices back to argument

    if(spt.size()>0){
      storePars.resize(spt.size());
      for(size_t ii=0;ii<spt.size();ii++) storePars.coeffRef(ii) = spt[ii];
    } else {
      std::cout << "WARNING: none of the provided store.pars are in model, storing everything" << std::endl;
      storePars.resize(dim());
      for(size_t ii=0;ii<storePars.size();ii++) storePars.coeffRef(ii) = ii;
    }
  }

  inline stan::math::var getTarget() const {return(target_);}
  inline void getGenerated(Eigen::VectorXd& genOut) const {genOut = generated_;}
  inline double getTargetDouble() const {return target_.val(); }
  inline Eigen::VectorXd getLinConstraint() const {
    Eigen::VectorXd ret(linConstrCount_);
    for(size_t i=0;i<linConstrCount_;i++) ret.coeffRef(i) = linConstr_.coeff(i).val();
    return(ret);
  }
  inline Eigen::VectorXd getSpLinConstraint() const {
    Eigen::VectorXd ret(splinConstrCount_);
    for(size_t i=0;i<splinConstrCount_;i++) ret.coeffRef(i) = splinConstr_.coeff(i).val();
    return(ret);
  }

  inline Eigen::VectorXd getSpLinL1Lhs(const int which) const {
    Eigen::VectorXd ret(splinL1lhs_[which].size());
    for(size_t i=0;i<splinL1lhs_[which].size();i++) ret.coeffRef(i) = splinL1lhs_[which].coeff(i).val();
    return(ret);
  }

  inline Eigen::VectorXd getSpLinL2Lhs(const int which) const {
    Eigen::VectorXd ret(splinL2lhs_[which].size());
    for(size_t i=0;i<splinL2lhs_[which].size();i++) ret.coeffRef(i) = splinL2lhs_[which].coeff(i).val();
    return(ret);
  }

  inline Eigen::VectorXd getSpLinFLhs(const int which) const {
    Eigen::VectorXd ret(splinFlhs_[which].size());
    for(size_t i=0;i<splinFlhs_[which].size();i++) ret.coeffRef(i) = splinFlhs_[which].coeff(i).val();
    return(ret);
  }

  inline Eigen::VectorXd getSpLinL1Rhs() const {return splinL1rhs_;}
  inline Eigen::VectorXd getSpLinL2Rhs() const {return splinL2rhs_;}
  inline std::vector<constraintFunctor*> splinFfun() const {return splinFfun_;}

  inline double getNonLinConstraint(const int which) const {return nonLinConstr_.coeff(which).val();}
  inline Eigen::VectorXd getNonLinConstraint() const {
    Eigen::VectorXd ret(nonLinConstr_.size());
    for(size_t i=0;i<nonLinConstr_.size();i++) ret.coeffRef(i) = nonLinConstr_.coeff(i).val();
    return(ret);
  }
  inline double minConstraint() const {
    //std::cout << "amtModel::minConstraint() " << splinL2ConstrCount_ << std::endl;
    double ret = 1.0;
    if(nonLinConstrCount_>0){
      for(size_t i=0;i<nonLinConstrCount_;i++) ret = std::min(ret,nonLinConstr_.coeff(i).val());
    }
    if(linConstrCount_>0){
      for(size_t i=0;i<linConstrCount_;i++) ret = std::min(ret,linConstr_.coeff(i).val());
    }
    if(splinConstrCount_>0){
      for(size_t i=0;i<splinConstrCount_;i++) ret = std::min(ret,splinConstr_.coeff(i).val());
    }
    if(splinL1ConstrCount_>0){
      double tmp;
      for(size_t i=0;i<splinL1ConstrCount_;i++){
        tmp = splinL1rhs_.coeff(i) - asDouble(splinL1lhs_[i]).array().abs().sum();
        ret = std::min(ret,tmp);
      }
    }
    if(splinL2ConstrCount_>0){
      double tmp;
      for(size_t i=0;i<splinL2ConstrCount_;i++){
        //std::cout << splinL2rhs_.coeff(i) << "   " << asDouble(splinL2lhs_[i]).norm() << std::endl;
        tmp = splinL2rhs_.coeff(i) - asDouble(splinL2lhs_[i]).norm();
        ret = std::min(ret,tmp);
      }
    }
    if(splinFConstrCount_>0){
      for(size_t i=0;i<splinFConstrCount_;i++)
      ret = std::min(ret,(*splinFfun_[i])(asDouble(splinFlhs_[i])));
    }

    // add further special constraints here
    return(ret);
  }

  inline bool checkConstraints(){
    if(nonLinConstrCount_<1 && linConstrCount_<1 && splinConstrCount_<1 && splinL1ConstrCount_<1 && splinL2ConstrCount_<1 && splinFConstrCount_<1){
      std::cout << "no constraints" << std::endl;
      return true;
    }
    if(minConstraint()>=0.0){
      //std::cout << "constraints OK" << std::endl;
      return true;
    } else {
      std::cout << "the following constraints are violated" << std::endl;
      for(size_t i=0;i<nonLinConstrCount_;i++){
        if(nonLinConstr_.coeff(i).val()<0.0){
          std::cout << "nonlinConstraint # " << i+1 << " evaluates to " << nonLinConstr_.coeff(i).val() << std::endl;
        }
      }
      for(size_t i=0;i<linConstrCount_;i++){
        if(linConstr_.coeff(i).val()<0.0){
          std::cout << "linConstraint # " << i+1 << " evaluates to " << linConstr_.coeff(i).val() << std::endl;
        }
      }
      for(size_t i=0;i<splinConstrCount_;i++){
        if(splinConstr_.coeff(i).val()<0.0){
          std::cout << "sparselinConstraint # " << i+1 << " evaluates to " << splinConstr_.coeff(i).val() << std::endl;
        }
      }

      double tmp;
      for(size_t i=0;i<splinL1ConstrCount_;i++){
        tmp = splinL1rhs_.coeff(i) - asDouble(splinL1lhs_[i]).array().abs().sum();
        if(tmp<0.0){
          std::cout << "sparselinL1Constraint # " << i+1 << " evaluates to " <<  tmp << std::endl;
        }
      }
      for(size_t i=0;i<splinL2ConstrCount_;i++){
        tmp = splinL2rhs_.coeff(i) - asDouble(splinL2lhs_[i]).norm();
        if(tmp<0.0){
          std::cout << "sparselinL2Constraint # " << i+1 << " evaluates to " <<  tmp << std::endl;
        }
      }
      if(splinFConstrCount_>0){
        for(size_t i=0;i<splinFConstrCount_;i++){
          tmp = (*splinFfun_[i])(asDouble(splinFlhs_[i]));
          if(tmp<0.0){
            std::cout << "sparselinFunConstraint # " << i+1 << " evaluates to " <<  tmp << std::endl;
          }
        }
      }

      return false;
    }
  }


  inline void getTargetGradient(Eigen::VectorXd& grad){
    if(grad.size()!=par_.size()) grad.resize(par_.size());
    if constexpr (std::is_same_v<stan::math::var, varType>){
      target_.grad();
      for(size_t i=0;i<grad.size();i++) grad.coeffRef(i) = par_.coeff(i).adj();
    } else {
      std::cout << "getTargetGradient not implemented for varType!=stan::math::var" << std::endl;
      throw(1);
    }
    stan::math::recover_memory();
  }

  inline void nonLinConstraintGradient(const int whichConstraint,
                                       Eigen::VectorXd& grad){
    if(grad.size()!=par_.size()) grad.resize(par_.size());
    if constexpr (std::is_same_v<stan::math::var, varType>){
      nonLinConstr_.coeffRef(whichConstraint).grad();
      for(size_t i=0;i<grad.size();i++) grad.coeffRef(i) = par_.coeff(i).adj();
    } else {
      std::cout << "getTargetGradient not implemented for varType!=stan::math::var" << std::endl;
      throw(1);
    }
    stan::math::recover_memory();
  }

  inline void linConstraintJacobian(Eigen::MatrixXd& linConstrJac){
    if(linConstrCount_>0) linConstrJac.resize(linConstrCount_,par_.size());
    for(size_t i=0; i<linConstrCount_;i++){
      linConstr_.coeffRef(i).grad();
      for(size_t j=0; j<par_.size();j++) linConstrJac.coeffRef(i,j) = par_.coeff(j).adj();
      stan::math::set_zero_all_adjoints();
    }
  }

  inline void splinConstraintJacobian(compressedRowMatrix<double>& splinConstrJac){
    Eigen::VectorXd tmpRow(par_.size());
    for(size_t i=0; i<splinConstrCount_;i++){
      splinConstr_.coeffRef(i).grad();
      for(size_t j=0; j<par_.size();j++) tmpRow(j) = par_.coeff(j).adj();
      splinConstrJac.pushDenseRow(tmpRow);
      stan::math::set_zero_all_adjoints();
    }
  }

  inline void splinRep(compressedRowMatrix<double>& Jac,
                       Eigen::VectorXd& Const,
                       const int which,
                       const int type){
    // type==0 : splinL1
    // type==1 : splinL2
    Eigen::VectorXd tmpRow(par_.size());
    if(type==0){
      for(size_t i=0;i<splinL1lhs_[which].size();i++){
        splinL1lhs_[which].coeffRef(i).grad();
        for(size_t j=0; j<par_.size();j++) tmpRow(j) = par_.coeff(j).adj();
        Jac.pushDenseRow(tmpRow);
        stan::math::set_zero_all_adjoints();
      }
      Const = asDouble(splinL1lhs_[which]) - Jac*asDouble(par_);
    } else if(type==1){
      for(size_t i=0;i<splinL2lhs_[which].size();i++){
        splinL2lhs_[which].coeffRef(i).grad();
        for(size_t j=0; j<par_.size();j++) tmpRow(j) = par_.coeff(j).adj();
        Jac.pushDenseRow(tmpRow);
        stan::math::set_zero_all_adjoints();
      }
      Const = asDouble(splinL2lhs_[which]) - Jac*asDouble(par_);
    } else if(type==2){
      for(size_t i=0;i<splinFlhs_[which].size();i++){
        splinFlhs_[which].coeffRef(i).grad();
        for(size_t j=0; j<par_.size();j++) tmpRow(j) = par_.coeff(j).adj();
        Jac.pushDenseRow(tmpRow);
        stan::math::set_zero_all_adjoints();
      }
      Const = asDouble(splinFlhs_[which]) - Jac*asDouble(par_);
    } else {
      std::cout << "error in amtModel::splinRep" << std::endl;
      throw(1);
    }
  }




  /*
   inline void checkLinConstraintJacobian(){
   if(linConstrCount_>0){
   double dev;
   for(size_t i=0; i<linConstrCount_;i++){
   linConstr_.coeffRef(i).grad();
   for(size_t j=0; j<par_.size();j++){
   dev = linConstrJac_(j,i)-par_.coeff(j).adj();
   if(fabs(dev)/(1.0e-12 + 1.0e-12*linConstrJac_(j,i))>1.0){
   std::cout << "linConstraint # " << i+1 << " is not linear, use nonlinConstraint() instead" << std::endl;
   throw(765);
   }
   }
   stan::math::set_zero_all_adjoints();
   }
   std::cout << "Linear constraints OK" << std::endl;
   }
   }
   */
  inline void linConstraint(const varType& constr){
    linConstrCount_++;
    if(linConstr_.size()<linConstrCount_) linConstr_.conservativeResize(linConstrCount_);
    if constexpr(std::is_same_v<amtVar,varType>){
      std::cout << "constraints not implemented for varType!=stan::math::var" << std::endl;
      throw(1);
    } else {
      linConstr_.coeffRef(linConstrCount_-1) = constr;
    }
  }

  inline void sparseLinConstraint(const varType& constr){
    splinConstrCount_++;
    if(splinConstr_.size()<splinConstrCount_) splinConstr_.conservativeResize(splinConstrCount_);
    if constexpr(std::is_same_v<amtVar,varType>){
      std::cout << "constraints not implemented for varType!=stan::math::var" << std::endl;
      throw(1);
    } else {
      splinConstr_.coeffRef(splinConstrCount_-1) = constr;
    }
  }

  // constraints of the type ||lhs||_1 <= rhs
  inline void sparseLinL1Constraint(const Eigen::Matrix<varType,Eigen::Dynamic,1>& lhs,
                                    const double rhs){
    if constexpr(std::is_same_v<amtVar,varType>){
      std::cout << "constraints not implemented for varType!=stan::math::var" << std::endl;
      throw(1);
    }
    splinL1ConstrCount_++;
    if(splinL1lhs_.size()<splinL1ConstrCount_){
      splinL1lhs_.push_back(lhs);
      splinL1rhs_.conservativeResize(splinL1ConstrCount_);
      splinL1rhs_.coeffRef(splinL1ConstrCount_-1) = rhs;
    } else {
      splinL1lhs_[splinL1ConstrCount_-1] = lhs;
      splinL1rhs_.coeffRef(splinL1ConstrCount_-1) = rhs;
    }
  }
  // constraints of the type ||lhs||_2 <= rhs
  inline void sparseLinL2Constraint(const Eigen::Matrix<varType,Eigen::Dynamic,1>& lhs,
                                    const double rhs){
    if constexpr(std::is_same_v<amtVar,varType>){
      std::cout << "constraints not implemented for varType!=stan::math::var" << std::endl;
      throw(1);
    }
    splinL2ConstrCount_++;
    if(splinL2lhs_.size()<splinL2ConstrCount_){
      splinL2lhs_.push_back(lhs);
      splinL2rhs_.conservativeResize(splinL2ConstrCount_);
      splinL2rhs_.coeffRef(splinL2ConstrCount_-1) = rhs;
    } else {
      splinL2lhs_[splinL2ConstrCount_-1] = lhs;
      splinL2rhs_.coeffRef(splinL2ConstrCount_-1) = rhs;
    }
  }

  template <class cT>
  inline void sparseLinFunConstraint(const Eigen::Matrix<varType,Eigen::Dynamic,1>& lhs,
                                     cT& constraintFtor){
    if constexpr(std::is_same_v<amtVar,varType>){
      std::cout << "constraints not implemented for varType!=stan::math::var" << std::endl;
      throw(1);
    }
    splinFConstrCount_++;
    if(splinFlhs_.size()<splinFConstrCount_){
      splinFlhs_.push_back(lhs);
      splinFfun_.push_back(&constraintFtor);
    } else {
      splinFlhs_[splinFConstrCount_-1] = lhs;
    }
  }

  inline void nonlinConstraint(const varType& constr){
    nonLinConstrCount_++;
    if(nonLinConstr_.size()<nonLinConstrCount_) nonLinConstr_.conservativeResize(nonLinConstrCount_);
    if constexpr(std::is_same_v<amtVar,varType>){
      std::cout << "constraints not implemented for varType!=stan::math::var" << std::endl;
      throw(1);
    } else {
      nonLinConstr_.coeffRef(nonLinConstrCount_-1) = constr;
    }
  }



  inline varType parameterScalar(const std::string& parName,
                                 const double default_val=0.0){
    if constexpr (storeNames){
      parNames_.push_back(parName);
      parFrom_.push_back(parCount_);
      parTo_.push_back(parCount_);
      parType_.push_back(0);
      parDims_.push_back({1});
    }
    size_t thisInd = parCount_;
    parCount_++;
    if(thisInd<par_.size()){
      return(par_.coeff(thisInd));
    } else {
      defaultVals_.push_back(default_val);
      return(default_val);
    }
  }



  inline Eigen::Matrix<varType,Eigen::Dynamic,1>
    parameterVector(const std::string& parName,
                    const size_t dim,
                    const double default_val=0.0){
      if constexpr (storeNames){
        parNames_.push_back(parName);
        parFrom_.push_back(parCount_);
        parTo_.push_back(parCount_+dim-1);
        parType_.push_back(1);
        parDims_.push_back({dim});
      }
      size_t thisInd = parCount_;
      parCount_+=dim;
      Eigen::Matrix<varType,Eigen::Dynamic,1> ret(dim);
      if(par_.size()<parCount_){
        ret.setConstant(dim,default_val);
        defaultVals_.resize(defaultVals_.size()+dim,default_val);
      } else {
        for(size_t i=0;i<dim;i++) ret.coeffRef(i) = par_.coeff(thisInd+i);
      }
      return(ret);
    }

  inline Eigen::Matrix<varType,Eigen::Dynamic,1>
    parameterVector(const std::string& parName,
                    const size_t dim,
                    const Eigen::VectorXd& default_val){
      if(default_val.size()!=dim){
        throw std::runtime_error("bad dimension of default value in amtModel::parameterVector(...,VectorXd)");
      }
      if constexpr (storeNames){
        parNames_.push_back(parName);
        parFrom_.push_back(parCount_);
        parTo_.push_back(parCount_+dim-1);
        parType_.push_back(1);
        parDims_.push_back({dim});
      }
      size_t thisInd = parCount_;
      parCount_+=dim;
      Eigen::Matrix<varType,Eigen::Dynamic,1> ret(dim);
      if(par_.size()<parCount_){
        for(size_t i=0;i<dim;i++) ret.coeffRef(i) = default_val.coeff(i); //.setConstant(dim,default_val);
        size_t oldSize = defaultVals_.size();
        defaultVals_.resize(oldSize+dim); //,default_val);
        for(size_t i=0;i<dim;i++) defaultVals_[i+oldSize] = default_val.coeff(i);
      } else {
        for(size_t i=0;i<dim;i++) ret.coeffRef(i) = par_.coeff(thisInd+i);
      }
      return(ret);
    }

  inline Eigen::Matrix<varType,Eigen::Dynamic,Eigen::Dynamic>
    parameterMatrix(const std::string& parName,
                    const size_t dim1,
                    const size_t dim2,
                    const double default_val=0.0){
      size_t numElems = dim1*dim2;
      if constexpr (storeNames){
        parNames_.push_back(parName);
        parFrom_.push_back(parCount_);
        parTo_.push_back(parCount_+numElems-1);
        parType_.push_back(2);
        parDims_.push_back({dim1,dim2});
      }
      size_t thisInd = parCount_;
      parCount_+=numElems;

      Eigen::Matrix<varType,Eigen::Dynamic,Eigen::Dynamic> ret(dim1,dim2);
      if(par_.size()<parCount_){
        ret.setConstant(dim1,dim2,default_val);
        defaultVals_.resize(defaultVals_.size()+numElems,default_val);
      } else {
        for(size_t colu = 0; colu < dim2; colu++){
          ret.col(colu) = par_.segment(thisInd+colu*dim1,dim1);
        }
      }
      return(ret);
    }
  //inline void operator+=(const double increment){target_+= increment;}
  inline void operator+=(const stan::math::var& increment){
    if constexpr(std::is_same_v<amtVar,varType>){
      std::cout << "WARNING: incrementing target without *_ld()" << std::endl;
    }
    target_+= increment;
  }
  /*
   inline void operator+=(const amtTripletReturn& increment){
   target_ += increment.lp_;
   for(size_t i=0;i<increment.i_.size();i++){
   tenPtr_->pushScalar(increment.i_[i],
   increment.j_[i],
   increment.tenVals_[i]);
   }
   }
   */
  template <class lpdfType>
  inline void operator+=(const lpdfType& lpdf){
    target_+=lpdf(tenPtr_);
  }


  inline void generated(const double value,
                        const std::string& name = ""){
    if constexpr (storeNames){
      if(name.empty()){
        generatedNames_.push_back("generated_" + std::to_string(emptyNameCount_));
        emptyNameCount_++;
      } else {
        generatedNames_.push_back(name);
      }
      defaultGens_.push_back(value);
      genFrom_.push_back(generatedCount_);
      genTo_.push_back(generatedCount_);
    }
    if(generated_.size()>0){
      generated_.coeffRef(generatedCount_) = value;
    }
    generatedCount_++;
  }

  inline void generated(const Eigen::VectorXd& value,
                        const std::string& name = "vector"){
    if constexpr (storeNames){
      for(size_t i=0;i<value.size();i++){
        generated(value.coeff(i),name+"["+std::to_string(i+1)+"]");
      }
    } else {
      for(size_t i=0;i<value.size();i++){
        generated(value.coeff(i),name);
      }
    }
  }



  inline void generated(const stan::math::var& value,
                        const std::string& name = ""){
    generated(value.val(),name);
  }
  inline void generated(const amtVar& value,
                        const std::string& name = ""){
    generated(value.value().val(),name);
  }

  inline void generated(const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& value,
                        const std::string& name = ""){generated(asDouble(value),name); }
  inline void generated(const Eigen::Matrix<amt::amtVar,Eigen::Dynamic,1>& value,
                        const std::string& name = ""){generated(asDouble(value),name); }


  template <class T>
  inline void generated(const SPDmatrix<T>& P,
                        const std::string& name = "SPDmatrix"){
    if constexpr (storeNames){
      for(size_t j=0;j<P.dim();j++){
        for(size_t i=j;i<P.dim();i++){
          generated(P.coeff_double(i,j),name+"["+std::to_string(i+1)+","+std::to_string(j+1)+"]");
        }
      }
    } else {
      for(size_t j=0;j<P.dim();j++){
        for(size_t i=j;i<P.dim();i++){
          generated(P.coeff_double(i,j),name);
        }
      }
    }
  }

  void showSummary(){
    std::cout << "model contains the following parameters:" << std::endl;
    for(int i=0;i<parNames_.size();i++){
      std::cout << parNames_[i] << "\t storage type: " << parTypeNames_[parType_[i]] << "\t total dimension: " << parTo_[i]-parFrom_[i]+1 << std::endl;
    }
    std::cout << "target : "  << target_ << std::endl;
    std::cout << "total dimension : " << parCount_ << std::endl;
    std::cout << "generated dimension : " << generatedCount_ << std::endl;
    if(defaultVals_.size()>0){
      std::cout << "default values" << std::endl;
      for(int i=0;i<parNames_.size();i++){
        std::cout << parNames_[i] << " : " << std::endl;
        if(epn_.size()==0){
          for(int ii=parFrom_[i];ii<=parTo_[i];ii++) std::cout << defaultVals_[ii] << std::endl;
        } else {
          for(int ii=parFrom_[i];ii<=parTo_[i];ii++){
            std::cout << epn_[ii] << " : " << defaultVals_[ii] << std::endl;
          }
        }
      }
    }
    if(generatedNames_.size()>0){
      std::cout << "generated : " << std::endl;
      for(int i=0;i<generatedNames_.size();i++){
        std::cout << generatedNames_[i] << std::endl;
        if(generated_.size()>0){
          for(int j=genFrom_[i];j<=genTo_[i];j++){
            std::cout << generated_.coeff(j) << std::endl;
          }
        }
      }
    }
    std::cout << "# linear constraints: " << linConstrCount_ << std::endl;
    if(linConstrCount_>0) std::cout << linConstr_ << std::endl;
    std::cout << "# sparse linear constraints: " << splinConstrCount_ << std::endl;
    if(splinConstrCount_>0) std::cout << splinConstr_ << std::endl;
    std::cout << "# nonlinear constraints: " << nonLinConstrCount_ << std::endl;
    if(nonLinConstrCount_>0) std::cout << nonLinConstr_ << std::endl;
    std::cout << "# sparse linear L1 constraints: " << splinL1ConstrCount_ << std::endl;
    std::cout << "# sparse linear L2 constraints: " << splinL2ConstrCount_ << std::endl;
    std::cout << "# sparse linear Fun constraints: " << splinFConstrCount_ << std::endl;
    if(splinFConstrCount_>0){
    std::cout << "sparse linear Fun constraints: functor types:" << std::endl;
    for(size_t i=0;i<splinFConstrCount_;i++) std::cout << (*splinFfun_[i]).name() << std::endl;
    }
  }


  void auxToJSON(jsonOut& jf){

    jf.push("dim",dim());
    jf.push("dimGen",dimGen());


    if(epn_.size()==0) expandParNames();
    jf.push("parNames",epn_);
    jf.push("defaultPars",defaultVals_);
    if(egn_.size()==0) expandGeneratedNames();
    jf.push("genNames",egn_);
    jf.push("defaultGenerated",defaultGens_);

  }

  void getDefaultVals(Eigen::VectorXd& defVals){
    if(defVals.size()!=defaultVals_.size()) defVals.resize(defaultVals_.size());
    for(size_t i=0;i<defVals.size();i++) defVals.coeffRef(i) = defaultVals_[i];
  }




};





}// namespace tensorSpec
#endif
