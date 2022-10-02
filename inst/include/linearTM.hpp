#ifndef _LINEARTM_HPP_
#define _LINEARTM_HPP_



class identityTM{
  int dim_;
public:
  inline void setup(const int dim){dim_=dim;}
  
  inline void toPar(Eigen::VectorXd &var) const {}
  inline void toPar(const Eigen::VectorXi &which,
             Eigen::Ref<Eigen::VectorXd> par) const {}
  inline void toPar(const Eigen::VectorXd &q, 
             const Eigen::VectorXi &which,
             Eigen::VectorXd &par) const {
    if(par.size()!=which.size()) par.resize(which.size());
    for(int i=0;i<which.size();i++) par.coeffRef(i) = q.coeff(which.coeff(i));
  }
  inline void toParJacTransposed(Eigen::VectorXd &var) const {}
  inline void toParJac(Eigen::VectorXd &var) const {}
  inline void toParInverseJacTransposed(Eigen::VectorXd &var) const {}
  template <class varType>
  inline void toParInverseJac(Eigen::Matrix<varType,Eigen::Dynamic,1> &var) const {}
  inline void toQ(Eigen::VectorXd &var) const {}
  inline bool massAllowsFixedSubvector() const {return(false);}
  inline bool setFixedMinv(const Eigen::VectorXd &fixedMi){
    std::cout << "identityTM: attempting to set fixedMinv" << std::endl;
    return(false);
  }
  bool allowsAdaptation() const {return(false);}
  inline double LMax() const {return(1.0);}
  inline double LMin() const {return(1.0);}
  inline double massDiag(const size_t which) const {return(1.0);}
  inline void monitor(const Eigen::VectorXd &q,
                      const Eigen::VectorXd &qdot,
                      const Eigen::VectorXd &qdotdot,
                      Eigen::VectorXd &mon) {
    if(mon.size()!=0) mon.resize(0);
    std::cout << "identityTM: monitor: should not be called" << std::endl;
  }
  void monitor(const Eigen::VectorXd &q,
               const Eigen::VectorXd &p,
               const Eigen::VectorXd &qdot,
               const Eigen::VectorXd &pdot,
               Eigen::VectorXd &mon) {
    if(mon.size()!=0) mon.resize(0);
    std::cout << "identityTM: monitor: should not be called" << std::endl;
  }
  inline void push(const double eps_,
                   const Eigen::VectorXd &monInt_){
  }
  inline void adaptUpdate(){
    std::cout << "identityTM: attempting adaptUpdate()" << std::endl;
  }
  
  
};



class diagLinearTM_base{
public:
  Eigen::VectorXd mu_;
  Eigen::VectorXd L_; // \approx marginal standard deviations of target
  Eigen::VectorXi adaptable_;
  
  
  void setup(const int dim){
    mu_.resize(dim);
    mu_.setZero();
    L_.resize(dim);
    L_.setOnes();
    adaptable_.resize(dim);
    for(int i=0;i<dim;i++) adaptable_.coeffRef(i) = i;
  }
  bool massAllowsFixedSubvector() const {return(true);}
  bool setFixedMinv(const Eigen::VectorXd &fixedMi){
    if(fixedMi.size()!=L_.size()){
      std::cout << "WARNING : bad dimension in setFixedMinv" << std::endl; 
      return(false);
    }
    
    adaptable_.resize((fixedMi.array()<=0.0).count());
    int k=0;
    for(int i=0;i<L_.size();i++){
      if(fixedMi.coeff(i)>0.0){
        L_.coeffRef(i) = sqrt(fixedMi.coeff(i));
      } else {
        adaptable_.coeffRef(k) = i;
        k++;
      }
    }
    return(true);
  }
  inline double massDiag(const size_t which){return(pow(L_.coeff(which),-2));}
  bool allowsAdaptation() const {return(adaptable_.size()>0);}
  void toPar(Eigen::VectorXd &var) const {
    var.array() *= L_.array();
    var+=mu_;
  }
  // NOTE, presumes diagonal matrix!!
  void toPar(const Eigen::VectorXi &which,
             Eigen::Ref<Eigen::VectorXd> par) const {
    for(int i=0;i<which.size();i++){
      par.coeffRef(i) *= L_.coeff(which.coeff(i));
      par.coeffRef(i) += mu_.coeff(which.coeff(i));
    }
  }
  void toPar(const Eigen::VectorXd &q, 
             const Eigen::VectorXi &which,
             Eigen::VectorXd &par) const {
    if(par.size()!=which.size()) par.resize(which.size());
    for(int i=0;i<which.size();i++) par.coeffRef(i) = mu_.coeff(which.coeff(i)) +  L_.coeff(which.coeff(i))*q.coeff(which.coeff(i));
  }
  void toParJacTransposed(Eigen::VectorXd &var) const {
    var.array() *= L_.array();
  }
  void toParJac(Eigen::VectorXd &var) const {
    var.array() *= L_.array();
  }
  void toParInverseJacTransposed(Eigen::VectorXd &var) const {
    var.array() /= L_.array();
  }
  template <class varType>
  void toParInverseJac(Eigen::Matrix<varType,Eigen::Dynamic,1> &var) const {
    var.array() /= L_.array();
  }
  void toQ(Eigen::VectorXd &var) const {
    var -= mu_;
    var.array() /= L_.array();
  }
  
  inline double LMax(){return L_.maxCoeff();}
  inline double LMin(){return L_.minCoeff();}
  void toJSON(jsonOut &outf) const {
    outf.push("TM_center",mu_);
    outf.push("TM_scaling",L_);
  }
};


class diagLinearTM_VARI: public diagLinearTM_base {
  Eigen::VectorXd parTmp_;
  int numPushed_;
  Eigen::VectorXd M12_;
  Eigen::VectorXd tmpVec_;
  double T_;
  
public:
  diagLinearTM_VARI() :  numPushed_(0), T_(0.0) {}
  void monitor(const Eigen::VectorXd &q,
               const Eigen::VectorXd &qdot,
               const Eigen::VectorXd &qdotdot,
               Eigen::VectorXd &mon) {
    if(adaptable_.size()>0){
      toPar(q,adaptable_,parTmp_);
      if(mon.size()!=2*adaptable_.size()) mon.resize(2*adaptable_.size());
      mon.head(adaptable_.size()) = parTmp_;
      mon.tail(adaptable_.size()) = parTmp_.array().square().matrix();
    } else {
      if(mon.size()!=0) mon.resize(0);
    }
  }
  Eigen::VectorXd emptyVec_;
  void monitor(const Eigen::VectorXd &q,
               const Eigen::VectorXd &p,
               const Eigen::VectorXd &qdot,
               const Eigen::VectorXd &pdot,
               Eigen::VectorXd &mon) {
    monitor(q,emptyVec_,emptyVec_,mon);
  }
  inline void push(const double eps,
                   const Eigen::VectorXd &Sample){
    double Tnew;
    if(numPushed_==0){
      M12_ = (1.0/eps)*Sample;
      T_ = eps;
    } else {
      Tnew = T_+eps;
      M12_ = (T_/Tnew)*M12_ + (1.0/Tnew)*Sample;
      T_ = Tnew;
    }
    numPushed_++;
    //std::cout << "pushed : " << numPushed_ << std::endl;
    //std::cout << M12_ << std::endl;
  }
  inline void adaptUpdate(){
    if(numPushed_>_MASS_MIN_SAMPLES_){
      int asize = adaptable_.size();
      tmpVec_ =  M12_.tail(asize) - M12_.head(asize).array().square().matrix();
      double old;
      int ii;
      for(int i=0;i<adaptable_.size();i++){
        ii = adaptable_.coeff(i);
        old = L_.coeff(ii);
        L_.coeffRef(ii) = std::fmin(2.0*old,std::fmax(0.5*old,sqrt(tmpVec_.coeff(i))));
        mu_.coeffRef(ii) = M12_.coeff(i);
      }
    } 
  }
};




class diagLinearTM_ISG: public diagLinearTM_base {
private:
  Eigen::VectorXd parTmp_;
  int numPushed_;
  Eigen::VectorXd M12_;
  Eigen::VectorXd tmpVec_;
  double T_;
public:
  diagLinearTM_ISG() : numPushed_(0), T_(0.0) {}
  inline void monitor(const Eigen::VectorXd &q,
               const Eigen::VectorXd &qdot,
               const Eigen::VectorXd &qdotdot,
               Eigen::VectorXd &mon) {
    int asize = adaptable_.size();
    int ii;
    if(asize>0){
      toPar(q,adaptable_,parTmp_);
      if(mon.size()!=2*asize) mon.resize(2*asize);
      mon.head(asize) = parTmp_;
      for(int i=0;i<asize;i++){
        // squared original gradient
        ii = adaptable_.coeff(i);
        mon.coeffRef(asize+i) = std::pow(qdotdot.coeff(ii)/L_.coeff(ii),2);
      }
    } else {
      if(mon.size()!=0) mon.resize(0);
    }
  }
  Eigen::VectorXd emptyVec_;
  inline void monitor(const Eigen::VectorXd &q,
                      const Eigen::VectorXd &p,
                      const Eigen::VectorXd &qdot,
                      const Eigen::VectorXd &pdot,
                      Eigen::VectorXd &mon){
    monitor(q,emptyVec_,pdot,mon);
  }
  
  
  
  inline void push(const double eps,
                   const Eigen::VectorXd &Sample){
    double Tnew;
    if(numPushed_==0){
      M12_ = (1.0/eps)*Sample;
      T_ = eps;
    } else {
      Tnew = T_+eps;
      M12_ = (T_/Tnew)*M12_ + (1.0/Tnew)*Sample;
      T_ = Tnew;
    }
    numPushed_++;
    //std::cout << "pushed : " << numPushed_ << std::endl;
    //std::cout << M12_ << std::endl;
  }
  inline void adaptUpdate(){
    if(numPushed_>_MASS_MIN_SAMPLES_){
      int asize = adaptable_.size();
      double old;
      int ii;
      for(int i=0;i<asize;i++){
        ii = adaptable_.coeff(i);
        old = L_.coeff(ii);
        L_.coeffRef(ii) = std::fmin(2.0*old,std::fmax(0.5*old,1.0/sqrt(M12_.coeff(asize+i))));
        mu_.coeffRef(ii) = M12_.coeff(i);
      }
    } 
  }
};









#endif