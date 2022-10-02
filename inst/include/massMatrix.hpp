#ifndef _MASSMATRIX_HPP_
#define _MASSMATRIX_HPP_

#include <Eigen/Dense>



class identityMass{
  int dim_;
public:
  
  Eigen::ArrayXi adaptable_;
  
  identityMass(){}
  inline std::string type() const {return "identityMass";}
  inline void setup(const int dim){dim_ = dim;}
  inline void Minv(Eigen::VectorXd &x) const {} // in place apply M^{-1}
  inline void M(Eigen::VectorXd &x) const {}
  inline void sqrtM(Eigen::VectorXd &x) const {} // in place apply Lower Cholesky factor of M
  inline void sqrtMinv(Eigen::VectorXd &x) const {}
  inline double Mdiag(const int which) const {return 1.0;}
  inline Eigen::VectorXd MinvDiag() const {
    Eigen::VectorXd ret(dim_);
    ret.setOnes();
    return(ret);
  }
  inline double MinvMax() const {return 1.0;}
  inline double MinvMin() const {return 1.0;}
  inline bool massAllowsFixedSubvector() const { return false;}
  bool setFixedMinv(const Eigen::VectorXd &fixedMinv) const {return false;}
  inline bool allowsAdaptation() const {return false;} 
  inline void monitor(const Eigen::VectorXd &q,
                      const Eigen::VectorXd &p,
                      const Eigen::VectorXd &qdot,
                      const Eigen::VectorXd &pdot,
                      Eigen::VectorXd &monitorVec){
    std::cout << "monitor: this method should not be called!" << std::endl;
  }
  void push(const double eps,
            const Eigen::VectorXd &Sample){
    std::cout << "push: this method should not be called!!!" << std::endl;
  }
  void adaptUpdate(){ 
    std::cout << "adaptUpdate: this method should not be called!!!" << std::endl;
  }
  
};


class diagMassBase{
public:
  Eigen::VectorXd Minv_;
  Eigen::VectorXd sqrtM_;
  Eigen::ArrayXi adaptable_;
  
  inline void update(){
    sqrtM_ = Minv_.array().sqrt().inverse().matrix();
  }
  
  diagMassBase() {}
  void setup(const int dim){
    Minv_.resize(dim);
    Minv_.setOnes();
    sqrtM_ = Minv_;
    adaptable_.setLinSpaced(dim, 0, dim-1);
  }
  
  bool setMinv(const Eigen::VectorXd &Minv){
    if(Minv.size() != Minv_.size()){
      std::cout << "WARNING : bad dimension in setMinv, ignored" << std::endl;
      return(false);
    }
    Minv_ = Minv;
    update();
    return(true);
  }
  
  /*
   * fixedMinv should be a vector of length dim. Non-positive entries 
   * are interpreted as non-fixed/adaptable elements in Minv 
   */
  
  bool setFixedMinv(const Eigen::VectorXd &fixedMinv){
    if(fixedMinv.size()!=Minv_.size()){
      std::cout << "WARNING : bad dimension in setFixedMinv" << std::endl; 
      return(false);
    }
    
    adaptable_.resize((fixedMinv.array()<=0.0).count());
    int k=0;
    for(int i=0;i<Minv_.size();i++){
      if(fixedMinv.coeff(i)>0.0){
        Minv_.coeffRef(i) = fixedMinv.coeff(i);
      } else {
        adaptable_.coeffRef(k) = i;
        k++;
      }
    }
    update();
    
    return(true);
  }
  
  inline double MinvMax() const {return Minv_.maxCoeff();}
  inline double MinvMin() const {return Minv_.minCoeff();}
  inline bool massAllowsFixedSubvector() const { return true;}
  void dump(){
    std::cout << "diagonal mass matrix, dimension : " << Minv_.size() << std::endl;
    std::cout << "Minv : \n" << Minv_ << std::endl;
    std::cout << "sqrtM : \n" << sqrtM_ << std::endl;
    std::cout << "adaptable : \n" << adaptable_ << std::endl << std::endl;
  }
  
  inline void Minv(Eigen::VectorXd &x) const {x.array()*=Minv_.array(); }
  inline void M(Eigen::VectorXd &x) const {x.array()/=Minv_.array(); }
  inline void sqrtM(Eigen::VectorXd &x) const {x.array()*=sqrtM_.array();}
  inline void sqrtMinv(Eigen::VectorXd &x) const {x.array()/=sqrtM_.array();}
  inline Eigen::VectorXd MinvDiag() const {return Minv_;}
  inline double Mdiag(const int which) const {return 1.0/Minv_.coeff(which);}
};


/*
 * Non-adaptable/constant diagonal 
 * 
 * Non-identiy mass is attained using the setMinv method in the base class
 */

class diagMassFixed: public diagMassBase{
public:
  diagMassFixed(){}
  inline std::string type() const {return "diagMassFixed";}
  inline bool allowsAdaptation() const {return false;} 
  inline void monitor(const Eigen::VectorXd &q,
                      const Eigen::VectorXd &p,
                      const Eigen::VectorXd &qdot,
                      const Eigen::VectorXd &pdot,
                      Eigen::VectorXd &monitorVec){
    std::cout << "monitor: this method should not be called!" << std::endl;
  }
  
  inline void push(const double eps,
            const Eigen::VectorXd &Sample){
    std::cout << "push: this method should not be called" << std::endl;
  }

  inline void adaptUpdate(){ 
    std::cout << "adaptUpdate: this method should not be called!!!" << std::endl;
  }
}; 


/*
 * mass matrix based on integrated squared gradients
 * 
 * 
 */
class diagMassISG: public diagMassBase{
private: 
  Eigen::VectorXd mean_,var_,logISG_,tmpVec_;
  int numPushed_;
public:
  diagMassISG() : numPushed_(0) {}
  inline std::string type() const {return "diagMassISG";}
  inline bool allowsAdaptation() const {return true;} 
  inline void monitor(const Eigen::VectorXd &q,
              const Eigen::VectorXd &p,
              const Eigen::VectorXd &qdot,
              const Eigen::VectorXd &pdot,
              Eigen::VectorXd &monitorVec){
    Eigen_utils_cp_index_array_right(pdot,adaptable_,monitorVec);
    monitorVec.array() *= monitorVec.array();
  }
  inline void push(const double eps, 
            const Eigen::VectorXd &Sample){
    if(numPushed_==0){
      mean_ = ((1.0/eps)*Sample).array().log().matrix();
      var_.resize(Sample.size());
      var_.setZero();
    } else {
      logISG_ = ((1.0/eps)*Sample).array().log().matrix();
      tmpVec_ = logISG_-mean_;
      mean_ += _MASS_EMA_ALPHA_*tmpVec_;
      var_ = (1.0-_MASS_EMA_ALPHA_)*(var_ + _MASS_EMA_ALPHA_*(tmpVec_.array().square().matrix()));
    }
    numPushed_++;
  }
  inline void adaptUpdate(){ 
    
    if(numPushed_>_MASS_MIN_SAMPLES_){
      tmpVec_ = (-mean_ - 0.5*var_).array().exp().matrix();
      double old;
      int ii;
      for(int i=0;i<adaptable_.size();i++){
        ii = adaptable_.coeff(i);
        old = Minv_.coeff(ii);
        Minv_.coeffRef(ii) = std::fmin(2.0*old,std::fmax(0.5*old,tmpVec_.coeff(i)));
      }
      //tmpVec_ = tmpVec_.cwiseMax(0.5*Minv_).cwiseMin(2.0*Minv_);
      //Eigen_utils_cp_index_array_left(tmpVec_,adaptable_,Minv_);
      update();
      //std::cout << "adaptUpdate" << std::endl;
      //std::cout << Minv_ << std::endl;
    }
    
  }
}; 

class diagMassVARI : public diagMassBase {
  int numPushed_;
  Eigen::VectorXd M12_;
  Eigen::VectorXd tmpVec_;
  double T_;
  
public:
  diagMassVARI() : numPushed_(0) {}
  inline std::string type() const {return "diagMassVARI";}
  bool allowsAdaptation() const {return true;} 
  inline void monitor(const Eigen::VectorXd &q,
                      const Eigen::VectorXd &p,
                      const Eigen::VectorXd &qdot,
                      const Eigen::VectorXd &pdot,
                      Eigen::VectorXd &monitorVec){
    int asize = adaptable_.size();
    if(monitorVec.size() != 2*asize) monitorVec.resize(2*asize);
    for(int i=0;i<asize;i++){
      monitorVec.coeffRef(i) = q.coeff(adaptable_.coeff(i));
      monitorVec.coeffRef(i+asize) = std::pow(monitorVec.coeff(i),2);
    }
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
        old = Minv_.coeff(ii);
        Minv_.coeffRef(ii) = std::fmin(2.0*old,std::fmax(0.5*old,tmpVec_.coeff(i)));
      }
      //tmpVec_ = tmpVec_.cwiseMax(0.5*Minv_).cwiseMin(2.0*Minv_);
      //Eigen_utils_cp_index_array_left(tmpVec_,adaptable_,Minv_);
      update();
    } 
   //std::cout << "adaptUpdate" << std::endl; 
  }
  
};




#endif
