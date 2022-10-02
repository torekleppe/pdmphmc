#ifndef _STARTWRAP_HPP_
#define _STARTWRAP_HPP_


template < class _process_type_ >
class startWrapHMCFirstOrderODE{
  _process_type_ *proc_;
  Eigen::VectorXd q_last_event_;
  Eigen::VectorXd p_tmp_;
  Eigen::VectorXd genDummy_;
  Eigen::VectorXd diagIntDummy_;
    int dim_;
    double oldTarget_;
    double halfChiSqThresh_;
    
  
public:
  startWrapHMCFirstOrderODE(){}
  void setup(_process_type_ &proc){
    proc_ = &proc;
    dim_ = (*proc_).systemDim();
    genDummy_.resize((*proc_).generatedDim());
    oldTarget_ = -1.0e100;
    halfChiSqThresh_ = 100.0*boost::math::gamma_p_inv(0.5*static_cast<double>(dim_), 0.999);
    
  }
  
  int dim(){return (*proc_).dim();}
  int generatedDim(){ return 0;}
  
  void setQLastEvent(const Eigen::VectorXd &q){q_last_event_=q;}
  
  void ode(const double time,
           const Eigen::VectorXd &y, // y = (q,p,lambda)
           Eigen::VectorXd &f,
           Eigen::VectorXd &gen,
           Eigen::VectorXd &diagInt){
    
    (*proc_).ode(time,y,f,genDummy_,diagIntDummy_);
    if(gen.size()>0) gen.resize(0);
    if(diagInt.size()>0) diagInt.resize(0);
    
    
  }
  
  
  inline int eventRootDim() const {return 2;}
  inline Eigen::VectorXd eventRoot(const double time,
                                   const odeState &state,
                                   const Eigen::VectorXd &f) const {
    Eigen::VectorXd ret(eventRootDim());
    // time derivative of potential energy
    ret(0) = state.y.segment(dim_,dim_).dot(f.segment(dim_,dim_)); 
    // kinetic energy threshold
    ret(1) = state.y.segment(dim_,dim_).dot(state.y.segment(dim_,dim_)) - halfChiSqThresh_;
    
    return(ret);
  }
  
  inline int warmupRoot() const {return -1;} // no warmup root
  
  bool event(const int EventType,
             const double time,
             const odeState &oldState, 
             const Eigen::VectorXd &f,
             odeState &newState){
    
    oldState.copyTo(newState); // q unchanged
    
    if(EventType==0){
      (*proc_).r_.rnorm(newState.y.segment(dim_,dim_)); // momentum updated 
    } else if(EventType==1){
      newState.y.segment(dim_,dim_) = sqrt(0.5*static_cast<double>(dim_)/halfChiSqThresh_)*oldState.y.segment(dim_,dim_);
    }
    
    
    
    
    q_last_event_ = oldState.y.head(dim_);
    newState.y(2*dim_) = 0.0; // reset integrated lambda
    
    // determine if stop
    double newTarget = (*proc_).target(oldState.y.head(dim_));
    std::cout << "startWrap, event type " << EventType << std::endl;
    std::cout << newTarget << std::endl;
    
    
    if(newTarget>oldTarget_){
      oldTarget_ = newTarget;
      return(true);
    } else {
      std::cout << "Decrease in target value; new target : " 
      << newTarget << " old target : " << oldTarget_ << std::endl;
      return(false);
    }
  }
  
  template <class stepType>
  inline void push_RK_step(const int integratorID,
                    const double lastTime,
                    const stepType &step){}
  
  template <class stepType>
  inline void getStorePars(const Eigen::VectorXi &storeParsInds,
                           const double stepTime,
                           const stepType &step,
                           Eigen::Ref<Eigen::VectorXd> out){
    (*proc_).getStorePars(storeParsInds,stepTime,step,out);
  }
  
};


template < class _process_type_ >
class startWrapHMCSecondOrderODE{
  _process_type_ *proc_;
  Eigen::VectorXd q_last_event_;
  Eigen::VectorXd p_tmp_;
  Eigen::VectorXd genDummy_;
  Eigen::VectorXd diagIntDummy_;
  int dim_;
  double oldTarget_;
  double halfChiSqThresh_;
  
  
public:
  startWrapHMCSecondOrderODE(){}
  void setup(_process_type_ &proc){
    proc_ = &proc;
    dim_ = (*proc_).systemDim();
    genDummy_.resize((*proc_).generatedDim());
    oldTarget_ = -1.0e100;
    halfChiSqThresh_ = 100.0*boost::math::gamma_p_inv(0.5*static_cast<double>(dim_), 0.999);
    
  }
  
  int dim(){return (*proc_).dim();}
  int generatedDim(){ return 0;}
  
  void setQLastEvent(const Eigen::VectorXd &q){q_last_event_=q;}
  
  void ode(const double time,
           const Eigen::VectorXd &y, // y = (q,p,lambda)
           Eigen::VectorXd &f,
           Eigen::VectorXd &gen,
           Eigen::VectorXd &diagInt){
    
    (*proc_).ode(time,y,f,genDummy_,diagIntDummy_);
    if(gen.size()>0) gen.resize(0);
    if(diagInt.size()>0) diagInt.resize(0);
    
    
  }
  
  
  inline int eventRootDim() const {return 2;}
  inline Eigen::VectorXd eventRoot(const double time,
                                   const odeState &state,
                                   const Eigen::VectorXd &f) const {
    Eigen::VectorXd ret(eventRootDim());
    
    
    // time derivative of potential energy
    ret(0) = f.dot(state.ydot); 
    // kinetic energy threshold
    ret(1) = state.ydot.dot(state.ydot) - halfChiSqThresh_;
    
    return(ret);
  }
  
  inline int warmupRoot() const {return -1;} // no warmup root
  
  bool event(const int EventType,
             const double time,
             const odeState &oldState, 
             const Eigen::VectorXd &f,
             odeState &newState){
    
    oldState.copyTo(newState); // q unchanged
    
    if(EventType==0){
      (*proc_).r_.rnorm(newState.ydot);  
    } else if(EventType==1){
      newState.ydot = sqrt(0.5*static_cast<double>(dim_)/halfChiSqThresh_)*oldState.ydot;
    }
    
    
    
    
    q_last_event_ = oldState.y;
    newState.M.setZero(); // reset integrated lambda
    
    // determine if stop
    double newTarget = (*proc_).target(oldState.y);
    std::cout << "startWrap, event type " << EventType << std::endl;
    std::cout << newTarget << std::endl;
    
    
    if(newTarget>oldTarget_){
      oldTarget_ = newTarget;
      return(true);
    } else {
      std::cout << "Decrease in target value; new target : " 
                << newTarget << " old target : " << oldTarget_ << std::endl;
      return(false);
    }
  }
  
  
  inline Eigen::VectorXd monitor(const double time,
                                 const Eigen::VectorXd &y,
                                 const Eigen::VectorXd &ydot,
                                 const Eigen::VectorXd &f) const {
    Eigen::VectorXd ret;
    return(ret);
  }
  
  template <class stepType>
  inline void push_RK_step(const int integratorID,
                    const double lastTime,
                    const stepType &step){}
  
  template <class stepType>
  inline void getStorePars(const Eigen::VectorXi &storeParsInds,
                           const double stepTime,
                           const stepType &step,
                           Eigen::Ref<Eigen::VectorXd> out){
    (*proc_).getStorePars(storeParsInds,stepTime,step,out);
  }
  
};


#endif