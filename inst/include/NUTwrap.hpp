


template < class _process_type_ >
class NUTwrapFirstOrderODE{
  _process_type_ *proc_;
  Eigen::VectorXd q_last_event_;
  Eigen::VectorXd genDummy_;
  Eigen::VectorXd diagIntDummy_;
    int dim_;
    specialRootSpec sps_;

public:
  NUTwrapFirstOrderODE(){}
  void setup(_process_type_ &proc){
    proc_ = &proc;
    dim_ = (*proc_).systemDim();
    genDummy_.resize((*proc_).generatedDim());
  }
  int dim(){return (*proc_).dim();}
  int generatedDim(){ return 0;}
  int eventRootDim(){return 1;}
  inline const specialRootSpec& spr() const {return sps_;}
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
  inline Eigen::VectorXd eventRoot(const double time,
                                   const odeState &state,
                                   const Eigen::VectorXd &f,
                                   const bool afterOde) const {
    Eigen::VectorXd ret(1);
    ret(0) = (state.y.head(dim_)-q_last_event_).dot(state.y.segment(dim_,dim_));
    return(ret);
  }
  inline int warmupRoot() const {return -1;} // no warmup root

  bool event(const rootInfo& eventOut,
             const double time,
             const odeState &oldState,
             const Eigen::VectorXd &f,
             odeState &newState){
    newState.y = oldState.y;
    return(false);
  }


  template <class stepType>
  void push_RK_step(const int integratorID,
                    const double lastTime,
                    const stepType &step){}

  template <class stepType>
  inline void getStorePars(const Eigen::VectorXi &storeParsInds,
                           const double stepTime,
                           const stepType &step,
                           Eigen::Ref<Eigen::VectorXd> out){
    (*proc_).getStorePars(storeParsInds,stepTime,step,out);
  }

  /*
  void pushDiag(const double eps_,
                const Eigen::VectorXd &diagInt){
    std::cout << "NUTwrap::pushDiag : should not be called" << std::endl;
  }
  */
};

template < class _process_type_ >
class NUTwrapSecondOrderODE{
  _process_type_ *proc_;
  Eigen::VectorXd q_last_event_;
  Eigen::VectorXd genDummy_;
  Eigen::VectorXd diagIntDummy_;
  int dim_;

public:
  NUTwrapSecondOrderODE(){}
  void setup(_process_type_ &proc){
    proc_ = &proc;
    dim_ = (*proc_).systemDim();
    genDummy_.resize((*proc_).generatedDim());
  }
  int dim(){return (*proc_).dim();}
  int generatedDim() const { return 0;}
  int eventRootDim() const {return 1;}

  void setQLastEvent(const Eigen::VectorXd &q){q_last_event_=q;}

  void ode(const double time,
           const Eigen::VectorXd &y, //=q
           Eigen::VectorXd &f,
           Eigen::VectorXd &gen,
           Eigen::VectorXd &diagInt){

    (*proc_).ode(time,y,f,genDummy_,diagIntDummy_);
    if(gen.size()>0) gen.resize(0);
    if(diagInt.size()>0) diagInt.resize(0);
  }
  inline Eigen::VectorXd eventRoot(const double time,
                                   const odeState &state,
                                   const Eigen::VectorXd &f) const {
    Eigen::VectorXd ret(eventRootDim());

    ret(0) = (state.y-q_last_event_).dot(state.ydot);
    return(ret);
  }
  inline int warmupRoot() const {return -1;} // no warmup root

  bool event(const int EventType,
             const double time,
             const odeState &oldState,
             const Eigen::VectorXd &f,
             odeState &newState){
    oldState.copyTo(newState);
    return(false);
  }
/*
  inline Eigen::VectorXd monitor(const double time,
                                 const Eigen::VectorXd &y,
                                 const Eigen::VectorXd &ydot,
                                 const Eigen::VectorXd &f){
    return((*proc_).monitor(time,y,ydot,f));
  }
 */
  template <class stepType>
  void push_RK_step(const int integratorID,
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
