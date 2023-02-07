#ifndef __RUNGEKUTTA_HPP__
#define __RUNGEKUTTA_HPP__


/*
 * Quite generic Runge Kutta (Nystrom )Solver
 *
 */

#define __INTEGRATOR_DIAG_BLOCK_SIZE_ 1000

//#define __DEBUG__


template <class _ode_type_,
          template<typename> class _step_type_,
          class _diagnostics_type_>
class rungeKuttaSolver{
private:

  _ode_type_ *ode_;
  _step_type_< _ode_type_ > step_;
  _diagnostics_type_ *diagnostics_;


  int dim_;
  int dimGenerated_;
  int dimEvent_;


  int maxSteps_;



  double Tmax_;
  double sample_interval_;
  int sample_count_;
  Eigen::Matrix<int,Eigen::Dynamic,1> storePars_;
  /*
   Eigen::MatrixXd force_;
   Eigen::MatrixXd ys_;
   Eigen::MatrixXd events_;
   Eigen::MatrixXd generated_;
   Eigen::MatrixXd diag_;
   */

  Eigen::Matrix<bool,Eigen::Dynamic,1> eventActive_;
  Eigen::MatrixXd eventCheckPoints_;
  Eigen::VectorXd eventTimes_;

  /*
   Eigen::VectorXd y1_low_;
   Eigen::VectorXd y_tmp_;
   Eigen::VectorXd force_tmp_;
   Eigen::VectorXd gen_tmp_;
   Eigen::VectorXd diag_tmp_;
   */

  Eigen::VectorXd tmpVec_;
  Eigen::VectorXd tmpVecEvent_;

  Eigen::VectorXd genIntPrevSteps_;
  //Eigen::VectorXd genIntStep_;
  Eigen::VectorXd genIntLast_;
  Eigen::VectorXd genIntSample_;

  //Eigen::VectorXd diagInt_;


  // step size controller
  //double stepErr_;
  double PI_beta_;
  double PI_alpha_;

  // diagnostics info
  int id_;
  std::string printPrefix_;


  //specialRootSpec sps_;

public:

  // samples
  Eigen::MatrixXd samples_;
  Eigen::MatrixXd intSamples_;

  // timing
  Eigen::VectorXd CPUtime_;

  rungeKuttaSolver() : dim_(0), maxSteps_(10000), PI_beta_(0.08), PI_alpha_(0.14)  {}
  inline int odeOrder() const {return step_.odeOrder();}
  void setup(_ode_type_ &ode){

    step_.setup(ode);
    PI_beta_ = 0.4/step_.errorOrderHigh();
    PI_alpha_ = 0.7/step_.errorOrderHigh();
    ode_ = &ode;


    dim_ = (*ode_).dim();
    dimGenerated_ = (*ode_).generatedDim();
    dimEvent_ = (*ode_).eventRootDim();


    if(dimGenerated_>0){

      genIntPrevSteps_.resize(dimGenerated_);
      genIntLast_.resize(dimGenerated_);
      genIntSample_.resize(dimGenerated_);
    }

    CPUtime_.resize(3);

#ifdef __DEBUG__
    std::cout << "rungeKuttaSolver setup: " << std::endl;
    std::cout << "system dimension: " << dim_ << std::endl;
    std::cout << "generated dimension: " << dimGenerated_ << std::endl;
    std::cout << "eventRoot dimension: " << dimEvent_ << std::endl;
#endif

      //(*ode_).specialRoots(sps_);


  }


  void registerDiagnostics(_diagnostics_type_ &diag,
                           const int integratorId){
    diagnostics_ = &diag;
    id_ = integratorId;
  }

  void setAbsTol(const double tol){step_.absTol_=tol;}
  void setRelTol(const double tol){step_.relTol_=tol;}
  double getAbsTol(){return step_.absTol_;}
  double getRelTol(){return step_.relTol_;}

  void setPrintPrefix(const std::string prefix){
    printPrefix_ = "[" + prefix +"]";
  }
  void auxiliaryDiagnosticsInfo(jsonOut &outf) const {
    outf.push("absTol",step_.absTol_);
    outf.push("relTol",step_.relTol_);
  }

  inline odeState getStateLastIntegrated() const {return step_.lastState();}

  bool setInitialState(const odeState &state0){
#ifdef __DEBUG__
    std::cout << "int id : " << id_ << " setInititialState : state0 " << std::endl << state0.y << std::endl;
#endif
    return step_.setInitialState(state0);
  }


  double run(const Eigen::Matrix<int,Eigen::Dynamic,1> &storePars,
             const double Tmax,
             const int nsamples){

    bool flag;
    bool stepGood;
    bool done;
    bool eventContinue = true;
    int nstep = 0;
    int nacc = 0;


    Eigen::Array<bool,Eigen::Dynamic,1> whichEvent;

    // for PI controller
    double stepErrOld = 1.0;


    // assorted storage
    //double eventTime;
    //int whichEventFirst;
    double sampleStepTime;
    eventTimes_.setZero();
    double nextSampleTime;
    rootInfo rootOut;
    step_.t_right_ = 0.0;
    done = false;


    storePars_ = storePars;
    samples_.resize(storePars_.size()+dimGenerated_,nsamples+1);
    intSamples_.resize(dimGenerated_,nsamples+1);

    sample_interval_ = Tmax/static_cast<double>(nsamples);
    sample_count_ = 0;
    Tmax_ = Tmax;
    //oldEventDim_ = -1;



    // store initial state
    for(int i=0;i<storePars_.size();i++ ) samples_.coeffRef(i,0) = step_.firstState(storePars_.coeff(i));
    //
    if(dimGenerated_>0){
      samples_.col(0).tail(dimGenerated_) = step_.firstGenerated(); //generated_.col(0);
      intSamples_.col(0).setZero();
      genIntPrevSteps_.setZero();
    }
    sample_count_ = 1;

    double timeStart = 0.0;
    double printCounter = 1.0;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_print_time = std::chrono::high_resolution_clock::now();
    CPUtime_.setZero();



    // main simulation loop

    while(nstep <= maxSteps_ && step_.t_right_<Tmax_){ // main time loop

      stepGood = false;
      while(nstep <= maxSteps_){ // step accept/reject loop
        step_.eps_ = std::min(Tmax_-step_.t_left_,step_.eps_); // make last mesh point = Tmax
#ifdef __DEBUG__
        std::cout << "integrator id : " << id_ << " : before step, eps: "  << step_.eps_ << std::endl;
       step_.dumpYs();
#endif
        flag = step_.step();
        nstep++;
#ifdef __DEBUG__
        std::cout << "integrator id: " << id_ << " step err: " << step_.stepErr_  << " eps: "  << step_.eps_ << std::endl;
        std::cout << "flag: " << flag << std::endl;
        step_.dumpYs();
#endif
        if(flag){ // no numerical problems
          if(step_.stepErr_<1.0){ // step accepted
            nacc++;
            stepGood = true;
            break;
          } else { // step rejected
            step_.eps_ *= std::max(0.2,0.9*std::pow(step_.stepErr_,-0.2));
          }

        } else { // numerical problems
          step_.eps_ *= 0.1;
        }
      }
      //std::cout << t_left_+eps_ << "  " << printCounter << std::endl;
      // print output
      if(id_ < 0.5){ // only print from the main integrator with id=0.0
        while(step_.t_right_ >= 0.1*printCounter*Tmax_){
          printCounter += 1.0;
          auto print_now = std::chrono::high_resolution_clock::now();
          std::chrono::duration<double> elapsed = print_now - last_print_time;
          if(elapsed.count()>0.1){
            std::cout << printPrefix_ << " t = " << step_.t_right_ << " (done with " <<
              100.0*(step_.t_right_)/Tmax_ << "% of simulation)" << std::endl;
            last_print_time = print_now;
          }
        }
      }

      //
      // check if events occurred
      //
      //if(step_.hasEventRootSolver()){
      rootOut = step_.eventRootSolver();


#ifdef __DEBUG__
      std::cout << rootOut << std::endl;
#endif

      //} else {
      //  eventTime = eventGridSearch(whichEventFirst);
      //}
      // store dense output
      nextSampleTime = static_cast<double>(sample_count_)*sample_interval_;
      if(dimGenerated_>0) genIntLast_.setZero();

      while(step_.t_left_ + rootOut.rootTime_ >= nextSampleTime){
        sampleStepTime = nextSampleTime-step_.t_left_;

#ifdef __RK_DO_NOT_TRANSFORM_STORE_PARS__
        // store the actual parameterization used, generally faster
        step_.denseState(storePars_,sampleStepTime,samples_.col(sample_count_).head(storePars_.size()));
#else
        (*ode_).getStorePars(storePars_,
         sampleStepTime,
         step_,
         samples_.col(sample_count_).head(storePars_.size()));
#endif

        if(dimGenerated_>0){
          step_.denseGenerated_Level(sampleStepTime,samples_.col(sample_count_).tail(dimGenerated_));

          step_.denseGenerated_Int(sampleStepTime,genIntSample_);

          intSamples_.col(sample_count_) =(1.0/sample_interval_)*(genIntPrevSteps_ + genIntSample_ - genIntLast_);
          genIntPrevSteps_.setZero();
          genIntLast_ = genIntSample_;
        }
        sample_count_++;
        nextSampleTime = static_cast<double>(sample_count_)*sample_interval_;
      }

      //std::cout << "dense sampling part 1 done" << std::endl;
      // reminder of step
      if(dimGenerated_>0){
        if(rootOut.rootDim_==-1){
          genIntPrevSteps_ += step_.genIntStep_ - genIntLast_;
        } else {
          //genIntPrevSteps_ += generated_*denseWts(eventTime/eps_) - genIntLast_;
          genIntPrevSteps_ += step_.denseGenerated_Int(rootOut.rootTime_) - genIntLast_;
        }
      }
      // end dense output


      // pass the finished step to the ODE in order to collect diagnostics info etc
      (*ode_).push_RK_step(id_,rootOut.rootTime_,step_);

      //std::cout << "dense sampling done" << std::endl;

      // from now on, modifications of the quantities calculated in the step are done
      // carry out what ever happened at the event
      if(rootOut.rootDim_ != -1 ){

        if(rootOut.rootTime_==(*ode_).warmupRoot()){
          auto warmup_time = std::chrono::high_resolution_clock::now();
          std::chrono::duration<double> warmup_elapsed = warmup_time-start_time;
          CPUtime_(0) = warmup_elapsed.count();
          std::cout << printPrefix_ <<  " warmup done" << std::endl;
        }

        eventContinue = step_.event(rootOut); //step_.event(whichEventFirst,eventTime);

        // store diagonstics info leading to
        (*diagnostics_).push("intID",id_);
        (*diagnostics_).push("timeStart",timeStart);
        timeStart = step_.t_left_+rootOut.rootTime_;
        (*diagnostics_).push("timeEnd",timeStart);
        (*diagnostics_).push("eps",step_.eps_);
        (*diagnostics_).push("stepErr",step_.stepErr_);
        (*diagnostics_).push("nstep",static_cast<double>(nstep));
        (*diagnostics_).push("nacc",static_cast<double>(nacc));
        (*diagnostics_).push("eventType",static_cast<double>(rootOut.rootType_));
        (*diagnostics_).push("eventDim",static_cast<double>(rootOut.rootDim_));

        (*diagnostics_).newRow();
        // prepare next integration leg
        nstep=0;
        nacc=0;
      }

      //std::cout << "RK::event done" << std::endl;


      // Stop simulation if the event function returns false
      if(! eventContinue){
        //std::cout << id_ << " simulation stopped by event" << std::endl;
        auto endTime_e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> end_elapsed_e = endTime_e-start_time;
        CPUtime_(2) = end_elapsed_e.count();
        return(step_.t_right_);
      }

      //std::cout << "prepare next start" << std::endl;

      // prepare for next integration step
      step_.prepareNext();

      //std::cout << "prepare next done" << std::endl;


      step_.eps_ *= std::min(5.0,std::max(0.2,0.95*std::pow(step_.stepErr_,-PI_alpha_)*std::pow(stepErrOld,PI_beta_)));
      stepErrOld = step_.stepErr_;
    } // main time loop

    if(step_.t_right_<Tmax_){
      std::cout << "integrator failed, possibly increase maxSteps_ ?" << std::endl;
      std::cout << "nstep : " << nstep << "maxSteps_ : " << maxSteps_ << std::endl;
      std::cout << "integrator id: " << id_ << " step err: " << step_.stepErr_  << " eps: "  << step_.eps_ << std::endl;
#ifdef __DEBUG__
      std::exit(0);
#endif
    }

    // final row of diagonsitics

    (*diagnostics_).push("intID",id_);
    (*diagnostics_).push("timeStart",timeStart);
    (*diagnostics_).push("timeEnd",step_.t_right_);
    (*diagnostics_).push("eps",step_.eps_);
    (*diagnostics_).push("stepErr",step_.stepErr_);
    (*diagnostics_).push("nstep",static_cast<double>(nstep));
    (*diagnostics_).push("nacc",static_cast<double>(nacc));
    (*diagnostics_).push("eventType",static_cast<double>(rootOut.rootType_));
    (*diagnostics_).push("eventDim",static_cast<double>(rootOut.rootDim_));

    // timing
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> end_elapsed = endTime-start_time;
    CPUtime_(2) = end_elapsed.count();
    CPUtime_(1) = CPUtime_(2)-CPUtime_(0);

    return(step_.t_right_);
  } // run


  void pointSamplesToFile(const std::string filename){
    Eigen::IOFormat CSV(8, Eigen::DontAlignCols, ", ", "\n");
    std::ofstream file;
    file.open(filename);
    file << samples_.transpose().format(CSV) << std::endl;
    file.close();
  }

  void intSamplesToFile(const std::string filename){
    Eigen::IOFormat CSV(8, Eigen::DontAlignCols, ", ", "\n");
    std::ofstream file;
    file.open(filename);
    file << intSamples_.transpose().format(CSV) << std::endl;
    file.close();
  }




  void samplesToFile(const int csvPrec,
                     const bool point, //otherwise integrated samples
                     const std::vector<std::string> colhead,
                     const std::string filename){
    Eigen::IOFormat CSV(csvPrec, Eigen::DontAlignCols, ", ", "\n");
    std::ofstream file;
    file.open(filename);
    file << "\"";
    for(int c=0;c<colhead.size()-1;c++) file << colhead[c] << "\" , \"" ;
    file << colhead[colhead.size()-1] << "\"" << std::endl;
    if(point){
      file << samples_.transpose().format(CSV) << std::endl;
    } else {
      file << intSamples_.transpose().format(CSV) << std::endl;
    }
    file.close();
  }

}; // end class rungeKuttaSolver









#endif
