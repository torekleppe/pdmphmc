#ifndef _PDPSAMPLER_HPP_
#define _PDPSAMPLER_HPP_

#define __REMOVE_FIELD(vec,field) vec.erase(std::find(vec.begin(), vec.end(), field));


template <template <typename, // target type
                    typename, // var type
                    template <typename,template <typename> class,typename> class, // integrator type
                    template <typename> class, // step type
                    class, // metric tensor type
                    typename, // mass Matrix/TM type
                    typename, // lambda type
                    typename // diagnostics type
                      > class process_template,
                      class target_type,
                      class var_type,
                      template <typename,template <typename> class,typename> class integrator_type,
                      template <typename> class step_type,
                      class metricTensor_type,
                      class massMatrix_TM_type,
                      class lambda_type,
                      class diagnostics_type >
class PDPsampler{

  typedef process_template<target_type,
                           var_type,
                           integrator_type,
                           step_type,
                           metricTensor_type,
                           massMatrix_TM_type,
                           lambda_type,
                           diagnostics_type> proc_type_;

  proc_type_ proc_;
  integrator_type< proc_type_, step_type ,diagnostics_type > int_;
  diagnostics_type diag_;
  propertyTable plist_;
  std::string printPrefix_;
public:

  PDPsampler(){}

  void setup(target_type& target,
             const int dim,
             const int dimGen,
             const amt::constraintInfo& ci){
    proc_.setup(target,dim,dimGen,ci);
    int_.setup(proc_);
  }


  inline int getDim() const {return proc_.dim();}

  inline void seed(const int seed){proc_.seed(seed);}

  void setPrintPrefix(const std::string prefix){
    printPrefix_ = prefix;
    int_.setPrintPrefix(prefix);
  }

  Eigen::VectorXd getProperty(const std::string which){
    Eigen::VectorXd ret;
    switch(plist_.id(which)){
    case 1 :
      ret.resize(1);
      ret(0) = int_.getAbsTol();
      break;
    case 2:
      ret.resize(1);
      ret(0) = int_.getRelTol();
      break;
    case 3:
      ret.resize(1);
      ret(0) = static_cast<double>(proc_.massAllowsFixedSubvector());
      break;
    case 5:
      ret = int_.CPUtime_;
      break;

    default :
      std::cout << "getProperty: no property/not readable : " << which << std::endl;

    }
    return(ret);
  }


  void setProperty(const std::string which,
                   const Eigen::VectorXd &val){
    switch(plist_.id(which)){
    case 0 :
      seed(static_cast<int>(val(0)));
      break;
    case 1:
      int_.setAbsTol(val(0));
      break;
    case 2:
      int_.setRelTol(val(0));
      break;
    case 4:
      proc_.massFixedMiSubvector(val);
      break;
    case 6:
      if(val.size()==proc_.lambda_.numPars()){
        proc_.lambda_.setPars(val);
      } else {
        std::cout << "bad number of parameters to lambda, should be " <<
          proc_.lambda_.numPars()<< ", ignored! "<< std::endl;
      }
      break;
    default :
      std::cout << "setProperty: no property/not writeable : " << which << std::endl;
    }
  }

  void setProperty(const std::string which,
                   const double val){
    Eigen::VectorXd valVec(1);
    valVec(0) = val;
    setProperty(which,valVec);
  }


  bool JSONparameters(const std::string filename){
    json_wrap jw(filename);
    if(! jw.isOpen()) return(false);

    std::vector<std::string> all_fields;
    jw.getAllNames(all_fields);


    int int_tmp;
    double double_tmp;
    Eigen::VectorXd vec_tmp;


    // seed
    if(jw.getNumeric("seed",int_tmp)){
      proc_.seed(int_tmp);
      __REMOVE_FIELD(all_fields,"seed")
    }

    // absTol
    if(jw.getNumeric("absTol",double_tmp)){
      int_.setAbsTol(std::abs(double_tmp));
      proc_.auxInt_.setAbsTol(std::abs(double_tmp));
      __REMOVE_FIELD(all_fields,"absTol")
    }

    // relTol
    if(jw.getNumeric("relTol",double_tmp)){
      int_.setRelTol(std::abs(double_tmp));
      proc_.auxInt_.setRelTol(std::abs(double_tmp));
      __REMOVE_FIELD(all_fields,"relTol")
    }

    // lambda Tuning parameters
    if(jw.getNumeric("lambda",vec_tmp)){
      if(vec_tmp.size()==proc_.lambda_.numPars()){
        proc_.lambda_.setPars(vec_tmp);
      } else {
        std::cout << "bad number of parameters to lambda, should be " <<
          proc_.lambda_.numPars()<< ", ignored! "<< std::endl;
      }
      __REMOVE_FIELD(all_fields,"lambda")
    }

    if(all_fields.size()>0){
      std::cout << "Warning: the following fields in JSON file *" << filename
                << "* \nwere not used by the sampler:\n";
      for(int i=0;i<all_fields.size();i++){
        std::cout << all_fields[i] << std::endl;
      }
      std::cout << std::endl;
    }
    return(true);
  }

  inline int targetCopies() const {return proc_.targetCopies();}

  void auxiliaryDiagnosticsInfo(jsonOut &outf) const {
    proc_.auxiliaryDiagnosticsInfo(outf);
    int_.auxiliaryDiagnosticsInfo(outf);
  }

  bool run(const int nSamples,
           const double Tmax,
           const double warmupFrac,
           const Eigen::Matrix<int,Eigen::Dynamic,1> &storePars,
           const Eigen::Matrix<double,Eigen::Dynamic,1 > &q0){


    proc_.registerDiagnostics(diag_);
    proc_.lastWarmupTime(warmupFrac*Tmax);
    int_.registerDiagnostics(diag_,0);


    //std::cout << "first eval" << std::endl;
    //std::cout << "q0 : \n" << q0 << std::endl;
    // try a first evaluation
    bool firstEvalGood = proc_.firstEval(q0);
    if(! firstEvalGood){
      std::cout << "bad initial state, exiting" << std::endl;
      return(false);
    }
    //std::cout << "done first eval" << std::endl;
    // get initial state


    odeState state0;
    proc_.SimulateIntialState(int_.odeOrder(),q0,state0);
    int_.setInitialState(state0);

    //std::cout << "done initial state" << std::endl;
    // acutal PDP simulation
    int_.run(storePars,Tmax,nSamples);

    return(true);
  }

  void diagnosticsToFile(const std::string filename){
    diag_.toFile(4,filename);
  }

  void samplesToFile(const int csvPrec,
                     const bool point, //otherwise integrated samples
                     const std::vector<std::string> colhead,
                     const std::string filename){

    size_t dd = point ? int_.samples_.rows() : int_.intSamples_.rows();
    if(colhead.size()>0 && dd == colhead.size()){
    Eigen::IOFormat CSV(csvPrec, Eigen::DontAlignCols, ", ", "\n");
    std::ofstream file;
    file.open(filename);
    file << "\"";
    for(int c=0;c<colhead.size()-1;c++) file << colhead[c] << "\" , \"" ;
    file << colhead[colhead.size()-1] << "\"" << std::endl;
    if(point){
      file << int_.samples_.transpose().format(CSV) << std::endl;
    } else {
      file << int_.intSamples_.transpose().format(CSV) << std::endl;
    }
    file.close();
    } else {
      if(point){
        std::cout << "no point samples written" << std::endl;
      } else {
        std::cout << "no integrated samples written" << std::endl;
      }

    }
  }

  void pointSamplesToFile(const std::string filename){
    int_.pointSamplesToFile(filename);
  }
  void intSamplesToFile(const std::string filename){
    int_.intSamplesToFile(filename);
  }

};


















#endif
