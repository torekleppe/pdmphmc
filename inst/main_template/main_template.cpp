
#include "PDMPHMC.hpp"

#define _PDMPHMC_DEBUG_

// utilities for keeping track of data
amtData dta__;


// include model specification
#include "___model_specification.hpp"
// include typedefs for sampler settings
#include "___model_typedefs.hpp"


int main(int argc, char *argv[]){
  // common
  std::string fileNameBase(argv[0]);
  // remove .exe in case of windows
  if(fileNameBase.length()>3){
  std::string last4 = fileNameBase.substr(fileNameBase.length()-4);
  if(last4.compare(".exe")==0 || last4.compare(".EXE")==0){
    fileNameBase = fileNameBase.substr(0,fileNameBase.length()-4);
  }
  }
  std::cout << "fileNameBase : " << fileNameBase << std::endl;

  // chain id from command line
  int chain_id = 0;
  if(argc>1) chain_id = atoi(argv[1]);



  // model specification
  modelSpec__ m;

  // read data from file
  int dataEflag = dta__.fromFile(fileNameBase+"_data.json");
  if(dataEflag != 0 ) return(dataEflag);
  std::cout << "reading of data done" << std::endl;

  // run the preprocess function
  m.preProcess();


  // run the model once to calculate dimensions, collect names and so on
  metricTensorDummy mtd;
  amtModel<varType,metricTensorDummy,true> modelSymb(mtd);
  m(modelSymb);
  modelSymb.finalize();
  //modelSymb.showSummary();

  size_t dim = modelSymb.dim();
  size_t dimGen = modelSymb.dimGen();

  typedef diagLinearTM_VARI TMtype;

  PDPsampler<ProcessType__,
             modelSpec__,
             varType,
             rungeKuttaSolver,
             RKDP54,
             metricTensorType__,
             TMtype,
             constantLambda,
             diagnostics> sampler;

  sampler.setup(m,dim,dimGen);
  sampler.setPrintPrefix("chain #" + std::to_string(chain_id));


  json_wrap ctrl(fileNameBase+"_control.json");
  std::cout << "reading of control done" << std::endl;

  /*----------------------------------------------------
   * Work out which parameters to store
   *
   *----------------------------------------------------*/
  Eigen::VectorXi storePars;
  std::vector<std::string> storeParsNames;
  if(ctrl.getStringVec("store.pars",storeParsNames)){
    std::cout << "store.pars : "<< std::endl;
    for(int i=0;i<storeParsNames.size();i++){
      std::cout << storeParsNames[i] << std::endl;
      modelSymb.getStorePars(storeParsNames,storePars);
    }
  } else {
    // otherwise store all parameters
    storePars.resize(dim);
    for(size_t i=0;i<dim;i++) storePars.coeffRef(i) = i;
  }

  /*-----------------------------------------------------
   * Which parameters should have fixed scaling
   *
   *----------------------------------------------------*/
  Eigen::VectorXd miDiag;
  miDiag.setConstant(dim,-1.0);
  if(ctrl.hasMember("fixedMiDiag")){
    if(sampler.getProperty("massallowsFixedSubvector").coeff(0)>0.5){
      modelSymb.getMiDiag(ctrl,miDiag);
      sampler.setProperty("massFixedMiSubvector",miDiag);
    } else {
      std::cout << "sampler settings does not permit fixedMiDiag, ignored" << std::endl;
    }
  }
  /*----------------------------------------------------
   * Get basic sampler parameters
   *
   *----------------------------------------------------*/
  Eigen::VectorXd tmp_d;
  Eigen::VectorXi tmp_i;

  double warmupFrac = 0.5;
  if(ctrl.getNumeric("warmupFrac",tmp_d) && tmp_d.size()==1) warmupFrac = tmp_d(0);

  double Tmax = 10000.0;
  if(ctrl.getNumeric("Tmax",tmp_d) && tmp_d.size()==1) Tmax = tmp_d(0);

  int samples = 2000;
  if(ctrl.getNumeric("samples",tmp_i) && tmp_i.size()==1) samples = tmp_i(0);

  int csvPrec = 8;
  if(ctrl.getNumeric("csvPrec",tmp_i) && tmp_i.size()==1) csvPrec = tmp_i(0);

  /*-------------------------------------------------------------------
   * Set additional parameters
   *-------------------------------------------------------------------*/

  int seed = 1000*chain_id;
  if(ctrl.getNumeric("seed",tmp_i) && tmp_i.size()==1) seed = tmp_i(0);
  sampler.seed(seed+1000*chain_id);

  if(ctrl.getNumeric("absTol",tmp_d) && tmp_d.size()==1) sampler.setProperty("absTol",tmp_d);

  if(ctrl.getNumeric("relTol",tmp_d) && tmp_d.size()==1) sampler.setProperty("relTol",tmp_d);

  if(ctrl.getNumeric("lambda",tmp_d)) sampler.setProperty("lambda",tmp_d);

/*--------------------------------------------------------------------
 *  work out initial configuration
 *-------------------------------------------------------------------*/


  Eigen::VectorXd q0(dim);
  modelSymb.getDefaultVals(q0);
#ifndef _NO_IPS_
  initialPointSolver<modelSpec__> ips(m,dim,dimGen);
  ips.seed(seed+1000*chain_id+13);
  if(ips.run(q0)) q0 = ips.bestQ();
#endif


  /*-------------------------------------------------------------------
   * Run the sampler
   *-------------------------------------------------------------------*/

  sampler.run(samples,Tmax,warmupFrac,storePars,q0);

  /*--------------------------------------------------------------
   *  return output to files
   *---------------------------------------------------------------*/




  sampler.diagnosticsToFile(fileNameBase+"_"+std::to_string(chain_id)+"_diagnostics.csv");
  //std::cout << "done diagnostics" << std::endl;
  sampler.samplesToFile(csvPrec,true,modelSymb.storeColNames(storePars),
                        fileNameBase+"_"+std::to_string(chain_id)+"_point.csv");
  //std::cout << "done point samples, egn_" << modelSymb.egn_.size() << std::endl;
  sampler.samplesToFile(csvPrec,false,modelSymb.egn_,
                        fileNameBase+"_"+std::to_string(chain_id)+"_int.csv");
  //std::cout << "done integrated samples" << std::endl;


  jsonOut outf;
  modelSymb.auxToJSON(outf);

  //std::cout << "done auxToJSON " << std::endl;

  outf.push("chain",chain_id);
  outf.push("CPUtime",sampler.getProperty("CPUtime"));
  outf.push("miDiag",miDiag);
  sampler.auxiliaryDiagnosticsInfo(outf);

  //std::cout << "done sampler aux " << std::endl;

  outf.toFile(fileNameBase+"_"+std::to_string(chain_id)+"_auxInfo.json");

  return(0);
}

