#ifndef _METRICTENSORSPARSE_HPP_
#define _METRICTENSORSPARSE_HPP_

template <class numType>
class metricTensorSparseNumeric{
  sparseCholSymbolic *csym_;
  metricTensorSymbolic *sym_;
  size_t writeC_;

public:
  Eigen::Matrix<numType,Eigen::Dynamic,1> Ax_;

  sparseCholNumeric<numType> L_;

  metricTensorSparseNumeric(sparseCholSymbolic &csym,
                            metricTensorSymbolic &sym) : csym_(&csym), sym_(&sym), writeC_(0) {}
  inline void setup(){
    L_.setup(*csym_);
    Ax_.resize((*csym_).nz_);
  }
  inline void zeroG(){
    Ax_.setZero();
    writeC_ = 0;
  }

  inline bool isFinite(){
    for(size_t i=0;i<Ax_.size();i++){
      if(! std::isfinite(doubleValue(Ax_.coeff(i)))) return(false);
    }
    return(true);
  }

  template <class TMType>
  void finalize(TMType &TM){
    size_t ii;
    for(size_t i=0;i<(*sym_).passiveDims_.size();i++){
      ii = (*sym_).passiveDims_.coeff(i);
      Ax_.coeffRef(writeC_) = TM.massDiag(ii);
      writeC_++;
    }
  }

  void dumpG(){
    Eigen::MatrixXd tmpMat((*csym_).n_,(*csym_).n_);
    tmpMat.setZero();
    for(int i=0;i<Ax_.size();i++){
      tmpMat((*csym_).i_[i],(*csym_).j_[i]) += doubleValue(Ax_[i]);
    }
    std::cout << "metricTensorSparseNumeric, dump of G\n" << tmpMat << std::endl;
  }


  inline bool chol(){
    bool eflag = L_.chol(*csym_,Ax_);
    //L_.dumpLDL();
    return(eflag);
  }


  template <class T2>
  void pushScalar(const size_t row,
                  const size_t col,
                  T2 scalar){
    Ax_.coeffRef(writeC_) = scalar;
    writeC_++;
  }

  template <class T2>
  void pushDenseDiagBlock(const size_t start,
                          const size_t end,
                          const Eigen::Matrix<T2,Eigen::Dynamic,Eigen::Dynamic> &block){

    size_t len = end-start+1;
    if(block.cols()!=len || block.rows()!=len){
      std::cout << "WARNING : pushDenseDiagBlock : bad dimension of block!"  << std::endl;
    }

    //G_.block(start,start,len,len) += block;
    for(int j=0;j<len;j++){
      for(int i=0;i<=j;i++){
        Ax_.coeffRef(writeC_) = block.coeff(i,j);
        writeC_++;
      }
    }
  }
  template <class T>
  void pushBandDiagBlock(const size_t start,
                         const size_t end,
                         const size_t bw,
                         Eigen::Matrix<T,Eigen::Dynamic,1> &blockVals){
    // expected number of elements
    size_t tmp = (end-start+1);
    size_t numElem = tmp*bw - (bw-1)*bw/2;
    if(blockVals.size() != numElem){
      std::cout << "WARNING : pushBandDiagBlock : bad dimension of blockVals!"  << std::endl;
    }
    Ax_.segment(writeC_,numElem) = blockVals;
    writeC_ += numElem;
  }

  template <class T2>
  inline Eigen::Matrix<typename boost::math::tools::promote_args<numType, T2>::type,Eigen::Dynamic,1>
  solveG(const Eigen::Matrix<T2,Eigen::Dynamic,1> &b) const {
    //Eigen::Matrix<typename boost::math::tools::promote_args<T, T2>::type,Eigen::Dynamic,1> tmp = solveL(b);
    return(L_.solve(b));
  }

  inline numType logDetL() const {
    return(L_.logDetL());
  }

  /*
   * Interface similar to the mass matrices
   * Notice, treats matrix numeric type as double, and
   * should not be used within derivative calulations
   */
  Eigen::VectorXd sqrtMtmp_;
  void sqrtM(Eigen::VectorXd &x) {
    sqrtMtmp_ = x;
    L_.applyL(sqrtMtmp_,x);
  }
};



class metricTensorSparse{
  Eigen::VectorXi t_i_,t_j_;
public:
  sparseCholSymbolic csym_;
  metricTensorSymbolic sym;
  metricTensorSparseNumeric<stan::math::var> adv;
  metricTensorSparseNumeric<double> dbv;

  metricTensorSparse() : adv(csym_,sym), dbv(csym_,sym) {}

  // before run with symbolic type
  void allocate(const size_t dim){
    sym.setup(dim);
  }

  void copyLtoDouble(){
    for(int i=0;i<adv.L_.Lx_.size();i++){
      dbv.L_.Lx_.coeffRef(i) = adv.L_.Lx_.coeff(i).val();
    }
    for(int i=0;i<adv.L_.D_.size();i++){
      dbv.L_.D_.coeffRef(i) = adv.L_.D_.coeff(i).val();
    }
    dbv.L_.Li_ = adv.L_.Li_;
  }
  void copyGtoDouble(){
    for(size_t i=0;i<adv.Ax_.size();i++) {
      dbv.Ax_.coeffRef(i) = adv.Ax_.coeff(i).val();
    }
  }

  void symbolicAnalysis(){
//    std::cout << "metricTensorSparse: symbolic analysis" << std::endl;
    sym.prepareCommon();

    int triplet_len = sym.triplet_i_.size();
    int total_len = triplet_len+sym.passiveDims_.size();
    t_i_.resize(total_len);
    t_j_.resize(total_len);

    // triplet structure from model spec
    for(int i=0;i<triplet_len;i++) t_i_.coeffRef(i) = sym.triplet_i_[i];
    for(int i=0;i<triplet_len;i++) t_j_.coeffRef(i) = sym.triplet_j_[i];

    // triplet from static dimensions
    for(int i=0;i<sym.passiveDims_.size();i++){
      t_i_.coeffRef(triplet_len+i) = sym.passiveDims_.coeff(i);
      t_j_.coeffRef(triplet_len+i) = sym.passiveDims_.coeff(i);
    }

    // do the Cholesky symbolic analysis
    csym_ = sparseCholSymbolic(sym.dim(),t_i_,t_j_);

    // allocate the numerical objects
    adv.setup();
    dbv.setup();
#ifdef __TENSOR_DEBUG__
    std::cout << "csym triplet \n";
    csym_.dumpTripletPattern();
    if(sym.passiveDims_.size()>0){
      std::cout << "using fixed mass"
                << " on dimensions :" << std::endl;
      std::cout << sym.passiveDims_ << std::endl;
    }
#endif
  }

  inline void toJSON(jsonOut& jso) const {
    jso.push("triplet_i",csym_.i_);
    jso.push("triplet_j",csym_.j_);
    std::vector<int> ii,jj;
    adv.L_.tripletL(ii,jj);
    jso.push("L_triplet_i",ii);
    jso.push("L_triplet_j",jj);
  }

};



#endif
