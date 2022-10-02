#ifndef _METRICTENSORDENSE_HPP_
#define _METRICTENSORDENSE_HPP_
/*
 * Dense metric tensor based on stan::math dense Cholesky factorization
 *
 */
template <class T>
class metricTensorDenseNumeric{
  metricTensorSymbolic *sym_;
  Eigen::VectorXd tmpVec_;

public:
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> G_;
  Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> L_;

  metricTensorDenseNumeric() {}
  metricTensorDenseNumeric(metricTensorSymbolic &sym) : sym_(&sym) {}
  void setup(){
    G_.resize((*sym_).dim(),(*sym_).dim());
    G_.setZero();
    L_.resize((*sym_).dim(),(*sym_).dim());
    L_.setZero();
  }

  inline void zeroG(){ G_.setZero();}

  inline bool isFinite(){
    for(int j=0;j<G_.cols();j++){
      for(int i=0;i<G_.rows();i++){
        if(! std::isfinite(doubleValue(G_.coeff(i,j)))) return(false);
      }
    }
    return(true);
  }


  template <class TMType>
  void finalize(TMType &TM){
    size_t ii;
    for(size_t i=0;i<(*sym_).passiveDims_.size();i++){
      ii = (*sym_).passiveDims_.coeff(i);
      G_.coeffRef(ii,ii) = TM.massDiag(ii);
    }
  }

  /*
   void setPassive(const Eigen::VectorXd vals){
   if(vals.size() != (*sym_).passiveDims_.size()){
   std::cout << "WARNING : setPassive : bad dimension of vals" << std::endl;
   }
   size_t ii;
   for(size_t i=0;i<vals.size();i++) {
   ii =  (*sym_).passiveDims_.coeff(i);
   G_.coeffRef(ii,ii)=vals.coeff(i);
   }
   }
   */
  template <class T2>
  void pushScalar(const size_t row,
                  const size_t col,
                  const T2 scalar){
    G_.coeffRef(row,col) += scalar;
    if(row!=col) G_.coeffRef(col,row) += scalar;
  }

  template <class T2>
  void pushDenseDiagBlock(const size_t start,
                          const size_t end,
                          Eigen::Matrix<T2,Eigen::Dynamic,Eigen::Dynamic> &block){

    size_t len = end-start+1;
    if(block.cols()!=len || block.rows()!=len){
      std::cout << "WARNING : pushDenseDiagBlock : bad dimension of block!"  << std::endl;
    }
    G_.block(start,start,len,len) += block;
  }

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
    size_t count = 0;
    for(size_t b = 0;b<bw;b++){
      for(size_t col = start; col<=end-b; col++){
        G_.coeffRef(col+b,col) += blockVals.coeff(count);
        count++;
      }
      if(b>0) for(size_t col = start; col<=end-b; col++) G_.coeffRef(col,col+b) = G_.coeff(col+b,col);
    }
  }

  inline bool chol(){
    try{
      L_ = stan::math::cholesky_decompose(G_);
    }
    catch(...){
      std::cout << "metricTensorDense::chol() matrix not SPD" << std::endl;
      dumpG();
      return(false);
    }
    return(true);
  }



  template <class T2>
  inline Eigen::Matrix<typename boost::math::tools::promote_args<T, T2>::type,Eigen::Dynamic,1>
  solveL(const Eigen::Matrix<T2,Eigen::Dynamic,1> &b) const {
    return(stan::math::mdivide_left_tri_low(L_,b));
  }

  template <class T2>
  inline Eigen::Matrix<typename boost::math::tools::promote_args<T, T2>::type,Eigen::Dynamic,1>
  solveLT(const Eigen::Matrix<T2,Eigen::Dynamic,1> &b) const {
    Eigen::Matrix<T2,1,Eigen::Dynamic> bT(b.size());
    for(int i=0;i<b.size();i++) bT.coeffRef(i) = b.coeff(i);
    bT = stan::math::mdivide_right_tri_low(bT,L_);
    return(bT.transpose());
  }

  template <class T2>
  inline Eigen::Matrix<typename boost::math::tools::promote_args<T, T2>::type,Eigen::Dynamic,1>
  applyL(const Eigen::Matrix<T2,Eigen::Dynamic,1> &b) const {
    return(L_*b);
  }



  template <class T2>
  inline Eigen::Matrix<typename boost::math::tools::promote_args<T, T2>::type,Eigen::Dynamic,1>
  applyLT(const Eigen::Matrix<T2,Eigen::Dynamic,1> &b) const {
    return(L_.transpose()*b);
  }

  template <class T2>
  inline Eigen::Matrix<typename boost::math::tools::promote_args<T, T2>::type,Eigen::Dynamic,1>
  solveG(const Eigen::Matrix<T2,Eigen::Dynamic,1> &b) const {
    Eigen::Matrix<typename boost::math::tools::promote_args<T, T2>::type,Eigen::Dynamic,1> tmp = solveL(b);
    return(solveLT(tmp));
  }

  template <class T2>
  inline Eigen::Matrix<typename boost::math::tools::promote_args<T, T2>::type,Eigen::Dynamic,1>
  applyG(const Eigen::Matrix<T2,Eigen::Dynamic,1> &b) const {
    return(G_*b);
  }

  inline T logDetL() const {
    return(L_.diagonal().array().log().sum());
  }

  void dumpG(){
    std::cout << "G = " << std::endl;
    std::cout << G_ << std::endl;
  }
  void dumpL(){
    std::cout << "L = " << std::endl;
    std::cout << L_ << std::endl;
    std::cout << "LL^T = " << std::endl;
    std::cout << L_*L_.transpose() << std::endl;
  }

  /*
   * Interface similar to the mass matrices
   * Notice, treats matrix numeric type as double, and
   * should not be used within derivative calulations
   */
  void sqrtM(Eigen::VectorXd &x) {
    if(tmpVec_.size()!=L_.rows()) tmpVec_.resize(L_.rows());
    tmpVec_.setZero();
    for(size_t i=0;i<L_.rows();i++){
      for(size_t j=0;j<=i;j++) {
        tmpVec_.coeffRef(i) += doubleValue(L_.coeff(i,j))*x.coeff(j);
      }
    }
    x=tmpVec_;
  }
};




class metricTensorDense{
  //Eigen::VectorXd mon_q_,mon_p_,mon_qdot_,mon_pdot_;
public:
  metricTensorSymbolic sym;
  metricTensorDenseNumeric<stan::math::var> adv;
  metricTensorDenseNumeric<double> dbv;

  metricTensorDense() : adv(sym), dbv(sym) {}

  // before run with symbolic type
  void allocate(const size_t dim){
    sym.setup(dim);
    adv.setup();
    dbv.setup();
  }

  void copyLtoDouble(){
    for(int j=0;j<adv.L_.cols();j++){
      for(int i=j;i<adv.L_.rows();i++){
        dbv.L_.coeffRef(i,j) = adv.L_.coeff(i,j).val();
      }
    }
  }

  void copyGtoDouble(){
    for(int j=0;j<adv.G_.cols();j++){
      for(int i=0;i<adv.G_.rows();i++){
        dbv.G_.coeffRef(i,j) = adv.G_.coeff(i,j).val();
      }
    }
  }

  // after run with symbolic type
  void symbolicAnalysis(){

    sym.prepareCommon();

    double dens = static_cast<double>(sym.nz())/pow(static_cast<double>(sym.dim()),2);
    std::cout << "Symbolic analysis for dense metric tensor done" << std::endl;
    std::cout << "# non-zero elements: " << sym.nz() << " fraction non-zero: " << dens << std::endl;
    if(dens<=0.5 && sym.dim()>10) std::cout << "Consider using a sparse metric tensor" << std::endl;

    std::cout << "# passive dimensions: " << sym.passiveDims_.size() << std::endl;

    if(sym.passiveDims_.size()>0){
      std::cout << "using fixed mass"
                << " on dimensions :" << std::endl;
      std::cout << sym.passiveDims_ << std::endl;
      // auxiliary mass matrix for the dimension
    }
  }

  inline bool hasPassiveDims() const {return sym.passiveDims_.size()>0;}

  inline void toJSON(jsonOut& jso) const {}


};

#endif
