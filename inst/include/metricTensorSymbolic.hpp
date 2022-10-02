#ifndef __metricTensorSymbolic_hpp__
#define __metricTensorSymbolic_hpp__




class metricTensorSymbolic{
  size_t d_;
  size_t nz_;
  Eigen::Matrix<size_t,Eigen::Dynamic,1> diagPresent_;
  
  
  std::vector<size_t> scalarRow_;
  std::vector<size_t> scalarCol_;
  
  std::vector<size_t> denseDiagBlockStart_;
  std::vector<size_t> denseDiagBlockEnd_;
  
  std::vector<size_t> bandStart_;
  std::vector<size_t> bandEnd_;
  std::vector<size_t> bandBW_;
  
  
  
public:
  std::vector<size_t> triplet_i_;
  std::vector<size_t> triplet_j_;
  
  Eigen::Matrix<size_t,Eigen::Dynamic,1> activeDims_;
  Eigen::Matrix<size_t,Eigen::Dynamic,1> passiveDims_;
  
  metricTensorSymbolic() : nz_(0) {} 
  metricTensorSymbolic(const size_t d) : d_(d), nz_(0) {
    diagPresent_.resize(d_);
    diagPresent_.setZero();
  }
  void setup(const size_t d){ 
    d_ = d;
    diagPresent_.resize(d_);
    diagPresent_.setZero();
  }
  size_t dim() const {return d_;}
  size_t nz() const {return nz_;}
  template <class T>
  void pushScalar(const size_t row, 
                  const size_t col, 
                  T scalar){
    scalarRow_.push_back(row);
    scalarCol_.push_back(col);
    nz_+=2;
    if(row==col){
      diagPresent_(row)++;
      nz_-=1;
    }
    triplet_i_.push_back(std::min(row,col));
    triplet_j_.push_back(std::max(row,col));
  }
  
  template <class T>
  void pushDenseDiagBlock(const size_t start, 
                          const size_t end, 
                          Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> &block){
    denseDiagBlockStart_.push_back(start);
    denseDiagBlockEnd_.push_back(end);
    for(int i=start;i<=end;i++) diagPresent_(i)++;
    nz_ += (end-start+1)*(end-start+1);
    
    size_t len = end-start+1;
    for(int j=0;j<len;j++){
      for(int i=0;i<=j;i++){
        triplet_i_.push_back(start+i);
        triplet_j_.push_back(start+j);
      }
    }
    
  }
  
  template <class T>
  void pushBandDiagBlock(const size_t start, 
                         const size_t end,
                         const size_t bw,
                         Eigen::Matrix<T,Eigen::Dynamic,1> &blockVals){
    bandStart_.push_back(start);
    bandEnd_.push_back(end);
    bandBW_.push_back(bw);
    for(int i=start;i<=end;i++) diagPresent_(i)++;
    nz_ += (2*bw-1)*(end-start+1) - (bw-1)*bw;
    
    
    for(size_t b = 0;b<bw;b++){
      for(size_t col = start; col<=end-b; col++){
        triplet_i_.push_back(col);
        triplet_j_.push_back(col+b);
      } 
    }
    
    
  }
  
  template <class T>
  void pushDiagBlock(const size_t start, 
                     const size_t end,
                     Eigen::Matrix<T,Eigen::Dynamic,1> &blockVals){
    pushBandDiagBlock(start,end,1,blockVals);
  }
  
  
  void prepareCommon(){
    
    size_t numActive = (diagPresent_.array()>0).count();
    if(numActive==0){
      std::cout << "WARNING: Using a RMHMCprocess without specifying any metric tensor diagonal elements" << std::endl;
      std::cout << "consider switching to a HMCprocess which may be more efficient!" << std::endl;
    }
    
    activeDims_.resize(numActive);
    passiveDims_.resize(d_-numActive);
    size_t ca = 0;
    size_t cp = 0;
    for(size_t i = 0;i<d_;i++){
      if(diagPresent_(i)>0){
        activeDims_(ca) = i;
        ca++;
      } else {
        passiveDims_(cp) = i;
        cp++;
      }
    }
  }
  
  void dumpTriplet() {
    Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> m(d_,d_);
    m.setZero();
    for(int i=0;i<triplet_i_.size();i++){
      m(triplet_i_[i],triplet_j_[i]) += 1;
    }
    std::cout << "metricTensorSymbolic: triplet pattern: \n" << m << std::endl;
  }
  
  void dumpPattern() {
    prepareCommon();
    Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> m(d_,d_);
    m.setZero();
    
    
    // dense diagonal blocks
    for(int i=0;i<denseDiagBlockStart_.size();i++){
      for(int ii=denseDiagBlockStart_[i];ii<=denseDiagBlockEnd_[i];ii++){
        for(int jj=denseDiagBlockStart_[i];jj<=denseDiagBlockEnd_[i];jj++) m(ii,jj)++;
      }
    }
    
    // scalar entries
    for(int i=0;i<scalarRow_.size();i++){ 
      m(scalarRow_[i],scalarCol_[i])++;
      if(scalarRow_[i] != scalarCol_[i]) m(scalarCol_[i],scalarRow_[i])++;
    }
    
    // band blocks
    for(int i=0; i<bandStart_.size(); i++){
      for(int b=0; b<bandBW_[i]; b++){
        for(int col=bandStart_[i]; col<=bandEnd_[i]-b; col++){
          m(col+b,col)++;
          if(b>0) m(col,col+b)++;
        }
      } 
    }
    
    // dump to terminal
    std::cout << "metricTensorSymbolic pattern : " << std::endl;
    std::cout << m << std::endl;
    std::cout << "diagonals present:" << std::endl;
    std::cout << diagPresent_ << std::endl;
  }
  
  
  
};





#endif


