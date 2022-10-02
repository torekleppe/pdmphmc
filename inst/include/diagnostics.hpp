#ifndef _DIAGNOSTICS_HPP_
#define _DIAGNOSTICS_HPP_

#define __DIAGNOSTICS_BLOCK_SIZE__ 1000
#define __DIAGNOSTICS_WIDTH__ 20

class diagnostics{
  Eigen::MatrixXd diag_;
  std::vector<std::string> colNames_;
  int row_;
  
public:
  diagnostics(){
    diag_.resize(__DIAGNOSTICS_BLOCK_SIZE__,__DIAGNOSTICS_WIDTH__);
    diag_.setZero();
    row_ = 0;
  }
  
  void push(const std::string &name,
            const double value){
    auto whichCol = std::find(colNames_.begin(),colNames_.end(),name);
    int col = whichCol-colNames_.begin();
    if(whichCol != colNames_.end()){
      // property already in
      diag_.coeffRef(row_,col) = value;
    } else {
      // add new column
      colNames_.push_back(name);
      col = colNames_.size()-1;
      if(col==diag_.cols()){
        diag_.conservativeResize(diag_.rows(),diag_.cols()+__DIAGNOSTICS_WIDTH__);
        diag_.rightCols(__DIAGNOSTICS_WIDTH__).setZero();
      }
      diag_.coeffRef(row_,col) = value;
    }
  }
  
  void push(const std::string &name,
            const int value){
    push(name,static_cast<double>(value));
  }
  
  void push(const std::string &nameBase,
            const Eigen::VectorXd &value){
    for(int i=0;i<value.size();i++) push(nameBase+"_"+std::to_string(i),value(i));
  }
  
  
  void newRow(){
    row_++;
    if(row_==diag_.rows()){
      diag_.conservativeResize(diag_.rows()+__DIAGNOSTICS_BLOCK_SIZE__,diag_.cols());
      diag_.bottomRows(__DIAGNOSTICS_BLOCK_SIZE__).setZero();
    }
  }
  void dump(){
    std::cout << "dump of diagnostics object:" << std::endl;
    for(int i=0;i<colNames_.size();i++){
      std::cout << colNames_[i] << "\t";
    }
    std::cout << std::endl;
    std::cout << diag_.block(0,0,row_+1,colNames_.size()) << std::endl;
    std::cout << "(total storage: " << diag_.rows() << " x "<< diag_.cols() << ")" << std::endl;
  }
  
  void toFile(const int csvPrec, 
              const std::string filename){
    Eigen::IOFormat CSV(csvPrec, Eigen::DontAlignCols, ", ", "\n");
    std::ofstream file;
    file.open(filename);
    file << "\"";
    for(int c=0;c<colNames_.size()-1;c++) file << colNames_[c] << "\" , \"" ;
    file << colNames_[colNames_.size()-1] << "\"" << std::endl;
    file << diag_.block(0,0,row_+1,colNames_.size()).format(CSV) << std::endl;
    file.close();
  }
  
};

/*
 * Used when not storing any diagnostics info, hopefully the compiler will optimize this
 * class completely away
 */

class diagnosticsOff{
public:
  diagnosticsOff(){}
  inline void push(const std::string &name,const double value) const {}
  inline void push(const std::string &nameBase,const Eigen::VectorXd &value) const {}
  inline void newRow() const {}
  inline void dump() const {
    std::cout << "No diagnostics info stored!" << std::endl;
  }
  inline void toFile(const int csvPrec, const std::string filename) const {
    std::cout << "No diagnostics info stored!" << std::endl;
  }
};


#endif




