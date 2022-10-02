#ifndef _metricTensorDummy_hpp_
#define _metricTensorDummy_hpp_

/*
 * Placeholder class to be used with the non-Riemannian process specifications
 * 
 */


class metricTensorDummy{
  bool toPrint_;
  inline void print(){
    if(toPrint_){
      std::cout << "model class contains non-trivial metric tensor class method(s),\nplease remove for better performance!!\n" ;
      toPrint_ = false;
    }
  }
public:
  metricTensorDummy() : toPrint_(true) {}
  template <class T2>
  void pushScalar(const size_t row, 
                  const size_t col, 
                  T2 scalar){
    print();
  }
  
  template <class T2>
  void pushDenseDiagBlock(const size_t start, 
                          const size_t end, 
                          Eigen::Matrix<T2,Eigen::Dynamic,Eigen::Dynamic> &block){
    print();
  }
  template <class T2>
  void pushBandDiagBlock(const size_t start, 
                         const size_t end,
                         const size_t bw,
                         Eigen::Matrix<T2,Eigen::Dynamic,1> &blockVals){
    print();
  }
  
};


#endif



