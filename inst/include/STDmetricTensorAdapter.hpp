#ifndef _STDMETRICTENSORADAPTER_HPP_
#define _STDMETRICTENSORADAPTER_HPP_

/*
 * adapter to make the (transport map-)standardized metric tensor
 * behave as a mass matrix (used for standardized lambda-classes)
 * 
 * 
 */

template <class metricTensorType,
          class linearTMType>
class STDmetricTensorAdapter{
  metricTensorType *tensor_;
  linearTMType *TM_;
public:
  STDmetricTensorAdapter() {}
  void setup(metricTensorType &tensor,
             linearTMType &TM){
    tensor_ = &tensor;
    TM_ = &TM;
  }
  
  void sqrtM(Eigen::VectorXd &var) const {
    (*tensor_).dbv.sqrtM(var);
    (*TM_).toParJacTransposed(var);
  }
};
#endif