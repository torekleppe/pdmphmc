#ifndef _AMTMULTINORMALCOV_HPP_
#define _AMTMULTINORMALCOV_HPP_

namespace amt{



template <class argType, class muType, class sigmaType>
inline stan::math::var multi_normal_lpdf_StanVal(const Eigen::Matrix<argType,Eigen::Dynamic,1>& arg,
                                          const Eigen::Matrix<muType,Eigen::Dynamic,1>& mu,
                                          const SPDmatrix<sigmaType>& Sigma){
  if(arg.size()!=Sigma.dim()){
    throw std::runtime_error("multi_normal_lpdf_StanVal : arg incompatible with precision matrix dimension");
  } else if(mu.size()!=Sigma.dim()){
    throw std::runtime_error("multi_normal_lpdf_StanVal : mu incompatible with precision matrix dimension");
  }

  Eigen::Matrix<typename amtNumType2<argType,muType>::type,Eigen::Dynamic,1> tmp(arg.size());
  for(std::size_t j=0;j<arg.rows();j++) tmp.coeffRef(j) = asStanVar(arg.coeff(j)) - asStanVar(mu.coeff(j));

  stan::math::var ret = -0.5*Sigma.quad_form_inv_StanVal(tmp,0);

  ret += -0.5*(Sigma.logDet_StanVal() + static_cast<double>(arg.size())*1.8378770664093454836);
  return(ret);
}



template <class argType, class muType, class sigmaType, class tenPtrType>
inline void multi_normal_LGC(const Eigen::Matrix<argType,Eigen::Dynamic,1>* argPtr,
                      const Eigen::Matrix<argType,Eigen::Dynamic,1>* muPtr,
                      const SPDmatrix<sigmaType>* SigmaPtr,
                      tenPtrType tensor){
  if constexpr(std::is_same_v<amtVar,argType> || std::is_same_v<amtVar,muType>){

  }

}




template <class argType, class muType, class SigmaType>
class multi_normal_ld{
public:
  multi_normal_ld(const Eigen::Matrix<argType,Eigen::Dynamic,1>& arg,
                  const Eigen::Matrix<muType,Eigen::Dynamic,1>& mu,
                  const SPDmatrix<SigmaType>& Sigma){}
};






} // namespace

#endif
