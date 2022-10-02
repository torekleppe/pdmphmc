#ifndef _AMTSTOCHVOLLEVERAGE_HPP_
#define _AMTSTOCHVOLLEVERAGE_HPP_


// optimized version of the stochastic volatility observation likelihood
// y_t \sim N(rho*(z_1-z_0)*exp(0.5*z0)/sz,(1.0-square(rho))*exp(z0)))

namespace amt{

template <class varType>
class stochVolLeverageObs_ld{
public:
  stochVolLeverageObs_ld(const double y,
                         const varType& z0,
                         const varType& z1,
                         const varType& rho,
                         const varType& sz){}
};

template <>
class stochVolLeverageObs_ld<amtVar>{
  const double y_;
  const amtVar* z0ptr_;
  const amtVar* z1ptr_;
  const amtVar* rhoptr_;
  const amtVar* szptr_;
public:
  stochVolLeverageObs_ld(const double y,
                         const amtVar& z0,
                         const amtVar& z1,
                         const amtVar& rho,
                         const amtVar& sz) : y_(y), z0ptr_(&z0), z1ptr_(&z1), rhoptr_(&rho), szptr_(&sz){}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    stan::math::var t1 = stan::math::exp(0.5*z0ptr_->val_); //exp(0.5*z0)
    stan::math::var expz = stan::math::square(t1);
    stan::math::var dif = z1ptr_->val_ - z0ptr_->val_; // z1-z0
    stan::math::var t2 = t1/szptr_->val_; //exp(0.5*z0)/sz
    stan::math::var t3 = t2*rhoptr_->val_; // exp(0.5*z0)*rho/sz = Jm_z1
    stan::math::var ret_m = t3*dif;
    stan::math::var ret_v = expz*(1.0-square(rhoptr_->val_));
    stan::math::var inv_ret_v_sq = 0.5*stan::math::inv(stan::math::square(ret_v));
    stan::math::var Jm_z0 = 0.5*ret_m - t3;
    stan::math::var Jm_rho = t2*dif;
    stan::math::var Jm_sz = -ret_m/szptr_->val_;
    stan::math::var Jv_rho = -2.0*expz*rhoptr_->val_;

    sparseVec::syr(z0ptr_->Jac_,stan::math::square(Jm_z0)/ret_v+0.5,tensor);
    sparseVec::syr2(z0ptr_->Jac_,
                    z1ptr_->Jac_,
                    Jm_z0*t3/ret_v,
                    tensor);
    sparseVec::syr2(z0ptr_->Jac_,
                    rhoptr_->Jac_,
                    (Jm_z0*Jm_rho +  0.5*Jv_rho)/ret_v,
                    tensor);
    sparseVec::syr2(z0ptr_->Jac_,
                    szptr_->Jac_,
                    Jm_z0*Jm_sz/ret_v,
                    tensor);
    sparseVec::syr(z1ptr_->Jac_,stan::math::square(t3)/ret_v,tensor);
    sparseVec::syr2(z1ptr_->Jac_,
                    rhoptr_->Jac_,
                    t3*Jm_rho/ret_v,
                    tensor);
    sparseVec::syr2(z1ptr_->Jac_,
                    szptr_->Jac_,
                    t3*Jm_sz/ret_v,
                    tensor);
    sparseVec::syr(rhoptr_->Jac_,
                   stan::math::square(Jm_rho)/ret_v + 0.5*stan::math::square(Jv_rho/ret_v),
                   tensor);
    sparseVec::syr2(rhoptr_->Jac_,
                    szptr_->Jac_,
                    Jm_rho*Jm_sz/ret_v,
                    tensor);
    sparseVec::syr(szptr_->Jac_,
                   stan::math::square(Jm_sz)/ret_v,
                   tensor);

    return(stan::math::normal_lpdf(y_,ret_m,stan::math::sqrt(ret_v)));
  }
};



template <>
class stochVolLeverageObs_ld<stan::math::var>{
  stan::math::var lpdf_;
public:
  stochVolLeverageObs_ld(const double y,
                         const stan::math::var& z0,
                         const stan::math::var& z1,
                         const stan::math::var& rho,
                         const stan::math::var& sz){
    stan::math::var t1 = stan::math::exp(0.5*z0);
    lpdf_ = stan::math::normal_lpdf(y,rho*t1*(z1-z0)/sz,t1*stan::math::sqrt(1.0-square(rho)));
  }
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    return(lpdf_);
  }
};


} // namespace
#endif

