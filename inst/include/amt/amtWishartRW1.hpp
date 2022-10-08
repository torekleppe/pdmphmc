#ifndef _AMTWISHARTRW1_HPP_
#define _AMTWISHARTRW1_HPP_


namespace amt{


#ifdef __WISHARTRW1_VARIANT2__

template <class vType,class tenPtrType>
stan::math::var __wishartRW1_core(const SPDmatrix<vType>* arg,
                                  const SPDmatrix<vType>* mean,
                                  const vType& nu,
                                  tenPtrType tensor){
  std::size_t n = arg->dim();
  double dn = static_cast<double>(n);
  stan::math::var lpdf = 0.0;
  stan::math::var shp;
  stan::math::var scal;
  stan::math::var nu_s = asStanVar(nu);
  stan::math::var ctmp;
  stan::math::var ttmp;
  stan::math::var mtmp = -0.5*dn/nu_s;
  stan::math::var t1 = 0.5*(nu_s+dn+1.0);
  stan::math::var t2,t3;
  // marginals
  //


  for(int i = 0;i<n;i++){
    shp = 0.5*(nu_s-static_cast<double>(i));
    scal = 2.0*mean->Lambda_StanVal(i)/nu_s;
    lpdf += -shp*log(scal) + shp*arg->x_StanVal(i) - arg->Lambda_StanVal(i)/scal - stan::math::lgamma(shp);

    if constexpr(std::is_same_v<amtVar,vType>){
      t3 = (0.5*(1.0-static_cast<double>(i+1)))/nu_s;
      mtmp += 0.25*stan::math::trigamma(shp) + t3/nu_s;
      //t2 = 0.5*(nu_s + 1.0 - static_cast<double>(i+1));

      sparseVec::syr(arg->x_.coeff(i).Jac_,shp,tensor);
      sparseVec::syr(mean->x_.coeff(i).Jac_,shp,tensor);
      sparseVec::syr2(arg->x_.coeff(i).Jac_,
                      mean->x_.coeff(i).Jac_,-shp,tensor);
      sparseVec::syr2(arg->x_.coeff(i).Jac_,
                      nu.Jac_,t3,tensor);
      sparseVec::syr2(mean->x_.coeff(i).Jac_,
                      nu.Jac_,-t3,tensor);
    }
  }

  //std::cout << "lpdf_marg : " << lpdf << std::endl;

  // Guassian conditionals
  packedSym<stan::math::var> Winv;
  double ddiag;
  if constexpr(std::is_same_v<amtVar,vType>){
    mean->packedInverse_StanVal(Winv,1);
    sparseVec::syr(nu.Jac_,mtmp,tensor);
  }
  stan::math::var yvnu = -0.5/nu_s;
  stan::math::var nuxi = stan::math::inv(nu_s);
  stan::math::var nunu = stan::math::square(nuxi);
  stan::math::var yiFac;
  std::size_t c = n;
  std::size_t cfirst;
  std::size_t clast;
  std::size_t ytOffset;

  double dfac;

  Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> dev(arg->dim()-1);
  for(int i=0;i<arg->dim()-1;i++){ //start loop at i=0

    //std::cout << "col " << i << std::endl;
    cfirst = c;
    for(int j=0;j<n-i-1;j++){
      dev.coeffRef(j) = arg->x_StanVal(c) - mean->x_StanVal(c);
      c++;
    }
    clast = c-1;
    ctmp = nu_s*(arg->Lambda_StanVal(i));

    lpdf -= 0.5*ctmp*mean->quad_form_inv_StanVal(dev,i+1);
    lpdf -= 0.5*static_cast<double>(i+1)*mean->x_StanVal(i+1)
      + 0.5*static_cast<double>(n-i-1)*(1.837877066409345-stan::math::log(ctmp));
    if constexpr(std::is_same_v<amtVar,vType>){
      // yv_loop
      for(size_t ii=i+1;ii<arg->dim();ii++){
        sparseVec::syr(mean->x_.coeff(ii).Jac_,0.5,tensor);
        sparseVec::syr2(mean->x_.coeff(ii).Jac_,nu.Jac_,yvnu,tensor);
        sparseVec::syr2(mean->x_.coeff(ii).Jac_,arg->x_.coeff(i).Jac_,-0.5,tensor);
      }

      // ytyt_loop
      ytOffset = clast+1;
      for(size_t ii = i+1; ii<n-1;ii++){
       // std::cout << " ii = " << ii << std::endl;
       // std::cout << " ytOffset " << ytOffset << std::endl;

        yiFac = mean->Lambda_StanVal(ii);
        for(size_t J=0;J<n-ii-1;J++){
          sparseVec::syr(mean->x_.coeff(ytOffset+J).Jac_,yiFac*Winv.read(ii+J,ii+J),tensor);
          for(size_t I=0;I<J;I++){
            sparseVec::syr2(mean->x_.coeff(ytOffset+J).Jac_,mean->x_.coeff(ytOffset+I).Jac_,yiFac*Winv.read(ii+I,ii+J),tensor);
          }
        }
        ytOffset += n-ii-1;

      }
      // x(i) and nu
      dfac = 0.5*static_cast<double>(n-(i+1));
      sparseVec::syr(nu.Jac_,dfac*nunu,tensor);
      sparseVec::syr2(nu.Jac_,arg->x_.coeff(i).Jac_,dfac*nuxi,tensor);
      sparseVec::syr(arg->x_.coeff(i).Jac_,dfac,tensor);

      // mu/arg loops
      ttmp = ctmp;
      for(size_t ii=cfirst;ii<=clast;ii++){
        mtmp = ttmp*Winv.read(ii-cfirst+i,ii-cfirst+i);
        sparseVec::syr(arg->x_.coeff(ii).Jac_,mtmp,tensor);
        sparseVec::syr2(arg->x_.coeff(ii).Jac_,
                        mean->x_.coeff(ii).Jac_,-mtmp,tensor);
        sparseVec::syr(mean->x_.coeff(ii).Jac_,mtmp,tensor);
        for(size_t jj=ii+1;jj<=clast;jj++){
          mtmp = ttmp*Winv.read(ii-cfirst+i,jj-cfirst+i);
          sparseVec::syr2(arg->x_.coeff(ii).Jac_,
                          arg->x_.coeff(jj).Jac_,mtmp,tensor);
          sparseVec::syr2(arg->x_.coeff(ii).Jac_,
                          mean->x_.coeff(jj).Jac_,-mtmp,tensor);
          sparseVec::syr2(arg->x_.coeff(jj).Jac_,
                          mean->x_.coeff(ii).Jac_,-mtmp,tensor);
          sparseVec::syr2(mean->x_.coeff(ii).Jac_,
                          mean->x_.coeff(jj).Jac_,mtmp,tensor);

        }
      }
    }
  }



  return(lpdf);
}

#else


template <class vType,class tenPtrType>
stan::math::var __wishartRW1_core(const SPDmatrix<vType>* arg,
                                  const SPDmatrix<vType>* mean,
                                  const vType& nu,
                                  tenPtrType tensor){
  std::size_t n = arg->dim();
  double dn = static_cast<double>(n);
  stan::math::var lpdf = 0.0;
  stan::math::var shp;
  stan::math::var scal;
  stan::math::var nu_s = asStanVar(nu);
  stan::math::var ctmp;
  stan::math::var ttmp;
  stan::math::var mtmp = -0.5*dn/nu_s;
  stan::math::var t1 = 0.5*(nu_s+dn+1.0);
  // marginals
  //


  for(int i = 0;i<n;i++){
    shp = 0.5*(nu_s-static_cast<double>(i));
    scal = 2.0*mean->Lambda_StanVal(i)/nu_s;
    lpdf += -shp*log(scal) + shp*arg->x_StanVal(i) - arg->Lambda_StanVal(i)/scal - stan::math::lgamma(shp);

    if constexpr(std::is_same_v<amtVar,vType>){

      mtmp += 0.25*stan::math::trigamma(shp);

      sparseVec::syr(arg->x_.coeff(i).Jac_,t1-static_cast<double>(i+1),tensor);
      sparseVec::syr(mean->x_.coeff(i).Jac_,0.5*nu_s,tensor);
      sparseVec::syr2(arg->x_.coeff(i).Jac_,
                      mean->x_.coeff(i).Jac_,-shp,tensor);
      for(int j=i+1;j<n;j++){
        sparseVec::syr2(arg->x_.coeff(i).Jac_,
                        mean->x_.coeff(j).Jac_,-0.5,tensor);
      }
      sparseVec::syr2(arg->x_.coeff(i).Jac_,
                      nu.Jac_,0.5*(dn+1.0-2.0*static_cast<double>(i+1))/nu_s,tensor);
    }
  }

  //std::cout << "lpdf_marg : " << lpdf << std::endl;

  // Guassian conditionals
  packedSym<stan::math::var> Winv;
  double ddiag;
  if constexpr(std::is_same_v<amtVar,vType>){
    mean->packedInverse_StanVal(Winv,1);
    sparseVec::syr(nu.Jac_,mtmp,tensor);
    //Winv.dump();
  }

  std::size_t c = n;
  std::size_t cfirst;
  std::size_t clast;
  Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> dev(arg->dim()-1);
  for(int i=0;i<arg->dim()-1;i++){
    //std::cout << "col " << i << std::endl;
    cfirst = c;
    for(int j=0;j<n-i-1;j++){
      dev.coeffRef(j) = arg->x_StanVal(c) - mean->x_StanVal(c);
      c++;
    }
    clast = c-1;
    ctmp = nu_s*(arg->Lambda_StanVal(i));

    lpdf -= 0.5*ctmp*mean->quad_form_inv_StanVal(dev,i+1);
    lpdf -= 0.5*static_cast<double>(i+1)*mean->x_StanVal(i+1)
        + 0.5*static_cast<double>(n-i-1)*(1.837877066409345-stan::math::log(ctmp));
    if constexpr(std::is_same_v<amtVar,vType>){
      ttmp = (nu_s-static_cast<double>(i))*mean->Lambda_StanVal(i);
      for(size_t ii=cfirst;ii<=clast;ii++){
        mtmp = ttmp*Winv.read(ii-cfirst+i,ii-cfirst+i);
        sparseVec::syr(arg->x_.coeff(ii).Jac_,mtmp,tensor);
        sparseVec::syr2(arg->x_.coeff(ii).Jac_,
                        mean->x_.coeff(ii).Jac_,-mtmp,tensor);
        mtmp = mean->Lambda_StanVal(i)*nu_s*Winv.read(ii-cfirst+i,ii-cfirst+i);
        sparseVec::syr(mean->x_.coeff(ii).Jac_,mtmp,tensor);
        for(size_t jj=ii+1;jj<=clast;jj++){
          mtmp = ttmp*Winv.read(ii-cfirst+i,jj-cfirst+i);
          sparseVec::syr2(arg->x_.coeff(ii).Jac_,
                          arg->x_.coeff(jj).Jac_,mtmp,tensor);
          sparseVec::syr2(arg->x_.coeff(ii).Jac_,
                          mean->x_.coeff(jj).Jac_,-mtmp,tensor);
          sparseVec::syr2(arg->x_.coeff(jj).Jac_,
                          mean->x_.coeff(ii).Jac_,-mtmp,tensor);
          mtmp = mean->Lambda_StanVal(i)*nu_s*Winv.read(ii-cfirst+i,jj-cfirst+i);
          sparseVec::syr2(mean->x_.coeff(ii).Jac_,
                          mean->x_.coeff(jj).Jac_,mtmp,tensor);

        }
      }
    }

  }
  return(lpdf);
}

#endif


template <class vType>
class wishartRW1_ld{
public:
  wishartRW1_ld(const SPDmatrix<vType>& arg,
                const SPDmatrix<vType>& mean,
                const vType& nu){}
};

template<>
class wishartRW1_ld<amtVar>{
  const SPDmatrix<amtVar>* arg_;
  const SPDmatrix<amtVar>* mean_;
  const amtVar nu_;
public:
  wishartRW1_ld(const SPDmatrix<amtVar>& arg,
                const SPDmatrix<amtVar>& mean,
                const amtVar& nu) : arg_(&arg),mean_(&mean),nu_(nu) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    return(__wishartRW1_core(arg_,mean_,nu_,tensor));
  }
};

template<>
class wishartRW1_ld<stan::math::var>{
  const SPDmatrix<stan::math::var>* arg_;
  const SPDmatrix<stan::math::var>* mean_;
  const stan::math::var nu_;
public:
  wishartRW1_ld(const SPDmatrix<stan::math::var>& arg,
                const SPDmatrix<stan::math::var>& mean,
                const stan::math::var& nu) : arg_(&arg),mean_(&mean),nu_(nu) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    return(__wishartRW1_core(arg_,mean_,nu_,tensor));
  }
};

}


#endif
