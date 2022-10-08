#ifndef _AMTEXPGAMMALPDF_
#define _AMTEXPGAMMALPDF_



namespace amt{

/*
 * Distribution for X so that exp(X) \sim Gamma(shape,scale)
 *
 *
 */

template <class argType, class shapeType, class scaleType>
inline typename amtReturnType3<argType,shapeType,scaleType>::type expGamma_lpdf(const argType& arg,
                                                                          const shapeType& shape,
                                                                          const scaleType& scale){
  return(shape*arg - cmn::exp(arg)/scale - shape*cmn::log(scale) - stan::math::lgamma(shape));
}

/*
inline stan::math::var expGamma_lpdf(const stan::math::var& arg,
                                     const stan::math::var& shape,
                                     const double scale){
  return(shape*arg - stan::math::exp(arg)/scale - shape*std::log(scale) - stan::math::lgamma(shape));
}
inline stan::math::var expGamma_lpdf(const stan::math::var& arg,
                                     const stan::math::var& shape,
                                     const stan::math::var& scale){
  return(shape*arg - stan::math::exp(arg)/scale - shape*stan::math::log(scale) - stan::math::lgamma(shape));
}
*/



template <class argType,class shapeType,class scaleType>
class expGamma_ld{

public:
  expGamma_ld(const argType& arg,
              const shapeType& shape,
              const scaleType& scale) {}
};

template <>
class expGamma_ld<amtVar,double,double>{
  const amtVar* arg_;
  const double shape_;
  const double scale_;
public:
  expGamma_ld(const amtVar& arg,
              const double shape,
              const double scale) : arg_(&arg), shape_(shape), scale_(scale) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    sparseVec::syr(arg_->Jac_,
                   //1.0/stan::math::trigamma(shape_), // to be consistent with notes
                   shape_,
                   tensor);
    return expGamma_lpdf(arg_->value(),shape_,scale_);
  }
};

template <>
class expGamma_ld<amtVar,amtVar,double>{
  const amtVar* arg_;
  const amtVar* shape_;
  const double scale_;
public:
  expGamma_ld(const amtVar& arg,
              const amtVar& shape,
              const double scale) : arg_(&arg), shape_(&shape), scale_(scale) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    stan::math::var tmp = stan::math::trigamma(shape_->value());
    sparseVec::syr(arg_->Jac_,
                   shape_->value(),
                   tensor);
    sparseVec::syr(shape_->Jac_,
                   tmp,
                   tensor);
    sparseVec::syr2(arg_->Jac_,
                    shape_->Jac_,
                    -1.0,
                    tensor);
    return expGamma_lpdf(arg_->value(),shape_->value(),scale_);
  }
};

template <>
class expGamma_ld<amtVar,amtVar,amtVar>{
  const amtVar* arg_;
  const amtVar* shape_;
  const amtVar* scale_;
public:
  expGamma_ld(const amtVar& arg,
              const amtVar& shape,
              const amtVar& scale) : arg_(&arg), shape_(&shape), scale_(&scale) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{
    stan::math::var tmp = stan::math::trigamma(shape_->value());
    stan::math::var tmp2 = shape_->value()/(scale_->value());
    sparseVec::syr(arg_->Jac_,
                   shape_->value(),
                   tensor);
    sparseVec::syr(shape_->Jac_,
                   tmp,
                   tensor);
    sparseVec::syr(scale_->Jac_,
                   tmp2,
                   tensor);
    sparseVec::syr2(arg_->Jac_,
                    shape_->Jac_,
                    -1.0,
                    tensor);
    sparseVec::syr2(arg_->Jac_,
                    scale_->Jac_,
                    -tmp2,
                    tensor);
    sparseVec::syr2(shape_->Jac_,
                    scale_->Jac_,
                    1.0/(scale_->value()),
                    tensor);
    return expGamma_lpdf(arg_->value(),shape_->value(),scale_->value());
  }
};



template <>
class expGamma_ld<stan::math::var,double,double>{
  const stan::math::var lpdf_;
public:
  expGamma_ld(const stan::math::var& arg,
              const double shape,
              const double scale) : lpdf_(expGamma_lpdf(arg,shape,scale)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{return lpdf_;}
};

template <>
class expGamma_ld<stan::math::var,stan::math::var,double>{
  const stan::math::var lpdf_;
public:
  expGamma_ld(const stan::math::var& arg,
              const stan::math::var& shape,
              const double scale) : lpdf_(expGamma_lpdf(arg,shape,scale)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{return lpdf_;}
};
template <>
class expGamma_ld<stan::math::var,stan::math::var,stan::math::var>{
  const stan::math::var lpdf_;
public:
  expGamma_ld(const stan::math::var& arg,
              const stan::math::var& shape,
              const stan::math::var& scale) : lpdf_(expGamma_lpdf(arg,shape,scale)) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{return lpdf_;}
};




} // namespace

#endif
