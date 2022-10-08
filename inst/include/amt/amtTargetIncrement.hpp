#ifndef _AMTTARGETINCREMENT_HPP_
#define _AMTTARGETINCREMENT_HPP_

namespace amt{

template <class argType>
class targetIncrement{
public:
  targetIncrement(const argType& increment){}
};

template <>
class targetIncrement<amtVar>{
  inline static bool print_ = true;
  stan::math::var inc_;
public:
  targetIncrement(const amtVar& increment) : inc_(increment.val_) {
    if(print_){
      std::cout<< " WARNING: Model includes instance of targetIncrement(), will not be properly accounted for in the metric tensor!!!" << std::endl;
      print_=false;
    }
  }
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{ return inc_;}
};

template <>
class targetIncrement<stan::math::var>{
  stan::math::var inc_;
public:
  targetIncrement(const stan::math::var& increment) : inc_(increment) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{ return inc_;}
};

template <>
class targetIncrement<double>{
  stan::math::var inc_;
public:
  targetIncrement(const double increment) : inc_(increment) {}
  template <class tenPtrType>
  inline stan::math::var operator()(tenPtrType tensor) const{ return inc_;}
};

}

#endif
