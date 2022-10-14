#ifndef _AMTTRAITS_HPP_
#define _AMTTRAITS_HPP_

namespace amt{

template <class T>
struct amtTargetType{
  typedef double type;
};

template<>
struct amtTargetType<stan::math::var>{
  typedef stan::math::var type;
};

template<>
struct amtTargetType<amtVar>{
  typedef stan::math::var type;
};




/*
 * amtReturnTypeX are used to find the appropriate return type out of double and
 * AD variables, with AD variables choosen when either of the templates are AD variables
 *
 * use as e.g.
 *
 * template <class aType,class bType>
 * inline typename amtReturnType2<aType,bType>::type functionName(const aType& a, const bType& b){...}
 *
 */


template <class Type1, class Type2>
struct amtReturnType2{
  typedef double type;
};
template<>
struct amtReturnType2<double,stan::math::var>{
  typedef stan::math::var type;
};
template<>
struct amtReturnType2<stan::math::var,double>{
  typedef stan::math::var type;
};
template<>
struct amtReturnType2<stan::math::var,stan::math::var>{
  typedef stan::math::var type;
};
template<>
struct amtReturnType2<amt::amtVar,double>{
  typedef amt::amtVar type;
};
template<>
struct amtReturnType2<double,amt::amtVar>{
  typedef amt::amtVar type;
};
template<>
struct amtReturnType2<amt::amtVar,amt::amtVar>{
  typedef amt::amtVar type;
};




template <class Type1, class Type2, class Type3>
struct amtReturnType3{
  typedef double type;
};

template<>
struct amtReturnType3<stan::math::var,double,double>{
  typedef stan::math::var type;
};
template<>
struct amtReturnType3<double,stan::math::var,double>{
  typedef stan::math::var type;
};
template<>
struct amtReturnType3<double,double,stan::math::var>{
  typedef stan::math::var type;
};
template<>
struct amtReturnType3<stan::math::var,stan::math::var,double>{
  typedef stan::math::var type;
};
template<>
struct amtReturnType3<stan::math::var,double,stan::math::var>{
  typedef stan::math::var type;
};
template<>
struct amtReturnType3<double,stan::math::var,stan::math::var>{
  typedef stan::math::var type;
};
template<>
struct amtReturnType3<stan::math::var,stan::math::var,stan::math::var>{
  typedef stan::math::var type;
};

template <class Type1, class Type2, class Type3, class Type4>
struct amtReturnType4{
  typedef double type;
};

template<>
struct amtReturnType4<stan::math::var,double,double,double>{
  typedef stan::math::var type;
};
template<>
struct amtReturnType4<double,stan::math::var,double,double>{
  typedef stan::math::var type;
};
template<>
struct amtReturnType4<double,double,stan::math::var,double>{
  typedef stan::math::var type;
};
template<>
struct amtReturnType4<double,double,double,stan::math::var>{
  typedef stan::math::var type;
};


template<>
struct amtReturnType4<stan::math::var,stan::math::var,double,double>{
  typedef stan::math::var type;
};
template<>
struct amtReturnType4<stan::math::var,double,stan::math::var,double>{
  typedef stan::math::var type;
};
template<>
struct amtReturnType4<stan::math::var,double,double,stan::math::var>{
  typedef stan::math::var type;
};
template<>
struct amtReturnType4<double,stan::math::var,stan::math::var,double>{
  typedef stan::math::var type;
};
template<>
struct amtReturnType4<double,stan::math::var,double,stan::math::var>{
  typedef stan::math::var type;
};
template<>
struct amtReturnType4<double,double,stan::math::var,stan::math::var>{
  typedef stan::math::var type;
};

template<>
struct amtReturnType4<stan::math::var,stan::math::var,stan::math::var,double>{
  typedef stan::math::var type;
};
template<>
struct amtReturnType4<stan::math::var,stan::math::var,double,stan::math::var>{
  typedef stan::math::var type;
};
template<>
struct amtReturnType4<stan::math::var,double,stan::math::var,stan::math::var>{
  typedef stan::math::var type;
};
template<>
struct amtReturnType4<double,stan::math::var,stan::math::var,stan::math::var>{
  typedef stan::math::var type;
};


template<>
struct amtReturnType4<stan::math::var,stan::math::var,stan::math::var,stan::math::var>{
  typedef stan::math::var type;
};


template <class Type1, class Type2>
struct amtNumType2{
  typedef double type;
};
template<>
struct amtNumType2<double,stan::math::var>{
  typedef stan::math::var type;
};
template<>
struct amtNumType2<stan::math::var,double>{
  typedef stan::math::var type;
};
template<>
struct amtNumType2<stan::math::var,stan::math::var>{
  typedef stan::math::var type;
};
template<>
struct amtNumType2<double,amt::amtVar>{
  typedef stan::math::var type;
};
template<>
struct amtNumType2<amt::amtVar,double>{
  typedef stan::math::var type;
};
template<>
struct amtNumType2<stan::math::var,amt::amtVar>{
  typedef stan::math::var type;
};
template<>
struct amtNumType2<amt::amtVar,stan::math::var>{
  typedef stan::math::var type;
};
template<>
struct amtNumType2<amt::amtVar,amt::amtVar>{
  typedef stan::math::var type;
};


template <class T1>
class AMT_NOT_IMPLEMENTED_ERROR__CONTACT_DEVELOPER_1;

template <class T1,class T2>
class AMT_NOT_IMPLEMENTED_ERROR__CONTACT_DEVELOPER_2;

template <class T1,class T2,class T3>
class AMT_NOT_IMPLEMENTED_ERROR__CONTACT_DEVELOPER_3;

template <class T1,class T2,class T3,class T4>
class AMT_NOT_IMPLEMENTED_ERROR__CONTACT_DEVELOPER_4;

} // namespace amt
/*
 namespace Eigen {

 template<> struct NumTraits<amt::amtVar >
 : NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
 {
 typedef amt::amtVar Real;
 typedef amt::amtVar NonInteger;
 typedef amt::amtVar Nested;

 enum {
 IsComplex = 0,
 IsInteger = 0,
 IsSigned = 1,
 RequireInitialization = 1,
 ReadCost = 1,
 AddCost = 3,
 MulCost = 3
 };
 };

 }*/
#endif
