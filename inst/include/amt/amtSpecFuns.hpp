#ifndef _amtSpecFuns_hpp_
#define _amtSpecFuns_hpp_

#include <cmath>
#include <stan/math.hpp>
#include "amt.hpp"

namespace amt{

/*
inline double Exp(const double x){return(std::exp(x));}
inline stan::math::var Exp(const stan::math::var& x){return(stan::math::exp(x));}
inline amtVar Exp(const amtVar& x){return(amt::exp(x));}

inline double Log(const double x){return(std::log(x));}
inline stan::math::var Log(const stan::math::var& x){return(stan::math::log(x));}
inline amtVar Log(const amtVar& x){return(amt::log(x));}


inline double Sqrt(const double x){return(std::sqrt(x));}
inline stan::math::var Sqrt(const stan::math::var& x){return(stan::math::sqrt(x));}
inline amtVar Sqrt(const amtVar& x){return(amt::sqrt(x));}

inline double Square(const double x){return(std::pow(x,2));}
inline stan::math::var Square(const stan::math::var& x){return(stan::math::square(x));}
inline amtVar Square(const amtVar& x){return(amt::square(x));}
*/

/*

inline double Logit(const double x){return std::log(x/(1.0-x));}
inline stan::math::var Logit(const stan::math::var& x){return stan::math::logit(x);}
inline amtVar Logit(const amtVar& x){return(amt::logit(x));}

inline stan::math::var Inv_logit(const stan::math::var& x){return stan::math::inv_logit(x);}
inline amtVar Inv_logit(const amtVar& x){return(amt::inv_logit(x));}
*/

//inline double square(const double x){return(std::pow(x,2));}

}
#endif
