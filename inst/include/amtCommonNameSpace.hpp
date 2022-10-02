#ifndef _AMTCOMMONNAMESPACE_HPP_
#define _AMTCOMMONNAMESPACE_HPP_

/*
 * Namespace for aiding compilation of heavily templated functions
 * Mainly in response to trouble with
 * stan::math::apply_scalar_unary<X,amt::amtVar>
 */


namespace cmn{
using amt::square;
using stan::math::square;

using amt::sqrt;
inline stan::math::var sqrt(const stan::math::var& x){return(stan::math::sqrt(x));}
using std::sqrt;

using amt::exp;
inline stan::math::var exp(const stan::math::var& x){return(stan::math::exp(x));}
using std::exp;

using amt::log;
inline stan::math::var log(const stan::math::var& x){return(stan::math::log(x));}
using std::log;

using amt::inv_logit;
using stan::math::inv_logit;

using amt::pow;
inline stan::math::var pow(const stan::math::var& x, const stan::math::var& y){return(stan::math::pow(x,y));}
inline stan::math::var pow(const stan::math::var& x, const double y){return(stan::math::pow(x,y));}
inline stan::math::var pow(const stan::math::var& x, const int y){return(stan::math::pow(x,static_cast<double>(y)));}
inline stan::math::var pow(const double x, const stan::math::var& y){return(stan::math::pow(x,y));}
using std::pow;

}


#endif
