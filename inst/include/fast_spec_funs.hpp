

#ifndef __FAST_SPEC_FUNS__
#define __FAST_SPEC_FUNS__

namespace fast_spec_funs{

template <class var> 
inline var fast_erfc(const var x){
  // erfc from numerical recipes
  var t,z,ans;
  bool posArg = doubleValue(x)>0.0;
  z = posArg ? x : -x;
  t=2./(2.+z); 
  ans=t*exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+
    t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+
    t*(-0.82215223+t*0.17087277))))))))); 
  if(!posArg) ans = 2.0-ans;
  return(ans);
}




template <class var>
inline var fast_erf(const var x){return 1.0-fast_erfc(x);}

template <class var>
inline var fast_std_pnorm(const var x){return 1.0-0.5*fast_erfc(0.7071067811865475*x);}

/*
 * logit and inverse logit
 * 
 * 
 */

template <class var>
inline var logit(const var p) {return log(p/(1.0-p));}

template <class var>
inline var logit_inverse(const var x){
  if(doubleValue(x)>0.0){
    return(1.0/(1.0+exp(-x)));
  } else {
    var tmp = exp(x);
    return(tmp/(tmp+1.0));
  }
}
  
}
#endif
