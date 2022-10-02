#ifndef _TRANSFORMEDPARAMETERS_HPP_
#define _TRANSFORMEDPARAMETERS_HPP_

template <class numType>
class identityTransformation{
  numType par_;
  int index_;
public:
  identityTransformation(const numType tpar,
                         const int index) : par_(tpar), index_(index) {}
  inline numType par() const {return par_;}
  inline numType tpar() const {return par_;}
  inline int index() const {return index_;}
};



template <class numType>
class logTransformation{
  numType tpar_;
  numType par_;
  int index_;
public:
  logTransformation(const numType tpar, 
                    const int index) : tpar_(tpar), par_(exp(tpar)), index_(index) {}
  logTransformation(const numType tpar) : tpar_(tpar), par_(exp(tpar)), index_(-1) {}
  inline numType par() const {return par_;}
  inline numType tpar() const {return tpar_;}
  inline numType jac() const {return par_;}
  inline numType logJac() const {return tpar_;}
  inline int index() const {return index_;}
};

template <class numType> 
class logitTransformation{
  double a_;
  double b_;
  numType tpar_;
  numType par_;
  int index_;
public:
  logitTransformation(const double lower,
                      const double upper,
                      const numType tpar,
                      const int index) : a_(lower), b_(upper), tpar_(tpar), par_(lower + (upper-lower)*fast_spec_funs::logit_inverse(tpar)), index_(index) {}
  logitTransformation(const double lower,
                      const double upper,
                      const numType tpar) : a_(lower), b_(upper), tpar_(tpar), par_(lower + (upper-lower)*fast_spec_funs::logit_inverse(tpar)), index_(-1) {}
  inline numType par() const {return par_;}
  inline numType tpar() const {return tpar_;}
  inline numType jac() const {return par_-a_-pow(par_-a_,2)/(b_-a_);}
  inline numType logJac() const {return(tpar_-2.0*log(1.0+exp(tpar_))+log(b_-a_));}
  inline int index() const {return index_;}
};



#endif