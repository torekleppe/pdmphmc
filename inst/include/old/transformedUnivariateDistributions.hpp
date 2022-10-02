#ifndef _TRANSFORMEDUNIVARIATEDISTRIBUTIONS_HPP_
#define _TRANSFORMEDUNIVARIATEDISTRIBUTIONS_HPP_


template <class numType, template <typename> class transformationType>
class transformedUniform{
  double a_;
  double b_;
  transformationType<numType> t_;
public:
  transformedUniform(const double lower, const double upper, const transformationType<numType> &t) : a_(lower), b_(upper), t_(t){}
  numType lpdf() const {
    return(t_.logJac() + log(b_-a_));}
};


template <class numType, template <typename> class transformationType>
class transformedGammaUnnormalized{
  double shape_;
  double scale_;
  transformationType<numType> t_;
public:
  transformedGammaUnnormalized(const double shape, 
                               const double scale, 
                               const transformationType<numType> &t) : shape_(shape), scale_(scale), t_(t) {}
  numType lpdf() const {
    numType par = t_.par();
    return(t_.logJac() + (shape_-1.0)*log(par) - par/scale_);
  }
  
};





#endif