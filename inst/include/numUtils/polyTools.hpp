#ifndef _POLYTOOLS_HPP_
#define _POLYTOOLS_HPP_

#ifndef _POLY_TOOLS_MAX_COEFS_
#define _POLY_TOOLS_MAX_COEFS_ 9 // allows representation of polynomial of order 8
#endif

namespace numUtils{

template <class T> // template to avoid cyclic dependency problem, really only intended for T=Poly below
class sturmChain{
  std::vector<T> series_;
  T q_,r_;
public:
  int numSignChanges(const double x){
    int c = 0;

    if(series_.size()>0){
      double firstVal = series_[0](x);
      bool oldPos = firstVal>0.0;
      size_t offset = 0;

      if(std::fabs(firstVal)<1.0e-15){
        offset = 1;
        oldPos = series_[1](x)>0.0;
      }
      bool newPos;
      double newVal;
      for(size_t i = 1+offset;i<series_.size();i++){
        newVal = series_[i](x);
        if(std::fabs(newVal)>1.0e-15){
          newPos = newVal>0.0;
          if(! (newPos==oldPos)) c++;
          oldPos = newPos;
        }
      }
    }
    //std::cout << "number of sign changes at x = " << x << " are " << c << std::endl;
    return(c);
  }

  sturmChain(const T& p0){
    if(p0.order()>0){
      series_.push_back(p0);
      series_.push_back(p0.derivative());
      while(! series_.back().isConstant()){
        series_.end()[-2].polyDiv(series_.back(),q_,r_);
        r_.negative();
        series_.push_back(r_);

      }
    }
  }
  inline int rootsInInterval(const double a,
                             const double b){
    int sa = numSignChanges(a);
    int sb = numSignChanges(b);
    std::cout << "sign changes at a: " << sa << std::endl;
    std::cout << "sign changes at b: " << sb << std::endl;
    return (std::abs(sa-sb));
  }
  void dump() const {
    std::cout << "dump of sturmChain object" << std::endl;
    for(size_t i=0;i<series_.size();i++){
      series_[i].dump();
    }
  }

};


class Poly{

  Eigen::Matrix<double,_POLY_TOOLS_MAX_COEFS_,1> coefs_;
  std::size_t order_;

  Eigen::VectorXd VecFromRevList(std::initializer_list<double> list){
    Eigen::VectorXd ret(list.size());
    int c=list.size()-1;
    for(auto elem : list){
      ret(c) = elem;
      c--;
    }
    return(ret);
  }

public:


  Poly(const Eigen::VectorXd& c){
    for(size_t i=c.size()-1; i>=0; i--){
      if(std::fabs(c.coeff(i))>1.0e-14){
        order_ = i;
        coefs_.head(i+1) = c.head(i+1);
        break;
      }
    }
  }


  Poly(const double constant){
    order_=0; coefs_.coeffRef(0)=constant;
  }
  Poly(const double lin,
       const double constant) : Poly(Eigen::Vector2d(constant,lin)) {}
  Poly(const double quad,
       const double lin,
       const double constant) : Poly(Eigen::Vector3d(constant,lin,quad)) {}
  Poly(const double cube,
       const double quad,
       const double lin,
       const double constant) : Poly(Eigen::Vector4d(constant,lin,quad,cube)) {}

  Poly() : Poly(0.0) {}
  Poly(std::initializer_list<double> list) : Poly(VecFromRevList(list)) {}

  inline size_t order() const {return order_;}
  inline bool isConstant() const {return order_==0;}

  inline double operator()(const double x) const {
    double ret = coefs_.coeff(order_);
    for(int j=order_-1;j>=0;j--) ret = ret*x + coefs_.coeff(j);
    return(ret);
  }

  inline Poly derivative() const {
    Poly ret;

    if(order_==0){
      ret.coefs_.coeffRef(0) = 0.0;
      ret.order_ = 0;
      return(ret);
    }

    for(int i=1;i<=order_;i++){
      ret.coefs_.coeffRef(i-1) = coefs_.coeff(i)*static_cast<double>(i);
    }
    ret.order_ = order_-1;
    return(ret);
  }

  inline Poly operator+(const Poly& rhs) const {
    Poly ret;
    if(rhs.order_>order_){
      ret.order_ = rhs.order_;
      ret.coefs_.head(rhs.order_+1) = rhs.coefs_.head(rhs.order_+1);
      ret.coefs_.head(order_+1).array() += coefs_.head(order_+1).array();
    } else {
      ret.order_ = order_;
      ret.coefs_.head(order_+1) = coefs_.head(order_+1);
      ret.coefs_.head(rhs.order_+1).array() += rhs.coefs_.head(rhs.order_+1).array();
    }
    return(ret);
  }

  inline void negative(){
    coefs_.head(order_+1).array() *= -1.0;
  }
  inline void addConstant(const double c){ coefs_.coeffRef(0)+=c;}

  inline Poly operator*(const Poly& rhs) const {
    size_t newOrder = order_+rhs.order_;
    if(newOrder+1>_POLY_TOOLS_MAX_COEFS_){
      std::cout << "Poly::MultiplyBy result of too high order" << std::endl;
      throw(900);
    }
    Poly ret;
    ret.coefs_.setZero();
    for(size_t i=0;i<=order_;i++){
      for(size_t j=0;j<=rhs.order_;j++){
        ret.coefs_.coeffRef(i+j) += coefs_.coeff(i)*rhs.coefs_.coeff(j);
      }
    }
    ret.order_ = newOrder;
    return(ret);
  }

  inline void polyDiv(const Poly& denom,
                      Poly& q,
                      Poly& r) const {
    if(denom.order_==0 && std::fabs(denom.coefs_.coeff(0))<1.0e-14){
      std::cout << "Poly::polyDiv :: division by zero" << std::endl;
      throw(901);
    }
    int n = order_;
    int nv = denom.order_;
    r.coefs_ = coefs_;
    r.order_ = order_;
    q.coefs_.setZero();
    q.order_ = order_;
    for(int k=n-nv;k>=0;k--){
      q.coefs_.coeffRef(k) = r.coefs_.coeff(nv+k)/denom.coefs_.coeff(nv);
      for(int j=nv+k-1;j>=k;j--) r.coefs_.coeffRef(j) -= q.coefs_.coeff(k)*denom.coefs_.coeff(j-k);
    }
    while(std::fabs(q.coefs_.coeff(q.order_))<1.0e-14 && q.order_>0) q.order_--;
    r.order_ -= n-nv+1;
    while(std::fabs(r.coefs_.coeff(r.order_))<1.0e-14 && r.order_>0) r.order_--;

    // check (to be removed)
    //std::cout << "original:" << std::endl;
    //dump();
    //std::cout << "computed:" << std::endl;
    //(q*denom + r).dump();
  }





  inline void dump() const {
    std::cout << "Poly: ";
    for(size_t i=0;i<=order_;i++){
      std::cout << coefs_(i);
      if(i>0){
        std::cout << "*x^" << i << " ";
      } else {
        std::cout << " ";
      }
      if(i<order_){
        std::cout << "+ ";
      }
    }
    std::cout << std::endl;
  }

  // safeguarded newton solver for finding a single root in the interval (a,b]
  inline double bracketed_root(const double a, const double b){
    double lb = a;
    double ub = b;
    double lf = operator()(lb);
    double uf = operator()(ub);
    if(lf*uf>0.0){
      std::cout << "bad input in bracketed_root" << std::endl;
      throw(905);
    }
    Poly dd = derivative();
    double tb = 0.5*(lb+ub);
    double newt;
    double fdiff;
    double dev=operator()(tb);
    for(size_t i=0;i<100;i++){
      if(std::fabs(dev)<1.0e-15*std::max(std::pow(std::fabs(tb),order_),1.0)){
        //std::cout << "success, dev: " << dev << " x: " << tb << std::endl;
        return(tb);
      }
      newt = tb - dev/dd(tb);
      fdiff = lf-uf;
      if(newt<=ub && newt>lb && isfinite(newt)){
        tb = newt;
        //std::cout << "newton" << std::endl;
      } else if(std::fabs(fdiff)>1.0e-6) {
        tb = -(lb*uf - lf*ub)/fdiff;
        //std::cout << "interpolation" << std::endl;
      } else {
        tb = 0.5*(lb+ub);
        //std::cout << "bisection" << std::endl;
      }

      dev=operator()(tb);
      //std::cout << "lb: " << lb << " ub: " << ub << " newt: " << newt << " dev: " << dev << std::endl;
      //std::cout << "lf: " << lf << " uf: " << uf << std::endl;

      if(dev*uf<0.0){
        lb = tb;
        lf = dev;
      } else {
        ub = tb;
        uf = dev;
      }
    }
    std::cout << "bracketed_root failed to converge" << std::endl;
    return(0.0);
  }

  // returns the smalles root in the interval [a,b]. If no such root exist, it
  // returns an arbitrary number > b
  double smallestRootInInterval(const double a,
                                const double b){
    double fl = operator()(a);
    if(std::fabs(fl)<1.0e-15) return(a);

    double lb = a;
    double ub = b;
    double ret = std::max(b+1.0,std::fabs(b)*2.0);
    sturmChain sc(*this);
    int li = sc.numSignChanges(lb);
    int ui = sc.numSignChanges(ub);
    //std::cout << "# roots in initial interval : "  << std::abs(li-ui) << std::endl;
    // no roots in interval
    if(li-ui==0) return(ret);
    // single root in interval
    if(li-ui==1) return(bracketed_root(lb,ub));
    // otherwise work out bracket for the smallest root

    //dump();

    double tb;
    int ti;
    for(size_t i=0;i<100;i++){
      tb = 0.5*(lb+ub);
      ti = sc.numSignChanges(tb);
      if(ti==li){
        lb = tb;
      } else if(std::abs(ti-li)==1) {
        return(bracketed_root(lb,tb));
      } else {
        ub = tb;
      }
    }
    return(0.0);
  }
};




// make (order 6) polynomial \sum_i (a(i)*x^3 + b(i)*x^2 + c(i)*x + d)^2
Poly sumOfSquaredCubics(const Eigen::VectorXd& a,
                        const Eigen::VectorXd& b,
                        const Eigen::VectorXd& c,
                        const Eigen::VectorXd& d){
  Eigen::VectorXd cs(7);
  cs.coeffRef(6) = a.squaredNorm();
  cs.coeffRef(5) = 2.0*(a.dot(b));
  cs.coeffRef(4) = 2.0*(a.dot(c))+b.squaredNorm();
  cs.coeffRef(3) = 2.0*(a.dot(d)+b.dot(c));
  cs.coeffRef(2) = 2.0*(b.dot(d))+c.squaredNorm();
  cs.coeffRef(1) = 2.0*(c.dot(d));
  cs.coeffRef(0) = d.squaredNorm();
  return(Poly(cs));
}















}
#endif




