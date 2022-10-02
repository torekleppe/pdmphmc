#ifndef _RNG_HPP_
#define _RNG_HPP_


/*
 
 wrapper class for standard library random number generation

*/

#ifdef __STORE_RNGS__
static std::vector<double> __rng_store_n;
static std::vector<double> __rng_store_u;
#endif

class rng{
  std::mt19937_64 gen_;
  std::normal_distribution<double> stdnormal_;
  std::uniform_real_distribution<double> unif_;
  
public:
  rng(const int seed){
    gen_.seed(seed);
    std::normal_distribution<>::param_type nparam(0.0,1.0);
    stdnormal_.param(nparam);
    std::uniform_real_distribution<>::param_type uparam(0.0,1.0);
    unif_.param(uparam);
  };
  
  rng(){rng(1);};
  
  void seed(const int seed){gen_.seed(seed);};
  inline double rnorm(){
#ifndef __STORE_RNGS__
    return stdnormal_(gen_);
#else
    double r = stdnormal_(gen_);
    __rng_store_n.push_back(r);
    return(r);
#endif
    };
  double runif(){
#ifndef __STORE_RNGS__
    return unif_(gen_);
#else
    double r = unif_(gen_);
    __rng_store_u.push_back(r);
    return(r);
#endif
    };
  void rnorm(Eigen::Matrix<double,Eigen::Dynamic,1> &v){
    for(int i=0;i<v.size();i++) v.coeffRef(i) = rnorm();
  };
  void rnorm(Eigen::Ref<Eigen::VectorXd> v){
    for(int i=0;i<v.size();i++) v.coeffRef(i) = rnorm();
  };
  Eigen::Matrix<double,Eigen::Dynamic,1> rnorm(const int d){
    Eigen::Matrix<double,Eigen::Dynamic,1> ret(d);
    rnorm(ret);
    return(ret);
  }
  void rnorm(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> &m){
    for(int i=0;i<m.rows();i++) for(int j=0;j<m.cols();j++) m.coeffRef(i,j)= rnorm();
  };
  double rgamma(const double alpha, const double beta){
    std::gamma_distribution<double> dstr(alpha,beta);
    return dstr(gen_);
  }
/*
  void randsample(const Eigen::Matrix<double,Eigen::Dynamic,1> wts, Eigen::Matrix<int,Eigen::Dynamic,1> &sample){
    int nout = sample.size();
    int wtssize = wts.size();
    std::vector<double> unifs(nout,0);
    std::vector<int> index(nout,0);
    for(int i=0;i<nout;i++){
      unifs[i]=runif();
      index[i]=i;
    }
    std::sort(index.begin(),index.end(),[&](const int& a, const int& b){ return (unifs[a] < unifs[b]);});
    Eigen::Matrix<double,Eigen::Dynamic,1> cdf(wtssize);
    cdf.coeffRef(0) = wts.coeff(0);
    for(int i=1;i<wtssize;i++) cdf.coeffRef(i) = cdf.coeff(i-1) + wts.coeff(i);
    double wtssum = cdf.coeff(wtssize-1);
    int k=0;
    for(int i=0;i<wtssize;i++){
      while(k<nout && unifs[index[k]]<cdf[i]/wtssum){
        sample.coeffRef(index[k])=i;
        k++;
      }
    }

    
  }*/
  
  
};

#endif

