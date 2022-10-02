using namespace amt;
struct normalMeanPrec{

  DATA_DOUBLE(y); // see point 1 below

  void preProcess(){} // see point 2 below


    // the following function defines the actual probability
    // distribution to sample from, see point 3 below
    template < class varType, class tensorType, bool storeNames>
    void operator()(amt::amtModel<varType,tensorType,storeNames> &model__){

      // the parameters and latent variables to be sampled
      // see point 4 below
      PARAMETER_SCALAR(logtau,1.0); // log-precision
      PARAMETER_SCALAR(mu); // mean

      // prior on logtau so that exp(logtau) has an
      // exponential distribution with expectation 1.0
      model__+=expGamma_ld(logtau,1.0,1.0);

      // standard normal prior on mu
      model__+=normal_ld(mu,0.0,1.0);

      // likelihood
      varType obsSD = exp(-0.5*logtau); // see point 5 below
      model__+=normal_ld(y,mu,obsSD); // (note sd and not variance)

      // generated quantities, see point 6 below
      model__.generated(exp(asDouble(logtau)),"tau");
      model__.generated(asDouble(mu),"mu_gen");

    } // end of operator()
  }; // end of struct
