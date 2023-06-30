using namespace amt;
struct model{

  DATA_INT(dummyDataNotUsed);

  void preProcess(){}

    template < class varType, class tensorType, bool storeNames>
    void operator()(amt::amtModel<varType,tensorType,storeNames> &model__){

      PARAMETER_VECTOR(x,2);
      model__+=normal_ld(x,0.0,1.0);
      model__.generated(asDouble(x),"x_gen");

      model__.sparseLinConstraint(x(0)+x(1)+100.0);

    } // end of operator()
  }; // end of struct
