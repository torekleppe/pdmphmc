#ifndef _AMTLPDFRETURNTYPES_HPP_
#define _AMTLPDFRETURNTYPES_HPP_

namespace amt{
class amtTripletReturn{
public:
  amtNumType lp_;

  std::vector<size_t> i_;
  std::vector<size_t> j_;
  std::vector<amtNumType> tenVals_;


/*
  template <class T>
  void FIscalarPar(const amtVar& var, const T& FI){
    size_t pd = var.Jac_.nz();
    for(size_t i=0;i<pd;i++){
      for(size_t j=i;j<pd;j++){
        i_.push_back(var.Jac_.inds_[i]);
        j_.push_back(var.Jac_.inds_[j]);
        tenVals_.push_back(var.Jac_.vals_[i]*var.Jac_.vals_[j]*FI);
      }
    }
  }

  // contribution  to tensor from diagonal fisher info matrix
  template <class T1,class T2>
  void FItwoParDiag(const amtVar& var1,
                    const amtVar& var2,
                    const T1& FI1,
                    const T2& FI2){

    std::vector<size_t> uni;
    findUnion(var1.Jac_.inds_,var2.Jac_.inds_,uni);

    if(uni.size()==var1.Jac_.inds_.size()+var2.Jac_.inds_.size()){
      // the parameters don't depend on an overlapping set of basic variables!
      FIscalarPar(var1,FI1);
      FIscalarPar(var2,FI2);
    } else {



    }
  }
 */
};


}

#endif
