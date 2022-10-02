#ifndef _SPARSEVEC_HPP_
#define _SPARSEVEC_HPP_
#include <iostream>
#include <vector>
#include <Eigen/Dense>


/*
 void findUnion(const sizetVec &v1,
 const sizetVec &v2,
 sizetVec &unionVec){
 // handle trivial cases
 if(v1.size()==0 && v2.size()==0){
 unionVec.resize(0);
 return;
 }
 if(v1.size()==0){
 unionVec = v2;
 return;
 }
 if(v2.size()==0){
 unionVec = v1;
 return;
 }
 // first pass counting number of distinct elements in union

 unionVec.resize(v1.size()+v2.size());

 size_t current;
 size_t cand1;
 size_t cand2;
 if(v1.coeff(0)<v2.coeff(0)){
 current = v1.coeff(0);
 cand1 = 1;
 cand2 = 0;
 } else if(v1.coeff(0)>v2.coeff(0)) {
 current = v2.coeff(0);
 cand1 = 0;
 cand2 = 1;
 } else {
 current = v1.coeff(0);
 cand1 = 1;
 cand2 = 1;
 }

 unionVec.coeffRef(0) = current;
 size_t ucount = 1;
 size_t cv1 = v1.coeff(cand1);
 size_t cv2 = v2.coeff(cand2);

 while(cand1<v1.size() || cand2<v2.size()){

 if(cv1<cv2){
 cand1++;
 if(cv1>current){
 unionVec.coeffRef(ucount) = cv1;
 ucount++;
 current = cv1;
 }
 cv1 = (cand1<v1.size()) ? v1.coeff(cand1) : std::numeric_limits<size_t>::max();
 } else if(cv1>cv2){
 cand2++;
 if(cv2>current){
 unionVec.coeffRef(ucount) = cv2;
 ucount++;
 current = cv2;
 }
 cv2 = (cand2<v2.size()) ? v2.coeff(cand2) : std::numeric_limits<size_t>::max();
 } else { // cv1 = cv2
 cand1++;
 cand2++;
 if(cv1>current){
 unionVec.coeffRef(ucount) = cv1;
 ucount++;
 current = cv1;
 }
 cv1 = (cand1<v1.size()) ? v1.coeff(cand1) : std::numeric_limits<size_t>::max();
 cv2 = (cand2<v2.size()) ? v2.coeff(cand2) : std::numeric_limits<size_t>::max();
 }
 }
 unionVec.conservativeResize(ucount);
 }
 */

namespace sparseVec{

inline size_t findUnion(const std::vector<size_t>& v1,
                        const std::vector<size_t>& v2,
                        std::vector<size_t>& out){
  out.clear();
  size_t v1size = v1.size();
  size_t v2size = v2.size();
  // handle trivial cases first
  if(v1size==0 && v2size==0){
    return(0);
  } else if(v1size>0 && v2size==0){
    out = v1;
    return(v1size);
  } else if(v1size==0 && v2size>0){
    out = v2;
    return(v2size);
  }
  // the non-trivial case:
  size_t c1 = 0;
  size_t c2 = 0;

  while(c1<v1size && c2<v2size){
    if(v1[c1]<v2[c2]){
      out.push_back(v1[c1]);
      c1++;
    } else if(v1[c1]>v2[c2]){
      out.push_back(v2[c2]);
      c2++;
    } else {
      out.push_back(v1[c1]);
      c1++;
      c2++;
    }
  }
  if(c1==v1size){
    for(size_t i=c2;i<v2size;i++) out.push_back(v2[i]);
  } else {
    for(size_t i=c1;i<v1size;i++) out.push_back(v1[i]);
  }
  return(out.size());
}

inline size_t findIntersection(const std::vector<size_t>& v1,
                               const std::vector<size_t>& v2,
                               std::vector<size_t>& out1,
                               std::vector<size_t>& out2){

  size_t v1size = v1.size();
  size_t v2size = v2.size();

  out1.clear();
  out2.clear();

  if(v1size==0 || v2size==0) return(0);


  if(v1size<=v2size){
    size_t j = 0;
    for(size_t i=0;i<v1size;i++){
      while(v2[j]<v1[i] && j+1<v2size) j++;
      if(v2[j]==v1[i]){
        out1.push_back(i);
        out2.push_back(j);
      }
      if( j+1 == v2size && v1[i]>v2[j]) break;
    }
  } else {
    size_t j = 0;
    for(size_t i=0;i<v2size;i++){
      while(v1[j]<v2[i] && j+1<v1size) j++;
      if(v1[j]==v2[i]){
        out1.push_back(j);
        out2.push_back(i);
      }
      if( j+1 == v1size && v2[i]>v1[j]) break;
    }
  }

  std::cout << out1.size() <<  " common values :" << std::endl;
  for(size_t i=0; i<out1.size();i++){
    std::cout << v1[out1[i]] << "   " << v2[out2[i]] << std::endl;
  }



  return(out1.size());

}

template <class numType>
class sparseVec{
  // keep these private
  inline double __dblVal(const double arg) const {return arg;}
  inline double __dblVal(const stan::math::var& arg) const {return arg.val();}
  std::vector<size_t> inds_;
  std::vector<numType> vals_;
  //bool basic_;
public:
  inline size_t nz() const {return inds_.size();}
  inline size_t firstInd() const {return inds_[0];}
  inline size_t ind(const size_t i) const {return inds_[i];}
  void dump() const {
    std::cout << "dump of sparseVec:\n";
    for(size_t i = 0;i<inds_.size();i++) std::cout << "vector[" << inds_[i] << "] = " << vals_[i] << std::endl;
  }
  // unit vector constructor
  sparseVec(const size_t dim) : inds_(1,dim), vals_(1,1.0) {}
  sparseVec(const size_t dim, const numType val) : inds_(1,dim), vals_(1,val) {}
  // zeroVector
  sparseVec() {}
  sparseVec(const sparseVec<numType> &arg){
    inds_ = arg.inds_;
    vals_ = arg.vals_;
  }
  inline bool empty() const {return inds_.empty();}

  inline void clear(){
    inds_.clear();
    vals_.clear();
  }

  inline void __alloc(const size_t nz){
    inds_.resize(nz);
    vals_.resize(nz);
  }

  inline void __alloc(const std::vector<size_t>& inds){
    inds_ = inds;
    vals_.resize(inds.size(),0.0);
  }

  inline void __incrVals(const size_t index,const numType& incr){vals_[index]+=incr; }
  inline numType __getVal(const size_t index) const {return vals_[index];}

  inline void setUnit(const size_t dim){
    inds_.clear(); inds_.push_back(dim);
    vals_.clear(); vals_.push_back(1.0);
  }
  inline bool isUnit() const {
    return(inds_.size()==1 && __dblVal(vals_[0]-1.0)<1.0e-14);
  }
  //y <- a*x+y (where *this=y)
  template <class aType>
  void axpy(const aType a, const sparseVec<numType> &x){
    // data structure of output
    std::vector<size_t> uni;
    std::vector<numType> resVals(findUnion(inds_,x.inds_,uni),0.0);
    //Eigen::Matrix<numType,Eigen::Dynamic,1> resVals(uni.size());
    //resVals.setZero();

    // spread *this onto new data structure
    size_t ui=0;
    for(size_t yi=0;yi<inds_.size();yi++){
      while(uni[ui]<inds_[yi]){ui++;}
      resVals[ui] = vals_[yi];
    }

    ui=0;
    for(size_t xi=0;xi<x.inds_.size();xi++){
      while(uni[ui]<x.inds_[xi]){ui++;}
      resVals[ui] += a*x.vals_[xi];
    }
    inds_ = uni;
    vals_ = resVals;
  }

  inline sparseVec<numType> operator-(){
    sparseVec<numType> ret(*this);
    for(size_t i=0;i<vals_.size();i++ ) ret.vals_[i] = -ret.vals_[i];
    return(ret);
  }

  inline void operator+=(const sparseVec<numType> &rhs){axpy(1.0,rhs);}
  inline void operator-=(const sparseVec<numType> &rhs){axpy(-1.0,rhs);}
  inline void scal(const numType a){
    for(size_t i=0;i<vals_.size();i++) vals_[i]*=a;
  }


  inline sparseVec<numType> operator*(const numType a) const {
    sparseVec<numType> ret(*this);
    //for(size_t i=0;i<ret.vals_.size();i++) ret.vals_[i]*=a;
    ret.scal(a);
    return(ret);
  }

  template <class numType_, class scaleType, class tensorPtrType >
  friend inline void syr(const sparseVec<numType_>& v,
                         const scaleType& alpha,
                         tensorPtrType tensorPtr);

  template <class numType_, class scaleType, class outType >
  friend inline void syrDense(const sparseVec<numType_>& v,
                              const scaleType& alpha,
                              Eigen::Matrix<outType,Eigen::Dynamic,Eigen::Dynamic>& A);


  template <class numType1, class numType2, class scaleType, class tensorPtrType>
  friend inline void syr2(const sparseVec<numType1>& v1,
                          const sparseVec<numType2>& v2,
                          const scaleType& alpha,
                          tensorPtrType tensorPtr);

  template <class numType1, class numType2, class scaleType, class outType>
  friend inline void syr2Dense(const sparseVec<numType1>& v1,
                               const sparseVec<numType2>& v2,
                               const scaleType& alpha,
                               Eigen::Matrix<outType,Eigen::Dynamic,Eigen::Dynamic>& A);
  template<class nT>
  friend std::ostream& operator<< (std::ostream& out, const sparseVec<nT>& obj);

}; // class done
template <class nT>
std::ostream& operator<< (std::ostream& out, const sparseVec<nT>& obj){
  for(size_t i=0;i<obj.inds_.size();i++){
    out << " (" << obj.inds_[i] << "):" << obj.vals_[i];
  }
  return(out);
}


template <class numType>
inline sparseVec<numType> operator*(const numType left, const sparseVec<numType> &right){return right*left;}

template <class numType>
inline sparseVec<numType> operator+(const sparseVec<numType> &left,
                                    const sparseVec<numType> &right){
  sparseVec<numType> ret(left);
  ret+=right;
  return ret;
}
template <class numType>
inline sparseVec<numType> operator-(const sparseVec<numType> &left,
                                    const sparseVec<numType> &right){
  sparseVec<numType> ret(left);
  ret-=right;
  return(ret);
}



template <class numType, class scaleType, class tensorPtrType >
inline void syr(const sparseVec<numType>& v,
                const scaleType& alpha,
                tensorPtrType tensorPtr){

  for(size_t j =0; j<v.inds_.size();j++){
    for(size_t i=0; i<=j;i++){
      tensorPtr->pushScalar(v.inds_[i],
                            v.inds_[j],
                                   alpha*v.vals_[i]*v.vals_[j]);
    }
  }
}

template <class numType, class scaleType, class outType >
inline void syrDense(const sparseVec<numType>& v,
                     const scaleType& alpha,
                     Eigen::Matrix<outType,Eigen::Dynamic,Eigen::Dynamic>& A){
  for(size_t j =0; j<v.inds_.size();j++){
    for(size_t i=0; i<=j;i++){
      A(v.inds_[i],v.inds_[j]) += alpha*v.vals_[i]*v.vals_[j];
    }
  }
}

template <class numType1, class numType2, class scaleType, class tensorPtrType>
inline void syr2(const sparseVec<numType1>& v1,
                 const sparseVec<numType2>& v2,
                 const scaleType& alpha,
                 tensorPtrType tensorPtr){
  for(size_t j = 0; j<v1.inds_.size(); j++){
    for(size_t i =0; i<v2.inds_.size(); i++){
      if(v1.inds_[j]<v2.inds_[i]){
        tensorPtr->pushScalar(v1.inds_[j],v2.inds_[i],alpha*v1.vals_[j]*v2.vals_[i]);
      } else if(v1.inds_[j]>v2.inds_[i]){
        tensorPtr->pushScalar(v2.inds_[i],v1.inds_[j],alpha*v1.vals_[j]*v2.vals_[i]);
      } else {
        tensorPtr->pushScalar(v1.inds_[j],v1.inds_[j],2.0*alpha*v1.vals_[j]*v2.vals_[i]);
      }
    }
  }
}

template <class numType1, class numType2, class scaleType, class outType>
inline void syr2Dense(const sparseVec<numType1>& v1,
                      const sparseVec<numType2>& v2,
                      const scaleType& alpha,
                      Eigen::Matrix<outType,Eigen::Dynamic,Eigen::Dynamic>& A){
  for(size_t j = 0; j<v1.inds_.size(); j++){
    for(size_t i =0; i<v2.inds_.size(); i++){
      if(v1.inds_[j]<v2.inds_[i]){
        A(v1.inds_[j],v2.inds_[i]) += alpha*v1.vals_[j]*v2.vals_[i];
      } else if(v1.inds_[j]>v2.inds_[i]){
        A(v2.inds_[i],v1.inds_[j]) += alpha*v1.vals_[j]*v2.vals_[i];
      } else {
        A(v1.inds_[j],v1.inds_[j]) += 2.0*alpha*v1.vals_[j]*v2.vals_[i];
      }
    }
  }
}




/*
 * Computes J^T*A*J where the rows of J are a pair of sparse vectors
 * v1 and v2, and A is a symmetric 2x2 matrix:
 * [a11,a12]
 * [a12,a22]
 *
 */

template <class numType1,
          class numType2,
          class Atype,
          class tensorPtrType>
inline void sym2x2outer(const sparseVec<numType1>& v1,
                             const sparseVec<numType2>& v2,
                             const Atype& a11,
                             const Atype& a12,
                             const Atype& a22,
                             tensorPtrType tensorPtr){
  syr(v1,a11,tensorPtr);
  syr2(v1,v2,a12,tensorPtr);
  syr(v2,a22,tensorPtr);
}


template <class numType1,
          class numType2,
          class Atype,
          class outType>
inline void sym2x2outerDense(const sparseVec<numType1>& v1,
                             const sparseVec<numType2>& v2,
                             const Atype& a11,
                             const Atype& a12,
                             const Atype& a22,
                             Eigen::Matrix<outType, Eigen::Dynamic,Eigen::Dynamic>& out){
  syrDense(v1,a11,out);
  syr2Dense(v1,v2,a12,out);
  syrDense(v2,a22,out);
}




} // namespace



#endif
