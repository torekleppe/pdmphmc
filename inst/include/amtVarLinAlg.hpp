#ifndef _AMTVARLINALG_HPP_
#define _AMTVARLINALG_HPP_
#include <iostream>
#include <queue>
#include <vector>

/*
 * (double)Matrix-(amtVar)Vector product
 *
 */

namespace sparseVec{

struct __ustr{
  std::size_t val_;
  std::size_t arr_;
  std::size_t ind_;
  /*  void operator= ( const __ustr& rhs){
   val_ = rhs.val_;
   arr_ = rhs.arr_;
   ind_ = rhs.ind_;
  };*/
  inline bool operator>(const __ustr& rhs) const {return val_>rhs.val_;}
  __ustr(const std::size_t val,
         const std::size_t arr,
         const std::size_t ind) : val_(val), arr_(arr), ind_(ind) {}
  __ustr(){}
};
template<typename T>
void print_queue(T q) { // NB: pass by value so the print uses a copy
  while(!q.empty()) {
    std::cout << q.top().val_ << " " << q.top().arr_ << " " << q.top().ind_ << std::endl;
    q.pop();
  }
}


void find_union(const Eigen::Matrix<amt::amtVar,Eigen::Dynamic,1>& y,
                std::vector<std::size_t>& uni,
                Eigen::Matrix<size_t,Eigen::Dynamic,1>& mapCount,
                Eigen::Matrix<size_t,Eigen::Dynamic,1>& mapVals){
  //std::cout << "find union" << std::endl;

  std::priority_queue<__ustr,std::vector<__ustr>,std::greater<__ustr> > pq;

  mapCount.resize(y.size()+1);
  mapCount.coeffRef(0) = 0;

  for(std::size_t i = 0; i<y.size();i++){
    if(! y.coeff(i).Jac_.empty()) pq.push(__ustr(y.coeff(i).Jac_.ind(0),i,0));
    mapCount.coeffRef(i+1) = mapCount.coeff(i) + y.coeff(i).Jac_.nz();
  }
  mapVals.resize(mapCount.coeff(y.size()));

  __ustr curr;
  uni.empty();


  size_t nextInd;

  while(pq.empty()==false){
    curr = pq.top();
    pq.pop();
    //std::cout << "current: " << curr.val_ << " " << curr.arr_ << " " << curr.ind_ << std::endl;

    // add to union vector if appropriate
    if(uni.size()==0 || uni.back()<curr.val_){
      uni.push_back(curr.val_);
    }

    mapVals.coeffRef(mapCount.coeff(curr.arr_)+curr.ind_) = uni.size()-1;

    // put next element in heap
    nextInd = curr.ind_+1;
    if(nextInd<y.coeff(curr.arr_).Jac_.nz()){
      pq.push(__ustr(y.coeff(curr.arr_).Jac_.ind(nextInd),curr.arr_,nextInd));
    }
  }
/*
  std::cout << "union" << std::endl;
  for(size_t i=0;i<uni.size();i++) std::cout << uni[i] << std::endl;
  std::cout << "map" << std::endl;
  for(size_t i=0;i<y.size();i++){
    std::cout << "y - element # " << i << std::endl;
    for(size_t j=mapCount[i];j<mapCount[i+1];j++) std::cout << mapVals[j] << " ";
    std::cout << std::endl;
  }
*/
}

}

namespace amt{

inline void matVecProd(const Eigen::MatrixXd& x,
                       const Eigen::Matrix<amtVar,Eigen::Dynamic,1>& y,
                       Eigen::Matrix<amtVar,Eigen::Dynamic,1>& ret){
  if(ret.size()!=x.rows()) ret.resize(x.rows());
  std::vector<std::size_t> uni;
  Eigen::Matrix<size_t,Eigen::Dynamic,1> mapCount,mapVals;
  sparseVec::find_union(y,uni,mapCount,mapVals);

  Eigen::Matrix<stan::math::var,Eigen::Dynamic,1> retVal = x*asStanVar(y);


  size_t ll;
  for(size_t i = 0; i<x.rows(); i++){
    ret.coeffRef(i).val_ = retVal.coeff(i);
    ret.coeffRef(i).Jac_.__alloc(uni);
    for(size_t j=0;j<y.size();j++){
      ll = 0;
      for(size_t ii=mapCount.coeff(j);ii<mapCount(j+1);ii++){
        ret.coeffRef(i).Jac_.__incrVals(mapVals.coeff(ii),x.coeff(i,j)*y.coeff(j).Jac_.__getVal(ll));
        ll++;
      }
    }




  }



}
inline void matVecProd(const Eigen::MatrixXd& x,
                       const Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& y,
                       Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>& ret){

  ret=x*y;
}

}

#endif

