#ifndef _AMTPACKEDSYM_HPP_
#define _AMTPACKEDSYM_HPP_

namespace amt{
  template <class numType>
  class packedSym{
    std::size_t dim_;
    Eigen::Matrix<numType,Eigen::Dynamic,1> vals_;
    inline std::size_t toLin(const std::size_t i, const std::size_t j){
      size_t jj = std::max(i,j);
      return(std::min(i,j) + ((jj+1)*jj)/2);
    }
  public:
    packedSym(){}
    packedSym(const size_t dim) : dim_(dim), vals_(dim) {}
    void allocate(const size_t dim){
      if(dim != dim_){
        dim_ = dim;
        vals_.resize((dim*(dim+1))/2);
        vals_.setZero();
      }
    }
    inline numType read(const size_t i, const size_t j){
      return(vals_.coeff(toLin(i,j)));
    }
    inline void write(const size_t i,const size_t j, const numType& val){
      vals_.coeffRef(toLin(i,j)) = val;
    }

    void dump(){
      Eigen::Matrix<numType,Eigen::Dynamic,Eigen::Dynamic> tmp(dim_,dim_);
      for(size_t i=0;i<dim_;i++){
        for(size_t j=0;j<dim_;j++) tmp(i,j) = read(i,j);
      }
      std::cout << tmp << std::endl;
    }

  };

}
#endif
