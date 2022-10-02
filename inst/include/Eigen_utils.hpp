#ifndef _EIGEN_UTILS_HPP_
#define _EIGEN_UTILS_HPP_


/*
 * Utility for doing to <- from(indexes) 
 * current version of Eigen used in RcppEigen does not seem to support
 * vector argument to operator()
 */

template <class indVecType> // array/vector of some sort
void Eigen_utils_cp_index_array_right(const Eigen::VectorXd &from,
                       const indVecType &indexes,
                       Eigen::VectorXd &to){
  if(to.size() != indexes.size()) to.resize(indexes.size());
  for(size_t i=0;i<indexes.size();i++) to.coeffRef(i) = from.coeff(indexes.coeff(i)); 
}

/*
 * Utility for doing to(indexes) <- from 
 * current version of Eigen used in RcppEigen does not seem to support
 * vector argument to operator()
 */
template <class indVecType> // array/vector of some sort
void Eigen_utils_cp_index_array_left(const Eigen::VectorXd &from,
                                      const indVecType &indexes,
                                      Eigen::VectorXd &to){
  if(indexes.size()!=to.size()){
    std::cout << "WARNING: length of indexes and to in Eigen_utils_cp_index_array_left not equal" << std::endl;
  }
  for(size_t i=0;i<indexes.size();i++) to.coeffRef(indexes.coeff(i)) = from.coeff(i);
}
#endif