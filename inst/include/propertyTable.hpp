#ifndef _PROPERTY_TABLE_HPP_
#define _PROPERTY_TABLE_HPP_
#include <vector>
#include <string>


class propertyTable{
  const std::vector<std::string> pn_ = {
    "seed", //0
    "absTol", //1
    "relTol", //2
    "massallowsFixedSubvector", //3
    "massFixedMiSubvector", //4
    "CPUtime", //5
    "lambda", // 6
    "fixedEps" // 7
  };
public:
  propertyTable(){}
  int id(const std::string prop){
    auto its = std::find(pn_.begin(),pn_.end(),prop);
    if(its != pn_.end()){
      return(its - pn_.begin());
    } else {
      return(-1);
    }
  }
};




#endif
