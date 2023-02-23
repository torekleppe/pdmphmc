#ifndef _AMTDATA_HPP_
#define _AMTDATA_HPP_
#include <iostream>
#include <fstream>
#include <streambuf>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include "rapidjson/document.h"

class amtData{
  // JSON file stuff
  rapidjson::Document doc_;
  bool isOpen_;

  bool empty_;

  // int scalars
  std::vector<int*> si_;
  std::vector<std::string> sin_;

  // double scalars
  std::vector<double*> sd_;
  std::vector<std::string> sdn_;

  // double vectors
  std::vector<Eigen::VectorXi*> vi_;
  std::vector<std::string> vin_;

  // double vectors
  std::vector<Eigen::VectorXd*> vd_;
  std::vector<std::string> vdn_;

  // double matrix
  std::vector<Eigen::MatrixXi*> mi_;
  std::vector<std::string> min_;


  // double matrix
  std::vector<Eigen::MatrixXd*> md_;
  std::vector<std::string> mdn_;


  inline int numFields(){
    return(si_.size()+sd_.size()+vi_.size()+vd_.size()+mi_.size() + md_.size());
  }

  template <class retType>
  int getScalar(const std::string& fieldName,
                retType* ret){
    if(!doc_.HasMember(fieldName.c_str())){
      std::cout << "no field " << fieldName << " in data file" << std::endl;
      return(4);
    } else {
      rapidjson::Value& field = doc_[fieldName.c_str()];
      if(field.IsNumber()){
        if constexpr (std::is_same_v<int, retType>){
          *ret = field.GetInt();
        } else if constexpr (std::is_same_v<double, retType>){
          *ret = field.GetDouble();
        }
        return(0);
      } else if(field.IsArray() && field.Size()==1){
        if constexpr (std::is_same_v<int, retType>){
          *ret = field[0].GetInt();
        } else if constexpr (std::is_same_v<double, retType>){
          *ret = field[0].GetDouble();
        }
        return(0);
      }
    }
    std::cout << "bad format of field : " << fieldName << std::endl;
    return(5);
  }

  template <class retType>
  int getVector(const std::string& fieldName,
                Eigen::Matrix<retType,Eigen::Dynamic,1>* ret){
    if(!doc_.HasMember(fieldName.c_str())){
      std::cout << "no field " << fieldName << " in data file" << std::endl;
      return(4);
    } else {
      rapidjson::Value& field = doc_[fieldName.c_str()];
      if(!field.IsArray()){
        std::cout << " bad format of field " << fieldName << " in data file" << std::endl;
        return(5);
      } else {
        (*ret).resize(field.Size());
        if constexpr (std::is_same_v<int, retType>){
          for(size_t i=0;i<field.Size();i++) (*ret).coeffRef(i) = field[i].GetInt();
        } else if constexpr (std::is_same_v<double, retType>){
          for(size_t i=0;i<field.Size();i++) (*ret).coeffRef(i) = field[i].GetDouble();
        }
        return(0);
      }
    }
  }

  template <class retType>
  int getMatrix(const std::string& fieldName,
                Eigen::Matrix<retType,Eigen::Dynamic,Eigen::Dynamic>* ret){
    if(!doc_.HasMember(fieldName.c_str())){
      std::cout << "no field " << fieldName << " in data file" << std::endl;
      return(4);
    } else {
      rapidjson::Value& field = doc_[fieldName.c_str()];
      if(field.IsArray() && field[0].IsArray()){

        size_t nrow = field.Size();
        size_t ncol = field[0].Size();
        (*ret).resize(nrow,ncol);
        if constexpr (std::is_same_v<int, retType>){
          for(size_t i=0;i<nrow;i++) for(size_t j=0;j<ncol;j++) (*ret).coeffRef(i,j) = field[i][j].GetInt();
        } else if constexpr (std::is_same_v<double, retType>){
          for(size_t i=0;i<nrow;i++) for(size_t j=0;j<ncol;j++) (*ret).coeffRef(i,j) = field[i][j].GetDouble();
        }

      } else {
        std::cout << " bad format of field " << fieldName << " in data file" << std::endl;

      }
    }
    return(0);
  }

  public:
    amtData() : isOpen_(false), empty_(true) {}
    inline int reg(int& scalar,
             const std::string& name){
      si_.push_back(&scalar);
      sin_.push_back(name);
      empty_ = false;
      return(0);
    }

    inline double reg(double& scalar,
             const std::string& name){
      sd_.push_back(&scalar);
      sdn_.push_back(name);
      empty_ = false;
      return(0.0);
    }

    inline Eigen::VectorXi reg(Eigen::VectorXi& vec,
             const std::string& name){
      vi_.push_back(&vec);
      vin_.push_back(name);
      empty_ = false;
      return(Eigen::VectorXi(0));
    }

    inline Eigen::VectorXd reg(Eigen::VectorXd& vec,
             const std::string& name){
      vd_.push_back(&vec);
      vdn_.push_back(name);
      empty_ = false;
      return(Eigen::VectorXd(0));
    }

    inline Eigen::MatrixXi reg(Eigen::MatrixXi& mat,
             const std::string& name){
      mi_.push_back(&mat);
      min_.push_back(name);
      empty_ = false;
      return(Eigen::MatrixXi(0,0));
    }

    inline Eigen::MatrixXd reg(Eigen::MatrixXd& mat,
             const std::string& name){
      md_.push_back(&mat);
      mdn_.push_back(name);
      empty_ = false;
      return(Eigen::MatrixXd(0,0));
    }


    int fromFile(const std::string& filename){
      if(empty_){
        std::cout << "no data registered " << std::endl;
        return(0);
      }
      std::ifstream file(filename);
      if(file.is_open()){

        std::string str;

        file.seekg(0, std::ios::end);
        str.reserve(file.tellg());
        file.seekg(0, std::ios::beg);

        str.assign((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());

        if(str.compare("[]")==0){
          std::cout << str << std::endl;
          if(numFields()>0){
            std::cout << "model contains data fields, but no data were provided!!!" << std::endl;
            std::cout << "please provide the following: " << std::endl;
            dispDataNames();
            return(8);
          } else{
            return(0);
          }
        }

        doc_.Parse(str.c_str());
        if(doc_.HasParseError()){
          std::cout << "Failed to parse JSON file named " << filename << std::endl;
          return(2);
        } else {
          isOpen_ = true;
        }

      } else {
        std::cout << "could not open JSON file" << std::endl;
        return(1);
      }

      //std::cout << "doc member count" << doc_.MemberCount() << std::endl;

      // get a list of all data fields
      std::vector<std::string> all;
      for(auto& m : doc_.GetObject()){
        all.push_back(m.name.GetString());
      }
      std::cout << " data file contains the following fields : " ;
      for(int i=0;i<all.size();i++){
        std::cout << all[i] << ", ";
      }
      std::cout << std::endl;



      int eflag = 0;
      // int scalar
      for(int i=0;i<sin_.size();i++){
        eflag = getScalar<int>(sin_[i],si_[i]);
      }
      // double scalar
      for(int i=0;i<sdn_.size();i++){
        eflag = std::max(eflag,getScalar<double>(sdn_[i],sd_[i]));
      }
      // int vector
      for(int i=0;i<vin_.size();i++){
        eflag = std::max(eflag,getVector<int>(vin_[i],vi_[i]));
      }
      // double vector
      for(int i=0;i<vdn_.size();i++){
        eflag = std::max(eflag,getVector<double>(vdn_[i],vd_[i]));
      }

      // int matrix
      for(int i=0;i<min_.size();i++){
        eflag = std::max(eflag,getMatrix<int>(min_[i],mi_[i]));
      }

      // double matrix
      for(int i=0;i<mdn_.size();i++){
        eflag = std::max(eflag,getMatrix<double>(mdn_[i],md_[i]));
      }



      if(numFields() != all.size()){
        std::cout << "WARNING, data file contains fields not present in model" << std::endl;
      }
      return(eflag);
    }


    void dispDataNames(){

      for(int i=0;i<si_.size();i++){
        std::cout << sin_[i] << " (int scalar) " << std::endl;
      }
      for(int i=0;i<sd_.size();i++){
        std::cout << sdn_[i] << " (double scalar) " << std::endl;
      }
      for(int i=0;i<vi_.size();i++){
        std::cout << vin_[i] << " (int vector) " << std::endl;
      }
      for(int i=0;i<vd_.size();i++){
        std::cout << vdn_[i] << " (double vector) " << std::endl;
      }

      for(int i=0;i<mi_.size();i++){
        std::cout << min_[i] << " (int matrix) " << std::endl;

      }

      for(int i=0;i<md_.size();i++){
        std::cout << mdn_[i] << " (double matrix) " << std::endl;
      }
    }


    void dispData(const bool show=true){
      std::cout << "Dump of amtData object: " << std::endl;
      std::cout << "int scalars" << std::endl;
      for(int i=0;i<si_.size();i++){
        std::cout << sin_[i] << " : " << std::endl;
        if(show) std::cout << *(si_[i]) << std::endl;
      }
      std::cout << "double scalars" << std::endl;
      for(int i=0;i<sd_.size();i++){
        std::cout << sdn_[i] << " : " << std::endl;
        if(show) std::cout << *(sd_[i]) << std::endl;
      }
      std::cout << "int vectors" << std::endl;
      for(int i=0;i<vi_.size();i++){
        std::cout << vin_[i] << " : " << std::endl;
        if(show) std::cout << *(vi_[i]) << std::endl;
      }
      std::cout << "double vectors" << std::endl;
      for(int i=0;i<vd_.size();i++){
        std::cout << vdn_[i] << " : " << std::endl;
        if(show) std::cout << *(vd_[i]) << std::endl;
      }
      std::cout << "int matrices" << std::endl;
      for(int i=0;i<mi_.size();i++){
        std::cout << min_[i] << " : " << std::endl;
        if(show) std::cout << *(mi_[i]) << std::endl;
      }
      std::cout << "double matrices" << std::endl;
      for(int i=0;i<md_.size();i++){
        std::cout << mdn_[i] << " : " << std::endl;
        if(show) std::cout << *(md_[i]) << std::endl;
      }
    }

  };


#endif
