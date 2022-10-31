#ifndef __JSON_WRAP_HPP__
#define __JSON_WRAP_HPP__


#include "rapidjson/document.h"
#include <iostream>
#include <fstream>
#include <streambuf>
#include <vector>
#include <string>
#include <Eigen/Dense>


class jsonOut{
  std::string f_;
public:
  jsonOut() : f_("{") {}
  void push(const std::string& fieldName, const int val){
    if(f_.compare("{")!=0) f_.append(",");
    f_.append("\""+fieldName+"\":"+std::to_string(val));
  }
  void push(const std::string& fieldName, const size_t val){push(fieldName,static_cast<int>(val));}
  void push(const std::string& fieldName, const double val){
    if(f_.compare("{")!=0) f_.append(",");
    std::ostringstream tmp;
    tmp << std::setprecision(14) << val;
    f_.append("\""+fieldName+"\":"+tmp.str());
  }
  void push(const std::string& fieldName, const std::vector<std::string>& val){
    if(f_.compare("{")!=0) f_.append(",");

    f_.append("\""+fieldName+"\":[");
    if(val.size()>0){
    for(int i=0;i<val.size()-1;i++){
      f_.append("\""+val[i]+"\",");
    }
    f_.append("\""+val[val.size()-1]+"\"]");
    } else {
      f_.append(" ]");
    }
  }
  void push(const std::string& fieldName, const Eigen::VectorXd val){
    if(f_.compare("{")!=0) f_.append(",");

    std::ostringstream tmp;
    f_.append("\""+fieldName+"\":[");
    if(val.size()>0){
    for(int i=0;i<val.size()-1;i++){
      tmp << std::setprecision(14) << val(i) << ",";
    }
    tmp << std::setprecision(14) << val(val.size()-1) << "]";
    f_.append(tmp.str());
    } else {
      f_.append(" ]");
    }
  }

  void push(const std::string& fieldName, const Eigen::Matrix<size_t,Eigen::Dynamic,1> val){
    if(f_.compare("{")!=0) f_.append(",");

    std::ostringstream tmp;
    f_.append("\""+fieldName+"\":[");
    if(val.size()>0){
      for(int i=0;i<val.size()-1;i++){
        tmp << val(i) << ",";
      }
      tmp << val(val.size()-1) << "]";
      f_.append(tmp.str());
    } else {
      f_.append(" ]");
    }
  }

  void push(const std::string& fieldName, const Eigen::VectorXi val){
    if(f_.compare("{")!=0) f_.append(",");

    std::ostringstream tmp;
    f_.append("\""+fieldName+"\":[");
    if(val.size()>0){
      for(int i=0;i<val.size()-1;i++){
        tmp << val(i) << ",";
      }
      tmp << val(val.size()-1) << "]";
      f_.append(tmp.str());
    } else {
      f_.append(" ]");
    }
  }






  void push(const std::string& fieldName, const std::vector<double>& val){
    Eigen::VectorXd valEigen(val.size());
    for(int i=0;i<val.size();i++){
      valEigen(i) = val[i];
    }
    push(fieldName,valEigen);
  }

  void push(const std::string& fieldName, const std::vector<int>& val){
    Eigen::VectorXi valEigen(val.size());
    for(int i=0;i<val.size();i++){
      valEigen(i) = val[i];
    }
    push(fieldName,valEigen);
  }



  void toFile(const std::string fileName) const {
    std::ofstream file;
    file.open(fileName);
    file << f_ << "}" << std::endl;
    file.close();
  }

  void dump(){
    std::cout << f_ << std::endl;
  }
};


class json_wrap{
  rapidjson::Document doc_;
  bool isOpen_;
public:
  json_wrap(const std::string filename) : isOpen_(false) {
    std::ifstream file(filename);

    if(file.is_open()){

      std::string str;

      file.seekg(0, std::ios::end);
      str.reserve(file.tellg());
      file.seekg(0, std::ios::beg);

      str.assign((std::istreambuf_iterator<char>(file)),
                 std::istreambuf_iterator<char>());

      doc_.Parse(str.c_str());
      if(doc_.HasParseError()){
          std::cout << "Failed to parse JSON file named " << filename << std::endl;
      } else {
        isOpen_ = true;
      }

    } else {

      std::cout << "file name:" << std::endl;
      std::cout << filename << std::endl;
      throw std::runtime_error("json_wrap : could not open JSON file ");

    }
  }

  inline bool isOpen(){return isOpen_;}

  bool hasMember(const std::string varname){
    return(doc_.HasMember(varname.c_str()));
  }

  bool subtreeHasMembers(const std::string subtree){
    if(! doc_.HasMember(subtree.c_str())){
      return(false);
    } else {
      if(! doc_[subtree.c_str()].IsArray()){
        return(true);
      } else {
        return(! doc_[subtree.c_str()].Empty());
      }
    }
  }


  bool getAllNamesCore(const rapidjson::Value& field,
                       std::vector<std::string>& vec){

    vec.clear();
    for(auto& m : field.GetObject()){
      vec.push_back(m.name.GetString());
    }
    return(true);
  }

  bool getAllNames(std::vector<std::string>& vec){
    rapidjson::Value& field = doc_;
    return(getAllNamesCore(field,vec));
  }

  bool getAllNames(const std::string varname,
                        std::vector<std::string>& vec){
    if(! doc_.HasMember(varname.c_str())){
      std::cout << "no field " << varname << " in JSON file" << std::endl;
      return(false);
    } else {
      rapidjson::Value& field = doc_[varname.c_str()];
      return(getAllNamesCore(field,vec));
    }
  }


  bool getStringCore(const rapidjson::Value& field,
                     std::string &str){
    if(! field.IsString()){
      std::cout << "field is not string" << std::endl;
      return(false);
    } else {
      str = field.GetString();
      return(true);
    }
  }

  bool getString(const std::string varname,
                    std::string& vec){

    if(! doc_.HasMember(varname.c_str())){
      std::cout << "no field " << varname << " in JSON file" << std::endl;
      return(false);
    } else {
      rapidjson::Value& field = doc_[varname.c_str()];
      return(getStringCore(field,vec));
    }
  }


  bool getStringVecCore(const rapidjson::Value& field,
                        std::vector<std::string>& vec){
    if(! field.IsArray()){ // not array

      if(! field.IsString()){
        std::cout << "bad type in field" << std::endl;
        return(false);
      } else {
        vec.push_back(field.GetString());
        return(true);
      }

    } else { // is array
      if(field.Size()==0){
        std::cout << "empty array, ignored " << std::endl;
        return(true);
      } else {
        if(! field[0].IsString()){
          std::cout << "bad type in field" << std::endl;
          return(false);
        } else {
          for(int i=0;i<field.Size();i++) vec.push_back(field[i].GetString());
          return(true);
        }
      }
    }
  }



  bool getStringVec(const std::string varname,
                 std::vector<std::string>& vec){

    if(! doc_.HasMember(varname.c_str())){
      std::cout << "no field " << varname << " in JSON file" << std::endl;
      return(false);
    } else {
      rapidjson::Value& field = doc_[varname.c_str()];
      return(getStringVecCore(field,vec));
    }
  }

  bool getStringVec(const std::string subtree,
                  const std::string varname,
                  std::vector<std::string>& vec){

    if(! doc_.HasMember(subtree.c_str())){
      std::cout << "no subtree " << subtree << " in JSON file" << std::endl;
      return(false);
    } else {
      rapidjson::Value& st = doc_[subtree.c_str()];
      if( ! st.HasMember(varname.c_str())){
        std::cout << "no variable " << varname << " in subtree " << subtree << " in JSON file" << std::endl;
        return(false);
      } else {
        rapidjson::Value& field = st[varname.c_str()];
        return(getStringVecCore(field,vec));
      }
    }
  }



  bool getDoubleCore(const rapidjson::Value& field,
                     double& dval){
    if(! field.IsNumber()){
      std::cout << "bad format in getDoubleCore, not a scalar" << std::endl;
      return(false);
    } else {
      dval = field.GetDouble();
      return(true);
    }
  }

  bool getNumeric(const std::string varname,
                 double &dval){
    if(! doc_.HasMember(varname.c_str())){
      std::cout << "no field " << varname << " in JSON file" << std::endl;
      return(false);
    } else {
      rapidjson::Value& field = doc_[varname.c_str()];
      return(getDoubleCore(field,dval));
    }
  }

  bool getNumeric(const std::string subtree,
                 const std::string varname,
                 double &dval){

    if(! doc_.HasMember(subtree.c_str())){
      //std::cout << "no subtree " << subtree << " in JSON file" << std::endl;
      return(false);
    } else {
      rapidjson::Value& st = doc_[subtree.c_str()];
      if( ! st.HasMember(varname.c_str())){
        //std::cout << "no variable " << varname << " in subtree " << subtree << " in JSON file" << std::endl;
        return(false);
      } else {
        rapidjson::Value& field = st[varname.c_str()];
        return(getDoubleCore(field,dval));
      }
    }
  }

  bool getVectorXdCore(const rapidjson::Value& field,
                       Eigen::VectorXd& dval){
    if(! field.IsArray()){
      //std::cout << "error in getVectorXdCore; not an array" << std::endl;
      return(false);
    } else {
      if(! field[0].IsNumber()){
        //std::cout << "wrong data type in getVectorXdCore" << std::endl;
        return(false);
      } else {
        dval.resize(field.Size());
        for (rapidjson::SizeType i = 0; i < field.Size(); i++) dval.coeffRef(i) = field[i].GetDouble();
        return(true);
      }
    }
  }

  bool getNumeric(const std::string varname,
                 Eigen::VectorXd &dval){
    if(! doc_.HasMember(varname.c_str())){
      //std::cout << "no field " << varname << " in JSON file" << std::endl;
      return(false);
    } else {
      rapidjson::Value& field = doc_[varname.c_str()];
      return(getVectorXdCore(field,dval));
    }
  }

  bool getNumeric(const std::string subtree,
                 const std::string varname,
                 Eigen::VectorXd &dval){

    if(! doc_.HasMember(subtree.c_str())){
      //std::cout << "no subtree " << subtree << " in JSON file" << std::endl;
      return(false);
    } else {
      rapidjson::Value& st = doc_[subtree.c_str()];
      if( ! st.HasMember(varname.c_str())){
        //std::cout << "no variable " << varname << " in subtree " << subtree << " in JSON file" << std::endl;
        return(false);
      } else {
        rapidjson::Value& field = st[varname.c_str()];
        return(getVectorXdCore(field,dval));
      }
    }
  }

  bool getMatrixXdCore(const rapidjson::Value& field,
                       Eigen::MatrixXd& dval){
    if(! (field.HasMember("nrow") && field.HasMember("vals"))){
      //std::cout << "bad format on matrix type" << std::endl;
      return(false);
    } else {
      if(! field["nrow"].IsNumber()){
        //std::cout << "bad nrow field in matrix type" << std::endl;
        return(false);
      } else {
        int nrow = field["nrow"].GetInt();

        if(! field["vals"].IsArray()){
          std::cout << "bad vals field in matrix type" << std::endl;
          return(false);
        } else {
          const rapidjson::Value& vals = field["vals"];
          if(! vals[0].IsNumber()){
            std::cout << "bad contents in vals field" << std::endl;
            return(false);
          } else {
            int ntot = vals.Size();
            int ncol = ntot/nrow;
            if(fabs(ntot % nrow)>1.0e-12){
              std::cout << "lenght of vals inconsitent with nrow field" << std::endl;
              return(false);
            } else {
              dval.resize(nrow,ncol);
              int k = 0;
              for(int j = 0; j < ncol; j++){
                for(int i = 0; i < nrow; i++){
                  dval.coeffRef(i,j) = vals[k].GetDouble();
                  k++;
                }
              }
              return(true);
            }
          }
        }
      }
    }
  }

  bool getNumeric(const std::string varname,
                  Eigen::MatrixXd &dval){
    if(! doc_.HasMember(varname.c_str())){
      //std::cout << "no field " << varname << " in JSON file" << std::endl;
      return(false);
    } else {
      rapidjson::Value& field = doc_[varname.c_str()];
      return(getMatrixXdCore(field,dval));
    }
  }

  bool getNumeric(const std::string subtree,
                  const std::string varname,
                  Eigen::MatrixXd &dval){

    if(! doc_.HasMember(subtree.c_str())){
      //std::cout << "no subtree " << subtree << " in JSON file" << std::endl;
      return(false);
    } else {
      rapidjson::Value& st = doc_[subtree.c_str()];
      if( ! st.HasMember(varname.c_str())){
        //std::cout << "no variable " << varname << " in subtree " << subtree << " in JSON file" << std::endl;
        return(false);
      } else {
        rapidjson::Value& field = st[varname.c_str()];
        return(getMatrixXdCore(field,dval));
      }
    }
  }

  /*
   *  Int types
   *
   *
   */

  bool getIntCore(const rapidjson::Value& field,
                     int& dval){
    if(! field.IsNumber()){
      //std::cout << "bad format in getDoubleCore, not a scalar" << std::endl;
      return(false);
    } else {
      dval = field.GetInt();
      return(true);
    }
  }

  bool getNumeric(const std::string varname,
                  int &dval){
    if(! doc_.HasMember(varname.c_str())){
      //std::cout << "no field " << varname << " in JSON file" << std::endl;
      return(false);
    } else {
      rapidjson::Value& field = doc_[varname.c_str()];
      return(getIntCore(field,dval));
    }
  }

  bool getNumeric(const std::string subtree,
                  const std::string varname,
                  int &dval){

    if(! doc_.HasMember(subtree.c_str())){
      //std::cout << "no subtree " << subtree << " in JSON file" << std::endl;
      return(false);
    } else {
      rapidjson::Value& st = doc_[subtree.c_str()];
      if( ! st.HasMember(varname.c_str())){
        //std::cout << "no variable " << varname << " in subtree " << subtree << " in JSON file" << std::endl;
        return(false);
      } else {
        rapidjson::Value& field = st[varname.c_str()];
        return(getIntCore(field,dval));
      }
    }
  }

  bool getVectorXiCore(const rapidjson::Value& field,
                       Eigen::VectorXi& dval){
    if(! field.IsArray()){
      //std::cout << "error in getVectorXdCore; not an array" << std::endl;
      return(false);
    } else {
      if(! field[0].IsNumber()){
        //std::cout << "wrong data type in getVectorXdCore" << std::endl;
        return(false);
      } else {
        dval.resize(field.Size());
        for (rapidjson::SizeType i = 0; i < field.Size(); i++) dval.coeffRef(i) = field[i].GetInt();
        return(true);
      }
    }
  }

  bool getNumeric(const std::string varname,
                  Eigen::VectorXi &dval){
    if(! doc_.HasMember(varname.c_str())){
      //std::cout << "no field " << varname << " in JSON file" << std::endl;
      return(false);
    } else {
      rapidjson::Value& field = doc_[varname.c_str()];
      return(getVectorXiCore(field,dval));
    }
  }

  bool getNumeric(const std::string subtree,
                  const std::string varname,
                  Eigen::VectorXi &dval){

    if(! doc_.HasMember(subtree.c_str())){
      //std::cout << "no subtree " << subtree << " in JSON file" << std::endl;
      return(false);
    } else {
      rapidjson::Value& st = doc_[subtree.c_str()];
      if( ! st.HasMember(varname.c_str())){
        //std::cout << "no variable " << varname << " in subtree " << subtree << " in JSON file" << std::endl;
        return(false);
      } else {
        rapidjson::Value& field = st[varname.c_str()];
        return(getVectorXiCore(field,dval));
      }
    }
  }

  bool getMatrixXiCore(const rapidjson::Value& field,
                       Eigen::MatrixXi& dval){
    if(! (field.HasMember("nrow") && field.HasMember("vals"))){
      //std::cout << "bad format on matrix type" << std::endl;
      return(false);
    } else {
      if(! field["nrow"].IsNumber()){
        //std::cout << "bad nrow field in matrix type" << std::endl;
        return(false);
      } else {
        int nrow = field["nrow"].GetInt();

        if(! field["vals"].IsArray()){
          std::cout << "bad vals field in matrix type" << std::endl;
          return(false);
        } else {
          const rapidjson::Value& vals = field["vals"];
          if(! vals[0].IsNumber()){
            std::cout << "bad contents in vals field" << std::endl;
            return(false);
          } else {
            int ntot = vals.Size();
            int ncol = ntot/nrow;
            if(fabs(ntot % nrow)>1.0e-12){
              std::cout << "lenght of vals inconsitent with nrow field" << std::endl;
              return(false);
            } else {
              dval.resize(nrow,ncol);
              int k = 0;
              for(int j = 0; j < ncol; j++){
                for(int i = 0; i < nrow; i++){
                  dval.coeffRef(i,j) = vals[k].GetInt();
                  k++;
                }
              }
              return(true);
            }
          }
        }
      }
    }
  }

  bool getNumeric(const std::string varname,
                  Eigen::MatrixXi &dval){
    if(! doc_.HasMember(varname.c_str())){
      //std::cout << "no field " << varname << " in JSON file" << std::endl;
      return(false);
    } else {
      rapidjson::Value& field = doc_[varname.c_str()];
      return(getMatrixXiCore(field,dval));
    }
  }

  bool getNumeric(const std::string subtree,
                  const std::string varname,
                  Eigen::MatrixXi &dval){

    if(! doc_.HasMember(subtree.c_str())){
      //std::cout << "no subtree " << subtree << " in JSON file" << std::endl;
      return(false);
    } else {
      rapidjson::Value& st = doc_[subtree.c_str()];
      if( ! st.HasMember(varname.c_str())){
        //std::cout << "no variable " << varname << " in subtree " << subtree << " in JSON file" << std::endl;
        return(false);
      } else {
        rapidjson::Value& field = st[varname.c_str()];
        return(getMatrixXiCore(field,dval));
      }
    }
  }




  /*
   bool getDouble(const std::string subtree,
   const std::string varname,
   double &dval){
   if(! doc_.HasMember(subtree.c_str())) return(false);
   const rapidjson::Value& st = doc_[subtree.c_str()];
   if(! st.HasMember(varname.c_str())){

   std::cout << "no field " << varname << "in subtree " << subtree << std::endl;
   return(false);

   } else {

   const rapidjson::Value& field = st[varname.c_str()];
   if(! field.HasMember("t") || ! field.HasMember("v")){

   std::cout << "bad format in getDouble" << std::endl;
   return(false);

   } else {

   if(! field["t"].IsNumber() || ! field["v"].IsNumber()){

   std::cout << "bad format in getDouble" << std::endl;
   return(false);

   } else {

   if(! (field["t"].GetInt()==0)){
   std::cout << "wrong type in getDouble" << std::endl;
   return(false);
   } else {
   dval = field["v"].GetDouble();
   return(true);
   }
   }
   }
   }
   }
   */

  bool getVectorXd(const std::string subtree,
                   const std::string varname,
                   Eigen::VectorXd &dval){

    if(! doc_.HasMember(subtree.c_str())) return(false);
    const rapidjson::Value& st = doc_[subtree.c_str()];
    if(! st.HasMember(varname.c_str())){
      std::cout << "no field " << varname << "in subtree " << subtree << std::endl;
      return(false);
    } else {
      const rapidjson::Value& field = st[varname.c_str()];
      if(! field.HasMember("t") ||  ! field.HasMember("v")){
        std::cout << "bad format in getVectorXd" << std::endl;
        return(false);
      } else {
        if(! field["t"].IsNumber()){
          std::cout << "bad format in getVector" << std::endl;
          return(false);
        } else {
          if(! (field["t"].GetInt()==1)){
            std::cout << "wrong type in getVectorXd" << std::endl;
            return(false);
          } else {
            if( !field["v"].IsArray()  ){
              std::cout << "bad format in getVectorXd, v should be array" << std::endl;
              return(false);
            } else {
              const rapidjson::Value& v = field["v"];
              if( ! v[0].IsNumber()){
                std::cout << "error in getVectorXd, v should be a numeric vector " << std::endl;
                return(false);
              } else {
                dval.resize(v.Size());
                for (rapidjson::SizeType i = 0; i < v.Size(); i++) dval.coeffRef(i) = v[i].GetDouble();
                return(true);
              }
            }
          }
        }
      }
    }
  }






};



#endif
