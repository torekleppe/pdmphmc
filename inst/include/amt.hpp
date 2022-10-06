#ifndef _AMT_HPP_
#define _AMT_HPP_
#include <functional>
#include <stan/math.hpp>



#include "sparseVec.hpp"
#include "amtVar.hpp"
#include "amtDynamicCast.hpp"
#include "amtSpecFuns.hpp"
#include "amtTraits.hpp"
#include "amtUtils.hpp"

#include "amtVarLinAlg.hpp"
#include "amtPackedSym.hpp"

//#include "amtLpdfReturnTypes.hpp"
#include "amtCommonNameSpace.hpp"
// specialized data types
#include "amtSPDmatrix.hpp"
#include "amtTriDiagChol.hpp"
// model specification
#include "amtModel.hpp"
// utilities for importing data
#include "amtData.hpp"

// write directly to target
#include "amtTargetIncrement.hpp"

// univariate continuous distibutions
#include "amtNormal.hpp"
#include "amtExpGamma.hpp"
#include "amtInvLogitBeta.hpp"
#include "amtInvLogitUniform.hpp"

// univariate discrete distributions
#include "amtBernoulli.hpp"
#include "amtPoisson.hpp"

// multivariate distributions
#include "amtMultiNormalPrec.hpp" // unrestricted precsison
#include "amtNormalRW1.hpp"
#include "amtNormalAR1.hpp"


// matrixvariate distributions
#include "amtWishart.hpp"
#include "amtWishartRW1.hpp"


// highly specialized likelihoods etc
#include "amtStochVolLeverage.hpp"

extern amtData dta__;

#define DATA_DOUBLE(name) double name = dta__.reg(name,#name);
#define DATA_INT(name) int name = dta__.reg(name,#name);
#define DATA_VECTOR(name) Eigen::VectorXd name = dta__.reg(name,#name);
#define DATA_IVECTOR(name) Eigen::VectorXi name = dta__.reg(name,#name);
#define DATA_MATRIX(name) Eigen::MatrixXd name = dta__.reg(name,#name);
#define DATA_IMATRIX(name) Eigen::MatrixXi name = dta__.reg(name,#name);


#define PARAMETER_SCALAR(name,...) varType name = model__.parameterScalar(#name,##__VA_ARGS__)
#define PARAMETER_VECTOR(name,dim,...) Eigen::Matrix<varType,Eigen::Dynamic,1> name = model__.parameterVector(#name,dim,##__VA_ARGS__)
#define PARAMETER_MATRIX(name,dim1,dim2,...) Eigen::Matrix<varType,Eigen::Dynamic,Eigen::Dynamic> name = model__.parameterMatrix(#name,dim1,dim2,##__VA_ARGS__)


#define _STRIFY(arg) #arg
#define _SPD_INT_NAME_STR(name) _STRIFY(name ## _internal)
#define _SPD_INT_NAME(name) name ## _internal
#define PARAMETER_SPD_MATRIX(name,dim,...) Eigen::Matrix<varType,Eigen::Dynamic,1> _SPD_INT_NAME(name) = model__.parameterVector(_SPD_INT_NAME_STR(name),(dim*(dim+1))/2,##__VA_ARGS__);\
   SPDmatrix<varType> name(dim,_SPD_INT_NAME(name) )

#endif
