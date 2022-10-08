#ifndef __PDPHMC_HPP__
#define __PDPHMC_HPP__

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

#include <stan/math.hpp>
#include <Eigen/Dense>

#include <algorithm>
#include <vector>
#include <random>
#include <chrono>
#include <iterator>


/*
 * Used with either var=double or var=stan::math::var
 */
#define VectorXv Eigen::Matrix<varType,Eigen::Dynamic,1>
#define MatrixXv Eigen::Matrix<varType,Eigen::Dynamic,Eigen::Dynamic>

/*
 * Vector of stan::math::var
 */

#define VectorXad Eigen::Matrix<stan::math::var,Eigen::Dynamic,1>



/*
 *
 *  Constants used in the adaptation routines
 *
 *
 */
#include "adapt_constants.hpp"

/*
 *
 * utilities
 *
 */


#include "Eigen_utils.hpp"
#include "diagnostics.hpp"
#include "json_wrap.hpp"
#include "propertyTable.hpp"
#include "stabilityMonitor.hpp"
/*
 * Model specifications
 *
 *
 */

#include "amt/amt.hpp"
#include "fast_spec_funs.hpp"

/*
 * various utilities for working with AD types:
 *
 */
#include "AD_overloads.hpp"




/*
 *
 *
 * Computational routines
 *
 *
 */

#include "odeUtils.hpp"
#include "sparseChol.hpp"
#include "rng.hpp"
#include "NUTwrap.hpp"
#include "startWrap.hpp"


#include "RKDP54.hpp"
#include "RKNDP64.hpp"
#include "RungeKutta.hpp"


#include "metricTensorSymbolic.hpp"

#include "metricTensorDummy.hpp"
#include "metricTensorDense.hpp"
#include "metricTensorSparse.hpp"
#include "metricTensorTraits.hpp"

#include "massMatrix.hpp"
#include "linearTM.hpp"
#include "lambda.hpp"

#include "STDmetricTensorAdapter.hpp"


#include "HMCProcess.hpp"
#include "RMHMCProcess.hpp"


#include "PDPsampler.hpp"
#include "initialPointSolver.hpp"



#endif
