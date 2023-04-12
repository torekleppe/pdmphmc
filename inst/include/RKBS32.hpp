#ifndef _RKBS32_HPP_
#define _RKBS32_HPP_
#include "odeUtils.hpp"
#include "numUtils/numUtils.hpp"
/*
 * Runge Kutta Bogacki-Shampine 3(2) pair with 3-order interpolation formula
 *
 */


#ifndef _EVENTROOTSOLVER_TOL_
#define _EVENTROOTSOLVER_TOL_ 1.0e-10
#endif

#ifndef _SPLINF_GRID_SIZE_
#define _SPLINF_GRID_SIZE_ 8
#endif


template <class _ode_type_>
class RKBS32{

  _ode_type_* ode_;

  size_t dim_;
  size_t dimGenerated_;
  size_t dimEvent_;

  Eigen::MatrixXd force_;
  Eigen::MatrixXd ys_;
  Eigen::MatrixXd generated_;
  Eigen::MatrixXd events_;
  Eigen::MatrixXd diag_;

  Eigen::VectorXd force_tmp_;
  Eigen::VectorXd y_tmp_;
  Eigen::VectorXd gen_tmp_;
  Eigen::VectorXd diag_tmp_;
  Eigen::VectorXd y1_low_;
  Eigen::VectorXd tmpVec_,tmpVecRoot_;
  Eigen::VectorXd event_tmp_;
  Eigen::VectorXd eventRootInt_,eventRootIntR_,eventRootIntLow_;
  Eigen::VectorXd eventA_,eventB_;

  odeState tmpState_,newState_;


  inline Eigen::VectorXd calcIntPoly(const double t) const {
    double ts = t/eps_;
    double tsSq = ts*ts;
    double tsQ = ts*tsSq;
    Eigen::VectorXd intPoly(4);
    intPoly.coeffRef(0) = 1.0 + 2.0*tsQ - 3.0*tsSq;
    intPoly.coeffRef(1) = 3.0*tsSq - 2.0*tsQ;
    intPoly.coeffRef(2) = eps_*(tsQ - 2.0*tsSq + ts);
    intPoly.coeffRef(3) = eps_*(tsQ - tsSq);
    return(intPoly);
  }

  inline Eigen::VectorXd calcLevelPoly(const double t) const {
    double ts = t/eps_;
    double tsSq = ts*ts;
    Eigen::VectorXd levelPoly(4);
    levelPoly.coeffRef(0) = 6.0*(tsSq-ts)/eps_;
    levelPoly.coeffRef(1) = 6.0*(ts-tsSq)/eps_;
    levelPoly.coeffRef(2) = 1.0 - 4.0*ts + 3.0*tsSq;
    levelPoly.coeffRef(3) = 3.0*tsSq - 2.0*ts;
    return(levelPoly);
  }


  inline rootInfo nonlinRootSolver(){
    int whichDim = -1;
    double ret = eps_;

    if(dimEvent_>0){
      eventA_ = 3.0*(events_.col(0)+events_.col(3)) - 6.0*eventRootIntR_;
      eventB_ = 6.0*eventRootIntR_ - 4.0*events_.col(0) - 2.0*events_.col(3);
      double q,eventDev,r;
      for(size_t i=0;i<dimEvent_;i++){
        if(fabs(eventA_.coeff(i))<_EVENTROOTSOLVER_TOL_){
          // poly is at most linear
          if(fabs(eventB_.coeff(i))>_EVENTROOTSOLVER_TOL_){
            // poly is linear, constant case gets ignored
            r = -eps_*events_.coeff(i,0)/eventB_.coeff(i);
            if(r>0.0 && r<ret){
              ret = r;
              whichDim = i;
              //std::cout << "linear Root, t = " << t_left_ + r << std::endl;
            }
          }
        } else {
          // poly is quadratic
          eventDev = pow(eventB_.coeff(i),2) - 4.0*eventA_.coeff(i)*events_.coeff(i,0);
          if(eventDev>=0.0){
            q = (eventB_.coeff(i)>=0.0) ? -0.5*(eventB_.coeff(i) + sqrt(eventDev)) : -0.5*(eventB_.coeff(i)-sqrt(eventDev));
            r = eps_*(q/eventA_.coeff(i));
            if(0.0<r && r<ret){
              ret = r;
              whichDim = i;
            }
            r = eps_*(events_.coeff(i,0)/q);
            if(0.0<r && r<ret){
              ret = r;
              whichDim = i;
            }
          }
        }
      }
    }
    return(rootInfo(ret,0,whichDim));
  }

  Eigen::VectorXd Sy0_,Sy1_,Sf0_,Sf1_,Sydif_,Sa_,Sb_;
  inline rootInfo splinRootSolver(const rootInfo& oldRoot){
    double ret = eps_;
    int whichDim = -1;
    //std::cout << "splinRootSolver" << std::endl;
    if((*ode_).spr().spLinRootJac_.rows()<1) return(rootInfo(ret,2,whichDim));
    (*ode_).spr().spLinRootJac_.rightMultiplyVec(ys_.col(0),Sy0_);
    Sy0_ += (*ode_).spr().spLinRootConst_;
    (*ode_).spr().spLinRootJac_.rightMultiplyVec(ys_.col(3),Sy1_);
    Sy1_ +=(*ode_).spr().spLinRootConst_;
    //std::cout << "Sy0\n" << Sy0_ << std::endl;
    //std::cout << "Sy1\n" << Sy1_ << std::endl;
    Sf0_ = eps_*((*ode_).spr().spLinRootJac_*force_.col(0));
    Sf1_ = eps_*((*ode_).spr().spLinRootJac_*force_.col(3));
    Sydif_ = Sy0_-Sy1_;
    Sa_ = Sf0_ + Sf1_ + 2.0*Sydif_;
    Sb_ = -(2.0*Sf0_ + Sf1_ + 3.0*Sydif_);
    double cand,dev,x;
    for(int i=0;i<Sy0_.size();i++){
      cand = eps_*numUtils::smallestCubicPolyRootsInInterval(0.0,1.0,
                                                             Sa_.coeff(i),
                                                             Sb_.coeff(i),
                                                             Sf0_.coeff(i),
                                                             Sy0_.coeff(i),
                                                             oldRoot.rootType_==2 && oldRoot.rootDim_ == i);
      if(cand<ret){
        ret = cand;
        whichDim = i;
      }
    }
    return(rootInfo(ret,2,whichDim));
  }

  Eigen::VectorXd Ty0_,Ty1_,Tf0_,Tf1_,Tydif_,Ta_,Tb_;
  inline rootInfo linRootSolver(const rootInfo& oldRoot){
    double ret = eps_;
    int whichDim = -1;
    //std::cout << "linRootSolver" << std::endl;
    if((*ode_).spr().linRootJac_.rows()<1) return(rootInfo(ret,1,whichDim));

    Ty0_ = (*ode_).spr().linRootJac_*ys_.col(0)+(*ode_).spr().linRootConst_;
    Ty1_ = (*ode_).spr().linRootJac_*ys_.col(3)+(*ode_).spr().linRootConst_;
    //std::cout << "Ty0\n" << Ty0_ << std::endl;
    //std::cout << "Ty1\n" << Ty1_ << std::endl;
    Tf0_ = eps_*((*ode_).spr().linRootJac_*force_.col(0));
    Tf1_ = eps_*((*ode_).spr().linRootJac_*force_.col(3));
    Tydif_ = Ty0_-Ty1_;
    Ta_ = Tf0_ + Tf1_ + 2.0*Tydif_;
    Tb_ = -(2.0*Tf0_ + Tf1_ + 3.0*Tydif_);
    double cand,dev,x;
    for(int i=0;i<Ty0_.size();i++){
      cand = eps_*numUtils::smallestCubicPolyRootsInInterval(0.0,1.0,
                                                             Ta_.coeff(i),
                                                             Tb_.coeff(i),
                                                             Tf0_.coeff(i),
                                                             Ty0_.coeff(i),
                                                             oldRoot.rootType_==1 && oldRoot.rootDim_ == i);
      if(cand<ret){
        ret = cand;
        whichDim = i;
      }
    }
    return(rootInfo(ret,1,whichDim));
  }


  std::vector<Eigen::VectorXd> L1y0_,L1y1_,L1f0_,L1f1_,L1ydif_,L1a_,L1b_,L1sign_;
  Eigen::VectorXd L1flipTimes_;
  Eigen::VectorXi L1flipInds_;
  inline rootInfo splinL1RootSolver(const rootInfo& oldRoot){
    double ret = eps_;
    int whichDim = -1;
    //std::cout << "splinL1RootSolver : oldRoot : " << oldRoot << std::endl;
    if((*ode_).spr().spLinL1RootJac_.size()<1) return(rootInfo(ret,3,whichDim));

    //std::cout << "non-trivial special root" << std::endl;
    if(L1y0_.size()==0){
      for(size_t i=0;i<(*ode_).spr().spLinL1RootJac_.size();i++){
        L1y0_.emplace_back((*ode_).spr().spLinL1RootConst_[i].size());
        L1y1_.emplace_back((*ode_).spr().spLinL1RootConst_[i].size());
        L1f0_.emplace_back((*ode_).spr().spLinL1RootConst_[i].size());
        L1f1_.emplace_back((*ode_).spr().spLinL1RootConst_[i].size());
        L1ydif_.emplace_back((*ode_).spr().spLinL1RootConst_[i].size());
        L1a_.emplace_back((*ode_).spr().spLinL1RootConst_[i].size());
        L1b_.emplace_back((*ode_).spr().spLinL1RootConst_[i].size());
        L1sign_.emplace_back((*ode_).spr().spLinL1RootConst_[i].size());
      }
    }
    int nSignFlip;
    double tl,tu,teval,cand,polyConst;
    // loop over the different constraints
    for(size_t i=0;i<(*ode_).spr().spLinL1RootJac_.size();i++){
      L1y0_[i] = (*ode_).spr().spLinL1RootJac_[i]*ys_.col(0)+(*ode_).spr().spLinL1RootConst_[i];
      L1y1_[i] = (*ode_).spr().spLinL1RootJac_[i]*ys_.col(3)+(*ode_).spr().spLinL1RootConst_[i];

      //std::cout << "solver, init L1: " << L1y0_[i].array().abs().sum() -(*ode_).spr().spLinL1RootRhs_[i] << std::endl;
      //std::cout << "eps : " << eps_ << std::endl;
      //std::cout << L1y1_[i] << std::endl;
      //std::cout << L1y0_[i].transpose() << std::endl;

      L1f0_[i] = eps_*((*ode_).spr().spLinL1RootJac_[i]*force_.col(0));
      L1f1_[i] = eps_*((*ode_).spr().spLinL1RootJac_[i]*force_.col(3));
      L1ydif_[i] = L1y0_[i]-L1y1_[i];
      L1a_[i] = L1f0_[i] + L1f1_[i] + 2.0*L1ydif_[i];
      L1b_[i] = -(2.0*L1f0_[i] + L1f1_[i] + 3.0*L1ydif_[i]);

      nSignFlip = numUtils::sequenceOfCubicPolyRootsInInterval(0.0,
                                                               1.0,
                                                               L1a_[i],
                                                                   L1b_[i],
                                                                       L1f0_[i],
                                                                            L1y0_[i],
                                                                                 L1flipTimes_,
                                                                                 L1flipInds_);

      //std::cout << "n flips: " << nSignFlip << std::endl;
      //std::cout << L1flipTimes_ << std::endl;
      tl = 0.0;
      for(size_t interval=0;interval<nSignFlip+1;interval++){
        if(interval<nSignFlip){
          tu = L1flipTimes_.coeff(interval);
        } else {
          tu = 1.0;
        }
        if(interval==0){
          // evaluate sign at interior point at first
          teval = 0.5*(tl+tu);
          L1sign_[i] = (teval*(teval*(teval*L1a_[i] + L1b_[i]) + L1f0_[i]) +  L1y0_[i]).array().sign().matrix();

        } else {
          L1sign_[i].coeffRef(L1flipInds_.coeff(interval-1)) *= -1.0;
        }

        //std::cout << "interval searched : " << tl << " " << tu << std::endl;
        //std::cout << L1sign_[i] << std::endl;
        polyConst = L1y0_[i].dot(L1sign_[i])-(*ode_).spr().spLinL1RootRhs_[i];
/*
        if(interval==0 && oldRoot.rootType_==3 && oldRoot.rootDim_ == i) {
          polyConst = 0.0;

          std::cout  << std::setprecision (15) << L1a_[i].dot(L1sign_[i]) << std::endl;
          std::cout  << std::setprecision (15) << L1b_[i].dot(L1sign_[i]) << std::endl;
          std::cout  << std::setprecision (15) << L1f0_[i].dot(L1sign_[i]) << std::endl;
          std::cout  << std::setprecision (15) << L1y0_[i].dot(L1sign_[i])-(*ode_).spr().spLinL1RootRhs_[i] << std::endl;
        }
*/
        cand = numUtils::smallestCubicPolyRootsInInterval(std::max(1.0e-12,tl),tu,
                                                          L1a_[i].dot(L1sign_[i]),
                                                          L1b_[i].dot(L1sign_[i]),
                                                          L1f0_[i].dot(L1sign_[i]),
                                                          polyConst,
                                                          interval==0 && oldRoot.rootType_==3 && oldRoot.rootDim_ == i);

        //std::cout << "cand : " << cand << std::endl;
        if(cand <= tu){
          //std::cout << "candidate constraint pass found: " << cand << std::endl;
          if(cand*eps_ < ret){
            // remove from here
            //std::cout << "candidate is earliest found: " << std::endl;
            //teval = cand;
            //Eigen::VectorXd tmp = (teval*(teval*(teval*L1a_[i] + L1b_[i]) + L1f0_[i]) +  L1y0_[i]);
            //std::cout << "state at pass:\n "  << tmp << std::endl;
            //std::cout << "L1 norm at pass: " << tmp.array().abs().sum() << std::endl;
            // to here
            ret = cand*eps_;
            whichDim = i;
          }
          break;
        }
        tl = tu; // update lower bound
      } // end loop over interval within constraint
    }// end loop over the different constraints
    //std::cout << "ret: " << ret << " whichDim: " << whichDim << std::endl;
    //if(ret<1.0e-5) throw 23;
    return(rootInfo(ret,3,whichDim));
  }

  std::vector<Eigen::VectorXd> L2y0_,L2y1_,L2f0_,L2f1_,L2ydif_,L2a_,L2b_;
  std::vector<double> L2SqRhs_;
  inline rootInfo splinL2RootSolver(const rootInfo& oldRoot){
    double ret = eps_;
    int whichDim = -1;
    //std::cout << "splinL2RootSolver, old root: " << oldRoot << std::endl;
    //dumpStep();
    if((*ode_).spr().spLinL2RootJac_.size()<1) return(rootInfo(ret,4,whichDim));

    //std::cout << "non-trivial special root" << std::endl;
    if(L2y0_.size()==0){
      for(size_t i=0;i<(*ode_).spr().spLinL2RootJac_.size();i++){
        L2y0_.emplace_back((*ode_).spr().spLinL2RootConst_[i].size());
        L2y1_.emplace_back((*ode_).spr().spLinL2RootConst_[i].size());
        L2f0_.emplace_back((*ode_).spr().spLinL2RootConst_[i].size());
        L2f1_.emplace_back((*ode_).spr().spLinL2RootConst_[i].size());
        L2ydif_.emplace_back((*ode_).spr().spLinL2RootConst_[i].size());
        L2a_.emplace_back((*ode_).spr().spLinL2RootConst_[i].size());
        L2b_.emplace_back((*ode_).spr().spLinL2RootConst_[i].size());
        L2SqRhs_.push_back(std::pow((*ode_).spr().spLinL2RootRhs_[i],2));

      }
    }

    numUtils::Poly L2poly;
    double cand;
    // loop over the different constraints
    for(size_t i=0;i<(*ode_).spr().spLinL2RootJac_.size();i++){
      L2y0_[i] = (*ode_).spr().spLinL2RootJac_[i]*ys_.col(0)+(*ode_).spr().spLinL2RootConst_[i];
      L2y1_[i] = (*ode_).spr().spLinL2RootJac_[i]*ys_.col(3)+(*ode_).spr().spLinL2RootConst_[i];

      //std::cout << L1y0_[i] << std::endl;
      //std::cout << L1y1_[i] << std::endl;

      L2f0_[i] = eps_*((*ode_).spr().spLinL2RootJac_[i]*force_.col(0));
      L2f1_[i] = eps_*((*ode_).spr().spLinL2RootJac_[i]*force_.col(3));
      L2ydif_[i] = L2y0_[i]-L2y1_[i];
      L2a_[i] = L2f0_[i] + L2f1_[i] + 2.0*L2ydif_[i];
      L2b_[i] = -(2.0*L2f0_[i] + L2f1_[i] + 3.0*L2ydif_[i]);

      L2poly = numUtils::sumOfSquaredCubics(L2a_[i],L2b_[i],L2f0_[i],L2y0_[i]);
      L2poly.addConstant(-L2SqRhs_[i]);

      // handle case of repeated roots by dividing polynomial by its variable

      //std::cout << "L2 solver poly at zero: " << L2poly(0.0)  << std::endl;
      //L2poly.dump();

      if(oldRoot.rootDim_==i && oldRoot.rootType_==4 && !(*ode_).spr().allowRepeatedRoots_){
        L2poly.divByVar();
      }
      //L2poly.dump();
      //std::cout << L2y0_[i].transpose() << std::endl;
      cand = L2poly.smallestRootInInterval(1.0e-10,1.0);
      if(eps_*cand<ret){
        ret = eps_*cand;
        whichDim = i;
        /*
         // remove from here
         double teval = cand;
         Eigen::VectorXd tmp = (teval*(teval*(teval*L2a_[i] + L2b_[i]) + L2f0_[i]) +  L2y0_[i]);
         std::cout << "L2 norm at event : " << sqrt(tmp.squaredNorm()) << std::endl;
         // to here
         */
      }
    }
    //std::cout << "whichdim : " << whichDim << std::endl;
    //if(oldRoot.rootTime_<1.0e-12) throw 234;
    return(rootInfo(ret,4,whichDim));
  }


  std::vector<Eigen::VectorXd> Fy0_,Fy1_,Ff0_,Ff1_,Fydif_,Fa_,Fb_,Fgr_,Fgrad_;
  Eigen::VectorXd Fgrid_,Fvals_;

  inline rootInfo splinFRootSolver(const rootInfo& oldRoot){
    double ret = eps_;
    int whichDim = -1;
    //std::cout << "splinFRootSolver" << std::endl;
    //dumpStep();
    //std::cout << (*ode_).spr() << std::endl;
    if((*ode_).spr().spLinFRootJac_.size()<1) return(rootInfo(ret,5,whichDim));
    //std::cout << "non-trivial special root" << std::endl;
    if(Fy0_.size()==0){
      for(size_t i=0;i<(*ode_).spr().spLinFRootJac_.size();i++){
        Fy0_.emplace_back((*ode_).spr().spLinFRootConst_[i].size());
        Fy1_.emplace_back((*ode_).spr().spLinFRootConst_[i].size());
        Ff0_.emplace_back((*ode_).spr().spLinFRootConst_[i].size());
        Ff1_.emplace_back((*ode_).spr().spLinFRootConst_[i].size());
        Fydif_.emplace_back((*ode_).spr().spLinFRootConst_[i].size());
        Fa_.emplace_back((*ode_).spr().spLinFRootConst_[i].size());
        Fb_.emplace_back((*ode_).spr().spLinFRootConst_[i].size());
        Fgr_.emplace_back((*ode_).spr().spLinFRootConst_[i].size());
        Fgrad_.emplace_back((*ode_).spr().spLinFRootConst_[i].size());
      }
      Fgrid_.setLinSpaced(_SPLINF_GRID_SIZE_,0.0,1.0);
      Fvals_.resize(_SPLINF_GRID_SIZE_);
    }
    //std::cout << "alloc done" << std::endl;

    double cand,lb,ub,lf,uf,teval,dev,der,fdif,funScale,abScale;
    bool converged;
    bool leftRoot;
    // loop over the different constraints
    for(size_t i=0;i<(*ode_).spr().spLinFRootJac_.size();i++){
      Fy0_[i] = (*ode_).spr().spLinFRootJac_[i]*ys_.col(0)+(*ode_).spr().spLinFRootConst_[i];
      Fy1_[i] = (*ode_).spr().spLinFRootJac_[i]*ys_.col(3)+(*ode_).spr().spLinFRootConst_[i];

      Ff0_[i] = eps_*((*ode_).spr().spLinFRootJac_[i]*force_.col(0));
      Ff1_[i] = eps_*((*ode_).spr().spLinFRootJac_[i]*force_.col(3));
      Fydif_[i] = Fy0_[i]-Fy1_[i];
      Fa_[i] = Ff0_[i] + Ff1_[i] + 2.0*Fydif_[i];
      Fb_[i] = -(2.0*Ff0_[i] + Ff1_[i] + 3.0*Fydif_[i]);

      // done making polynomial, now search for sign flips on a grid:
      ub = 2.0;
      Fvals_.setZero();
      leftRoot = false;
      for(int g=0;g<_SPLINF_GRID_SIZE_;g++){
        teval = Fgrid_.coeff(g);
        Fgr_[i] = (teval*(teval*(teval*Fa_[i] + Fb_[i]) + Ff0_[i]) +  Fy0_[i]);
       // std::cout << "Fgr: " << Fgr_[i].transpose() << std::endl;
        Fvals_.coeffRef(g) = (*((*ode_).spr().spLinFRootFun_[i]))(Fgr_[i]);

        if(! isfinite(Fvals_.coeff(g))){
          std::cout << "splinF: numerical problems in functor # " << i << std::endl;
          std::cout << "evaluates to " << Fvals_.coeff(g) << "for lhs=\n" << Fgr_[i] << std::endl;
          throw(756);
        }

        if(g==0 && (std::fabs(Fvals_.coeff(0))<1.0e-12 ||
           (!(*ode_).spr().allowRepeatedRoots_ && oldRoot.rootDim_==i && oldRoot.rootType_==5))){
          // if left endpoint is a root, set fun value equal to derivative in
          // order to avoid numerical rounding interferes with bracketing
          //std::cout << "ff: " << Fvals_(0) << std::endl;
          leftRoot = true;
          dev = (*((*ode_).spr().spLinFRootFun_[i]))(Fy0_[i],Fgrad_[i]);
          //std::cout << Fgrad_[i] << std::endl;
          Fvals_.coeffRef(0) = Ff0_[i].dot(Fgrad_[i]);
          if(Fvals_.coeffRef(0)<0.0){
            std::cout << "bad gradient at left endpoint in splinFRootSolver" << std::endl;
          }
        }

        if(g>0 && Fvals_.coeff(g-1)*Fvals_.coeff(g)<=0.0){

          lb = Fgrid_.coeff(g-1);
          ub = teval;
          lf = Fvals_.coeff(g-1);
          uf = Fvals_.coeff(g);
          //std::cout << "found interval with sign flip: " << Fvals_.coeff(g-1) << " , " << Fvals_.coeff(g) << std::endl;
          //std::cout << "lb: " << lb << " ub: " << ub << std::endl;
          break;
        }
      }

      //std::cout << "Fvals:" << Fvals_.transpose() << std::endl;

      // refine bracket if root is found in the first grid interval, and another
      // root is present at left endpoint (to avoid repeating the same root many times)

      if(leftRoot && lb<1.0e-14){
        //std::cout << Fvals_.transpose() << "\n" << Fgrid_.transpose() << std::endl;
        converged = false;
        teval = 0.5*ub;
        //std::cout << "left bracket proc" << std::endl;
        for(int iter=0;iter<40;iter++){
          Fgr_[i] = (teval*(teval*(teval*Fa_[i] + Fb_[i]) + Ff0_[i]) +  Fy0_[i]);
          dev = (*((*ode_).spr().spLinFRootFun_[i]))(Fgr_[i]);
          //std::cout << "teval: " << teval << " dev: " << dev << std::endl;
          if(uf*dev<0.0){
            converged = true;
            lb = teval;
            lf = dev;
            break;
          }
          teval*=0.5;
        }
        if(!converged){
          std::cout << "left root bracketing proc failed" << std::endl;
          throw(13443);
        }
      }

      converged = false;

      if(ub<1.5){ // bracket with sign change found, refine using safeguarded Newton's method
        funScale = 0.5*(std::fabs(lf)+std::fabs(uf));

        teval = 0.5*(lb+ub);


        Fgr_[i] = (teval*(teval*(teval*Fa_[i] + Fb_[i]) + Ff0_[i]) +  Fy0_[i]);
        dev = (*((*ode_).spr().spLinFRootFun_[i]))(Fgr_[i],Fgrad_[i]);
        der = (teval*((3.0*teval)*Fa_[i] + 2.0*Fb_[i])+Ff0_[i]).dot(Fgrad_[i]);

        for(int iter=0;iter<100;iter++){
          //std::cout << "dev: " << dev << " der: " << der << std::endl;
          if(std::fabs(dev)<std::max(1.0e-10*funScale,1.0e-12) || std::fabs(dev/der)<1.0e-14*std::max(0.1,teval)){
            //std::cout << "splinF success" << std::endl;
            //std::cout << "dev: " << dev << " der: " << der << std::endl;
            //std::cout << "lb: " << lb << " teval: " << teval << " ub: " << ub << std::endl;
            converged = true;
            break;
          }
          teval = teval - dev/der; // newton step
          if(teval<lb || teval>ub){ // restrict to current bracket
              // bisection
              teval = 0.5*(lb+ub);
          }

          // new eval
          Fgr_[i] = (teval*(teval*(teval*Fa_[i] + Fb_[i]) + Ff0_[i]) +  Fy0_[i]);
          dev = (*((*ode_).spr().spLinFRootFun_[i]))(Fgr_[i],Fgrad_[i]);
          der = (teval*((3.0*teval)*Fa_[i] + 2.0*Fb_[i])+Ff0_[i]).dot(Fgrad_[i]);
          //std::cout << "x: " << lb << " , " << teval << " , " << ub << std::endl;
          //std::cout << "fun: " << lf << " , " << dev << " , " << uf << std::endl;
          if(dev*lf>=0.0){
            lb = teval;
            lf = dev;
          } else {
            ub = teval;
            uf = dev;
          }
        }
        if(! converged || teval < 1.0e-10){
          std::cout << "RKBS32::splinFRootSolver failed to converge" << std::endl;
          std::cout << Fvals_.transpose() << "\n" << Fgrid_.transpose() << std::endl;
          std::cout << "constraint at left endpoint: " << (*((*ode_).spr().spLinFRootFun_[i]))(Fy0_[i],Fgrad_[i]) << std::endl;
          std::cout << "gradient at left endpoint:\n" << Fgrad_[i].dot(Ff0_[i]) << std::endl;
          std::cout << "dev: " << dev << " lb: " << lb << " ub: " << ub << std::endl;
          std::cout << "lf: " << lf << " uf: " << uf << std::endl;
          std::cout << std::setprecision(14) << "polynomial:\n" << Fa_[i] << "\n\n" << Fb_[i] << "\n\n" << Ff0_[i] << "\n\n" << Fy0_[i] << std::endl;
        }
      } // end safeguarded Newton's

      if(converged && eps_*teval<ret){
        ret = eps_*teval;
        whichDim = i;
      }
    }
    if(whichDim==-1){
      return(rootInfo(ret,5,whichDim));
    } else {
    return(rootInfo(ret,5,whichDim,Fgrad_[whichDim]));
    }
  }

public:

  double absTol_;
  double relTol_;
  double eps_; // integrator step size
  double stepErr_;
  double t_left_,t_right_; // time on adaptive mesh

  Eigen::VectorXd genIntStep_;
  Eigen::VectorXd diagInt_;


  RKBS32() : absTol_(1.0e-3), relTol_(1.0e-3), eps_(0.5) {}
  inline double errorOrderHigh() const {return(3.0);}
  inline int odeOrder() const {return 1;}
  inline bool hasEventRootSolver(){return true;}
  inline odeState firstState() const {return(odeState(ys_.col(0)));}
  inline double firstState(const size_t dimension){return ys_.coeff(dimension,0);}
  inline Eigen::VectorXd firstGenerated(){return generated_.col(0);}
  inline Eigen::VectorXd lastGenerated(){return generated_.col(3);}

  void dumpYs(){std::cout << "ys : \n" << ys_ << std::endl;}
  void dumpStep(){
    std::cout << "dump of RKBS32 step" << std::endl;
    dumpYs();
    std::cout << "forces : \n" << force_ << std::endl;
    if(dimGenerated_>0){
      std::cout << "generated, dimGenerated = " << dimGenerated_ << std::endl << generated_ << std::endl;
    }
  }

  inline odeState lastState() const {return(odeState(ys_.col(3)));}

  inline rootInfo eventRootSolver(const rootInfo& oldRoot){
    //std::cout << "old root: " << oldRoot << std::endl;
    rootInfo ret = linRootSolver(oldRoot);
    ret.earliest(splinRootSolver(oldRoot));
    ret.earliest(nonlinRootSolver());
    ret.earliest(splinL1RootSolver(oldRoot));
    ret.earliest(splinL2RootSolver(oldRoot));
    ret.earliest(splinFRootSolver(oldRoot));
    return(ret);
  }



  inline void setup(_ode_type_ &ode){
    ode_ = &ode;
    dim_ = (*ode_).dim();
    dimGenerated_ = (*ode_).generatedDim();
    dimEvent_ = (*ode_).eventRootDim();

    force_.resize(dim_,4);
    force_tmp_.resize(dim_);
    ys_.resize(dim_,4);
    y_tmp_.resize(dim_);

    if(dimGenerated_>0){
      generated_.resize(dimGenerated_,4);
      gen_tmp_.resize(dimGenerated_);
      genIntStep_.resize(dimGenerated_);
    }

    events_.resize(dimEvent_,4);
    events_.setZero();

  }


  bool setInitialState(const odeState &y0){
    if(y0.y.size() != dim_){
      std::cout << "RKBS32::setInitialState : dimension mismatch" << std::endl;
      return(false);
    }



    t_left_ = 0.0;
    ys_.col(0) = y0.y;

//std::cout << "before eval" << std::endl;
    // first evaluation
    (*ode_).ode(t_left_,
     ys_.col(0),force_tmp_,gen_tmp_,diag_tmp_);

//    std::cout << "eval done " << dimEvent_ << std::endl;

    force_.col(0) = force_tmp_;
    if(dimGenerated_>0) generated_.col(0) = gen_tmp_;
    if(diag_tmp_.size()!= diag_.rows()) diag_.resize(diag_tmp_.size(),4);
    if(diag_tmp_.size()>0) diag_.col(0) = diag_tmp_;

    events_.col(0) = (*ode_).eventRoot(0.0,odeState(ys_.col(0)),force_.col(0),true);

 //   std::cout << "eventRoot done " << events_.col(0) << std::endl;

    //dumpStep();
    return(force_.col(0).array().isFinite().all());
  }

  bool step(){
    //std::cout << "step" << std::endl;
    ys_.col(1) = ys_.col(0) +
      (eps_*0.5)*force_.col(0);

    (*ode_).ode(t_left_+0.5*eps_,
     ys_.col(1),force_tmp_,gen_tmp_,diag_tmp_);
    force_.col(1) = force_tmp_;
    if(dimGenerated_>0) generated_.col(1) = gen_tmp_;
    if(diag_tmp_.size()>0) diag_.col(1) = diag_tmp_;

    tmpState_.y = ys_.col(1);
    events_.col(1) = (*ode_).eventRoot(t_left_+0.5*eps_,
                tmpState_,force_.col(1),true);

    if(! force_.col(1).array().isFinite().all()) return(false);

    ys_.col(2) = ys_.col(0) +
      (eps_*0.75)*force_.col(1);
    (*ode_).ode(t_left_ + 0.75*eps_,
     ys_.col(2),force_tmp_,gen_tmp_,diag_tmp_);
    force_.col(2) = force_tmp_;
    if(dimGenerated_>0) generated_.col(2) = gen_tmp_;
    if(diag_tmp_.size()>0) diag_.col(2) = diag_tmp_;
    tmpState_.y = ys_.col(2);
    events_.col(2) = (*ode_).eventRoot(t_left_ + 0.75*eps_,
                tmpState_,force_.col(2),true);
    if(! force_.col(2).array().isFinite().all()) return(false);

    ys_.col(3) = ys_.col(0) +
      (eps_*(2.0/9.0))*force_.col(0) +
      (eps_*(1.0/3.0))*force_.col(1) +
      (eps_*(4.0/9.0))*force_.col(2);

    (*ode_).ode(t_left_ + eps_,
     ys_.col(3),force_tmp_,gen_tmp_,diag_tmp_);
    force_.col(3) = force_tmp_;
    if(dimGenerated_>0) generated_.col(3) = gen_tmp_;
    if(diag_tmp_.size()>0) diag_.col(3) = diag_tmp_;
    tmpState_.y = ys_.col(3);
    events_.col(3) = (*ode_).eventRoot(t_left_ + eps_,
                tmpState_,force_.col(3),true);

    if(! force_.col(3).array().isFinite().all()){ return(false);}



    // low order position
    y1_low_ = ys_.col(0) +
      (eps_*(7.0/24.0))*force_.col(0) +
      (eps_*0.25)*force_.col(1) +
      (eps_*(1.0/3.0))*force_.col(2) +
      (eps_*(1.0/8.0))*force_.col(3);

    t_right_ = t_left_+eps_;


    // step error
    tmpVec_ = (absTol_ + relTol_*ys_.col(0).array().abs().max(ys_.col(3).array().abs())).array();
    tmpVec_ = ((ys_.col(3)-y1_low_).array().abs()/tmpVec_.array());
    stepErr_ = tmpVec_.maxCoeff();

    //std::cout << "stepErr: " << stepErr_ << std::endl;

    // integrated generated quantities
    if(dimGenerated_>0){
      genIntStep_ = (eps_*(2.0/9.0))*generated_.col(0) +
        (eps_*(1.0/3.0))*generated_.col(1) +
        (eps_*(4.0/9.0))*generated_.col(2);
    }


    if(diag_.rows()>0){
      diagInt_ = (eps_*(2.0/9.0))*diag_.col(0) +
        (eps_*(1.0/3.0))*diag_.col(1) +
        (eps_*(4.0/9.0))*diag_.col(2);
    } else {
      if(diagInt_.size()>0) diagInt_.resize(0);
    }

    if(dimEvent_>0){
      eventRootIntR_ = (2.0/9.0)*events_.col(0) +
        (1.0/3.0)*events_.col(1) +
        (4.0/9.0)*events_.col(2);

      eventRootInt_ = eps_*eventRootIntR_;
      /*
       eventRootIntLow_ = (eps_*(7.0/24.0))*events_.col(0) +
       (eps_*0.25)*events_.col(1) +
       (eps_*(1.0/3.0))*events_.col(2) +
       (eps_*(1.0/8.0))*events_.col(3);

       tmpVecRoot_ = (absTol_ + relTol_*eventRootInt_.array().square()).matrix();
       tmpVecRoot_ = ((eventRootInt_-eventRootIntLow_).array().abs()/tmpVecRoot_.array()).matrix();

       double stepErrRoot = tmpVecRoot_.maxCoeff();

       if(stepErr_ < stepErrRoot){
       //std::cout << "integration error dominated by root equation" << std::endl;
       stepErr_ = stepErrRoot;
       }
       */
    }

    return(true);
  }



  void prepareNext(){
    ys_.col(0) = ys_.col(3);
    force_.col(0) = force_.col(3);
    if(dimGenerated_>0) generated_.col(0) = generated_.col(3);
    if(diag_.rows()>0) diag_.col(0) = diag_.col(3);
    events_.col(0) = events_.col(3);
    t_left_ = t_right_;
  }


  inline double denseState(const int which,
                           const double t) const {
    Eigen::VectorXd intPoly = calcIntPoly(t);
    return(ys_.coeff(which,0)*intPoly.coeff(0) +
           ys_.coeff(which,3)*intPoly.coeff(1) +
           force_.coeff(which,0)*intPoly.coeff(2) +
           force_.coeff(which,3)*intPoly.coeff(3));
  }

  Eigen::VectorXd denseState(const double t) const {
    Eigen::VectorXd intPoly = calcIntPoly(t);
    return(ys_.col(0)*intPoly.coeff(0) +
           ys_.col(3)*intPoly.coeff(1) +
           force_.col(0)*intPoly.coeff(2) +
           force_.col(3)*intPoly.coeff(3));
  }

  inline void denseState(const Eigen::VectorXi &which,
                         const double t,
                         Eigen::Ref<Eigen::VectorXd> out) const {
    if(which.size()!=out.size()){
      std::cout << "Error in RKBS32::denseState" << std::endl;
      throw(1);
    }
    Eigen::VectorXd intPoly = calcIntPoly(t);
    for(size_t i=0;i<which.size();i++){
      out.coeffRef(i) = ys_.coeff(which.coeff(i),0)*intPoly.coeff(0) +
        ys_.coeff(which.coeff(i),3)*intPoly.coeff(1) +
        force_.coeff(which.coeff(i),0)*intPoly.coeff(2) +
        force_.coeff(which.coeff(i),3)*intPoly.coeff(3);
    }
  }


  inline void denseGenerated_Int(const double t,
                                 Eigen::Ref<Eigen::VectorXd> out) const {
    Eigen::VectorXd intPoly = calcIntPoly(t);
    out = genIntStep_*intPoly.coeff(1) +
      generated_.col(0)*intPoly.coeff(2) +
      generated_.col(3)*intPoly.coeff(3);
  }

  inline Eigen::VectorXd denseGenerated_Int(const double t) const {
    Eigen::VectorXd intPoly = calcIntPoly(t);
    return(genIntStep_*intPoly.coeff(1) +
           generated_.col(0)*intPoly.coeff(2) +
           generated_.col(3)*intPoly.coeff(3));
  }

  inline void denseGenerated_Level(const double t,
                                   Eigen::Ref<Eigen::VectorXd> out) const {
    Eigen::VectorXd levelPoly = calcLevelPoly(t);
    out = genIntStep_*levelPoly.coeff(1)+
      generated_.col(0)*levelPoly.coeff(2) +
      generated_.col(3)*levelPoly.coeff(3);
  }

  inline bool event(const rootInfo& rootOut){
    int whichEvent = rootOut.rootDim_;
    double eventTime = rootOut.rootTime_;



    //std::cout << "RKBS32::event : ode time : " << t_left_ + rootOut.rootTime_ << " \n" << rootOut << std::endl;

    // dense state before event
    Eigen::VectorXd intPoly = calcIntPoly(eventTime);
    tmpState_.y = ys_.col(0)*intPoly.coeff(0) +
      ys_.col(3)*intPoly.coeff(1) +
      force_.col(0)*intPoly.coeff(2) +
      force_.col(3)*intPoly.coeff(3);

    Eigen::VectorXd levelPoly = calcLevelPoly(eventTime);
    force_tmp_ = ys_.col(0)*levelPoly.coeff(0) +
      ys_.col(3)*levelPoly.coeff(1) +
      force_.col(0)*levelPoly.coeff(2) +
      force_.col(3)*levelPoly.coeff(3);

    // evaluate eventRoot before event is done
    if(rootOut.rootType_==0){
      event_tmp_ = (*ode_).eventRoot(
        t_left_+eventTime,
        tmpState_,
        force_tmp_,false);
      if(std::fabs(event_tmp_(whichEvent))>200.0*absTol_){
        std::cout << "eventRoot at interpolated state: " << event_tmp_(whichEvent) << std::endl;
        std::cout << "whichEvent : " << whichEvent << std::endl;
      }
    }
    // evaluate the new state after event occurred
    bool eventContinue = (*ode_).event(
      rootOut, // which event
      t_left_+eventTime, // time of event
      tmpState_, // state at event
      force_tmp_, // force at event
      newState_);

    // evaluate ode and root fun after event occurred to prepare for subsequent
    // step
    ys_.col(3) = newState_.y;
    t_right_ = t_left_+eventTime;
    (*ode_).ode(t_right_,
     ys_.col(3),force_tmp_,gen_tmp_,diag_tmp_);
    force_.col(3) = force_tmp_;

    if(dimGenerated_>0) generated_.col(3) = gen_tmp_;
    if(diag_.rows() != diag_tmp_.size()) diag_.resize(diag_tmp_.size(),4);
    if(diag_tmp_.size()>0) diag_.col(3) = diag_tmp_;
    events_.col(3) = (*ode_).eventRoot(
      t_right_,
      newState_,
      force_.col(3),true);

    // set the active eventRoot at event artificially to exactly zero
    // if the event root equation did not change
    // to avoid repeating the event due to numerical inaccuracies
    //std::cout << "rk::event new root eval  : \n" << event_tmp_ << std::endl;
    //std::cout << "rk::event new root eval  : \n" << events_ << std::endl;
    //std::cout << rootOut << std::endl;
    if(rootOut.rootType_==0 && std::fabs(event_tmp_(whichEvent)-events_(whichEvent,3))<1.0e-14){
      events_(whichEvent,3) = 0.0;
    }

    if(! force_.col(3).array().isFinite().all()){
      eventContinue = false;
      std::cout << "Post event Numerical problems" << std::endl;
    }
    return(eventContinue);
  }
  /*
   inline bool manualEvent(const rootInfo& rootOut,
   const odeState& newState){
   bool eventContinue = true;
   ys_.col(3) = newState.y;
   t_right_ = t_left_+rootOut.rootTime_;
   (*ode_).ode(t_right_,
   ys_.col(3),force_tmp_,gen_tmp_,diag_tmp_);
   force_.col(3) = force_tmp_;

   if(dimGenerated_>0) generated_.col(3) = gen_tmp_;
   if(diag_.rows() != diag_tmp_.size()) diag_.resize(diag_tmp_.size(),4);
   if(diag_tmp_.size()>0) diag_.col(3) = diag_tmp_;
   events_.col(3) = (*ode_).eventRoot(
   t_right_,
   newState_,
   force_.col(3),true);

   // set the active eventRoot at event artificially to exactly zero
   // if the event root equation did not change
   // to avoid repeating the event due to numerical inaccuracies
   if(rootOut.rootDim_==0){
   events_(rootOut.rootDim_,3) = 0.0;
   }

   if(! force_.col(3).array().isFinite().all()){
   eventContinue = false;
   std::cout << "Post event Numerical problems" << std::endl;
   }
   return(eventContinue);

   }
   */
  inline Eigen::VectorXd firstEventRoot() const {return events_.col(0); }

  inline double denseEventRoot_Level(const int which,
                                     const double t){
    std::cout << "denseEventRoot_* should not be called" << std::endl;
    throw(567);
    return(-1.0);
  }

  inline double denseEventRoot_LevelDot(const int which,
                                        const double t){
    std::cout << "denseEventRoot_* should not be called" << std::endl;
    throw(567);
    return(0.0);
  }


};



#endif
