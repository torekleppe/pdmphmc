#include <iostream>
#include <vector>
#include <Eigen/Dense>

/*
 * General sparse Cholesky routines based on the ldl library
 * (T. A. Davis (2005), ‘Algorithm 849: A concise sparse Cholesky factorization
 * package’, ACM Trans. Math. Softw. 31(4), 587–591.).
 * A lightly edited version (mainly to allow for AD types)
 * of ldl is included here.
 *
 */


class sparseCholSymbolic{
private:

  void ldl_symbolic
  (
      int n,		/* A and L are n-by-n, where n >= 0 */
int Ap [ ],		/* input of size n+1, not modified */
int Ai [ ],		/* input of size nz=Ap[n], not modified */
int Lp [ ],		/* output of size n+1, not defined on input */
int Parent [ ],	/* output of size n, not defined on input */
int Lnz [ ],	/* output of size n, not defined on input */
int Flag [ ]	/* workspace of size n, not defn. on input or output */
  )
  {
    int i, k, p, kk, p2 ;

    for (k = 0 ; k < n ; k++)
    {
      /* L(k,:) pattern: all nodes reachable in etree from nz in A(0:k-1,k) */
      Parent [k] = -1 ;	    /* parent of k is not yet known */
      Flag [k] = k ;		    /* mark node k as visited */
      Lnz [k] = 0 ;		    /* count of nonzeros in column k of L */
      kk = k ;  /* kth original, or permuted, column */
      p2 = Ap [kk+1] ;
      for (p = Ap [kk] ; p < p2 ; p++)
      {
        /* A (i,k) is nonzero (original or permuted A) */
        i = Ai [p];
        if (i < k)
        {
          /* follow path from i to root of etree, stop at flagged node */
          for ( ; Flag [i] != k ; i = Parent [i])
          {
            /* find parent of i if not yet determined */
            if (Parent [i] == -1) Parent [i] = k ;
            Lnz [i]++ ;				/* L (k,i) is nonzero */
            Flag [i] = k ;			/* mark i as visited */
          }
        }
      }
    }
    /* construct Lp index array from Lnz column counts */
    Lp [0] = 0 ;
    for (k = 0 ; k < n ; k++)
    {
      Lp [k+1] = Lp [k] + Lnz [k] ;
    }
  }



  void findCompression(){

    if(Ap_.size() != n_+1) Ap_.resize(n_+1);
    if(Ai_.size() != nz_) Ai_.resize(nz_);
    if(map_.size() != nz_) map_.resize(nz_);
    Eigen::VectorXi counts(n_);
    int ii;
    // count elements in each column
    Ap_.setZero();
    for(int k=0;k<nz_;k++) Ap_.coeffRef(j_.coeff(k)+1)++;
    // cumulative sum
    for(int k=1;k<=n_;k++) Ap_.coeffRef(k) = Ap_.coeff(k-1) + Ap_.coeff(k);
    // Ap_ done
    counts.setZero();

    for(int k=0;k<nz_;k++){
      ii = Ap_.coeff(j_.coeff(k)) + counts.coeff(j_.coeff(k));
      Ai_.coeffRef(ii) = i_.coeff(k);
      map_.coeffRef(ii) = k;
      counts.coeffRef(j_.coeff(k))++;
    }


  }

public:

  int n_;
  int nz_;

  Eigen::VectorXi i_;
  Eigen::VectorXi j_;
  Eigen::VectorXi Ap_;
  Eigen::VectorXi Ai_;
  Eigen::VectorXi map_;
  Eigen::VectorXi Lp_;
  Eigen::VectorXi Parent_;
  Eigen::VectorXi Lnz_;
  Eigen::VectorXi Flag_;


  sparseCholSymbolic() {}
  // no permutation constructor version
  sparseCholSymbolic(const int n, // dimension of matrix
                     const Eigen::VectorXi &i,  // row index
                     const Eigen::VectorXi &j // column index
  ) : n_(n),nz_(i.size()),
  i_(i),j_(j) {
    // initial checks
    if(i.size() != j.size()){
      std::cout << "index vectors not of same length, exiting" << std::endl;
      return;
    }
    if(i.maxCoeff()>n-1 || i.minCoeff()<0){
      std::cout << "row index i out of range, exiting" << std::endl;
      return;
    }
    if(j.maxCoeff()>n-1 || j.minCoeff()<0){
      std::cout << "column index i out of range, exiting" << std::endl;
      return;
    }


    // check that all entries are in the upper triangle
    int tmp;
    for(int k=0;k<nz_;k++){
      if(i_.coeff(k)>j_.coeff(k)){
        std::cout << "triplet entry (i,j)=(" << i_.coeff(k) << "," << j_.coeff(k) << ") in lower triangle!!!" << std::endl;
        tmp = i_.coeff(k);
        i_.coeffRef(k) = j_.coeff(k);
        j_.coeffRef(k) = tmp;
      }
    }

    // find compressed column format

    findCompression();


    Lp_.resize(n_+1);
    Parent_.resize(n_);
    Lnz_.resize(n_);
    Flag_.resize(n_);

    ldl_symbolic(n,
                 Ap_.data(),
                 Ai_.data(),
                 Lp_.data(),
                 Parent_.data(),
                 Lnz_.data(),
                 Flag_.data());

  }

  void dumpCompessedPattern() const {
    Eigen::MatrixXi dense(n_,n_);
    dense.setZero();
    for(int j=0;j<n_;j++){
      for(int i=Ap_(j);i<Ap_(j+1);i++) dense(Ai_(i),j)++;
    }
    std::cout << "compressed pattern of " << n_ << " x " << n_ << " matrix, nz_ = " << nz_ << std::endl;
    std::cout << dense << std::endl;
  }


  void dumpTripletPattern() const {
    Eigen::MatrixXi dense(n_,n_);
    dense.setZero();
    for(int k=0;k<nz_;k++){
      dense(i_(k),j_(k))++;
    }
    std::cout << "triplet pattern of " << n_ << " x " << n_ << " matrix, nz_ = " << nz_ << std::endl;
    std::cout << dense << std::endl;
  }
  template <class T>
  void showAx(const Eigen::Matrix<T,Eigen::Dynamic,1> &Ax) const {
    Eigen::MatrixXd dense (n_,n_);
    dense.setZero();
    for(int j=0;j<n_;j++){
      for(int i=Ap_(j);i<Ap_(j+1);i++) dense(Ai_(i),j) = doubleValue(Ax(map_(i)));
    }
    std::cout << "Numerical values in Ax presented as dense matrix" << std::endl;
    std::cout << dense << std::endl;
  }
};


template <class var>
class sparseCholNumeric{
  template <class T>
  int ldl_numeric		/* returns n if successful, k if D (k,k) is zero */
    (
    const int n,		/* A and L are n-by-n, where n >= 0 */
    const int Ap [ ],		/* input of size n+1, not modified */
    const int Ai [ ],		/* input of size nz=Ap[n], not modified */
    const T Ax [ ],	/* input of size nz=Ap[n], not modified */
    const int xmap [],
    const int Lp [ ],		/* input of size n+1, not modified */
    const int Parent [ ],	/* input of size n, not modified */
    int Lnz [ ],	/* output of size n, not defn. on input */
    int Li [ ],		/* output of size lnz=Lp[n], not defined on input */
    T Lx [ ],	/* output of size lnz=Lp[n], not defined on input */
    T D [ ],	/* output of size n, not defined on input */
    T Y [ ],	/* workspace of size n, not defn. on input or output */
    int Pattern [ ],	/* workspace of size n, not defn. on input or output */
    int Flag [ ]	/* workspace of size n, not defn. on input or output */
    )
  {

    T yi, l_ki ;
    int i, k, p, kk, p2, len, top ;
    for (k = 0 ; k < n ; k++)
    {
      /* compute nonzero Pattern of kth row of L, in topological order */
      Y [k] = 0.0 ;		    /* Y(0:k) is now all zero */
      top = n ;		    /* stack for pattern is empty */
      Flag [k] = k ;		    /* mark node k as visited */
      Lnz [k] = 0 ;		    /* count of nonzeros in column k of L */
      kk = k;  /* kth original, or permuted, column */
      p2 = Ap [kk+1] ;
      for (p = Ap [kk] ; p < p2 ; p++)
      {
        i = Ai [p] ;	/* get A(i,k) */
      if (i <= k)
      {
        Y [i] += Ax [xmap[p]] ;  /* scatter A(i,k) into Y (sum duplicates) */
      for (len = 0 ; Flag [i] != k ; i = Parent [i])
      {
        Pattern [len++] = i ;   /* L(k,i) is nonzero */
      Flag [i] = k ;	    /* mark i as visited */
      }
      while (len > 0) Pattern [--top] = Pattern [--len] ;
      }
      }
      /* compute numerical values kth row of L (a sparse triangular solve) */
      D [k] = Y [k] ;		    /* get D(k,k) and clear Y(k) */
      Y [k] = 0.0 ;
      for ( ; top < n ; top++)
      {
        i = Pattern [top] ;	    /* Pattern [top:n-1] is pattern of L(:,k) */
      yi = Y [i] ;	    /* get and clear Y(i) */
      Y [i] = 0.0 ;
      p2 = Lp [i] + Lnz [i] ;
      for (p = Lp [i] ; p < p2 ; p++)
      {
        Y [Li [p]] -= Lx [p] * yi ;
      }
      l_ki = yi / D [i] ;	    /* the nonzero entry L(k,i) */
      D [k] -= l_ki * yi ;
      Li [p] = k ;	    /* store L(k,i) in column form of L */
      Lx [p] = l_ki ;
      Lnz [i]++ ;		    /* increment count of nonzeros in col i */
      }
      if (D [k] == 0.0) return (k) ;	    /* failure, D(k,k) is zero */
    }
    return (n) ;	/* success, diagonal of D is all nonzero */
  }

  /* ========================================================================== */
  /* === ldl_lsolve:  solve Lx=b ============================================== */
  /* ========================================================================== */
  template <class T, class T2>
  void ldl_lsolve
    (
        const int n,        /* L is n-by-n, where n >= 0 */
  T X [ ],    /* size n.  right-hand-side on input, soln. on output */
  const int Lp [ ],        /* input of size n+1, not modified */
  const int Li [ ],        /* input of size lnz=Lp[n], not modified */
  const T2 Lx [ ]    /* input of size lnz=Lp[n], not modified */
    ) const
  {
    int j, p, p2 ;
    for (j = 0 ; j < n ; j++)
    {
      p2 = Lp [j+1] ;
      for (p = Lp [j] ; p < p2 ; p++)
      {
        X [Li [p]] -= Lx [p] * X [j] ;
      }
    }
  }

  /* ========================================================================== */
  /* === ldl_ltsolve: solve L'x=b  ============================================ */
  /* ========================================================================== */
  template <class T, class T2>
  void ldl_ltsolve
    (
        const int n,        /* L is n-by-n, where n >= 0 */
  T X [ ],    /* size n.  right-hand-side on input, soln. on output */
  const int Lp [ ],        /* input of size n+1, not modified */
  const int Li [ ],        /* input of size lnz=Lp[n], not modified */
  const T2 Lx [ ]    /* input of size lnz=Lp[n], not modified */
    ) const
  {
    int j, p, p2 ;
    for (j = n-1 ; j >= 0 ; j--)
    {
      p2 = Lp [j+1] ;
      for (p = Lp [j] ; p < p2 ; p++)
      {
        X [j] -= Lx [p] * X [Li [p]] ;
      }
    }
  }


  int n_;
  int lnz_;
  Eigen::VectorXi Lp_;
  Eigen::VectorXi Lnz_;

  Eigen::Matrix<var,Eigen::Dynamic,1> Y_;
  Eigen::VectorXi Pattern_;
  Eigen::VectorXi Flag_;

  int exitFlag_;

public:

  Eigen::VectorXi Li_;
  Eigen::Matrix<var,Eigen::Dynamic,1> Lx_;
  Eigen::Matrix<var,Eigen::Dynamic,1> D_;


  sparseCholNumeric(){}
  void setup(const sparseCholSymbolic &symbolic){
    n_ = symbolic.n_;
    Lp_ = symbolic.Lp_; //copy

    lnz_ = Lp_.coeff(n_);
    Lnz_.resize(n_);
    Li_.resize(lnz_);
    Lx_.resize(lnz_);
    D_.resize(n_);
    Y_.resize(n_);
    Pattern_.resize(n_);
    Flag_.resize(n_);
  }

  bool chol(const sparseCholSymbolic &symbolic,
            const Eigen::Matrix<var,Eigen::Dynamic,1> &Ax){
    exitFlag_ = ldl_numeric(symbolic.n_,
                            symbolic.Ap_.data(),
                            symbolic.Ai_.data(),
                            Ax.data(),
                            symbolic.map_.data(),
                            symbolic.Lp_.data(),
                            symbolic.Parent_.data(),
                            Lnz_.data(),
                            Li_.data(),
                            Lx_.data(),
                            D_.data(),
                            Y_.data(),
                            Pattern_.data(),
                            Flag_.data()
    );
    if(exitFlag_<n_){
      std::cout << "Warning: Matrix not SPD in sparseCholNumeric" << std::endl;
      return(false);
    }
    return(true);
  }


  sparseCholNumeric(const Eigen::Matrix<var,Eigen::Dynamic,1> &Ax,
                    const sparseCholSymbolic &symbolic) {
    n_ = symbolic.n_;
    Lp_ = symbolic.Lp_; //copy

    lnz_ = Lp_.coeff(n_);
    Lnz_.resize(n_);
    Li_.resize(lnz_);
    Lx_.resize(lnz_);
    D_.resize(n_);
    Y_.resize(n_);
    Pattern_.resize(n_);
    Flag_.resize(n_);


    exitFlag_ = ldl_numeric(symbolic.n_,
                            symbolic.Ap_.data(),
                            symbolic.Ai_.data(),
                            Ax.data(),
                            symbolic.map_.data(),
                            symbolic.Lp_.data(),
                            symbolic.Parent_.data(),
                            Lnz_.data(),
                            Li_.data(),
                            Lx_.data(),
                            D_.data(),
                            Y_.data(),
                            Pattern_.data(),
                            Flag_.data()
                            );
    if(exitFlag_<n_){
      std::cout << "Warning: Matrix not SPD in sparseCholNumeric" << std::endl;
    }
  }

  sparseCholNumeric(int n, // dimension of matrix
                    const Eigen::VectorXi &i,  // row index
                    const Eigen::VectorXi &j, // column index
                    const Eigen::Matrix<var,Eigen::Dynamic,1> &Ax // numerical values
  ) : sparseCholNumeric(Ax,sparseCholSymbolic(n,i,j)) {}


  /*
   *
   * Computes x=L*b
   *
   */



  template <class T__>
  void applyL(const Eigen::Matrix<T__,Eigen::Dynamic,1> &b,
                Eigen::Matrix<typename boost::math::tools::promote_args<var,T__>::type,Eigen::Dynamic,1> &x) const {

    x.setZero();
    typename boost::math::tools::promote_args<var,T__>::type sqrtDb;
    // L is in column oriented format

    for(int j=0; j<n_;j++){ // loop over columns
      sqrtDb = b.coeff(j)*sqrt(D_.coeff(j));
      x.coeffRef(j) += sqrtDb;
      for(int ii=Lp_.coeff(j);ii<Lp_.coeff(j+1);ii++){
        //std::cout << "(i,j) = " << Li_.coeff(ii) << " " << j << std::endl;
        x.coeffRef(Li_.coeff(ii)) += Lx_.coeff(ii)*sqrtDb;
      }
    }

  }


  /*
   *  Solves L^T x = b for x where LL^T = A
   *
   */
  template <class T__>
  void LT_solve(const Eigen::Matrix<T__,Eigen::Dynamic,1> &b,
                Eigen::Matrix<typename boost::math::tools::promote_args<var,T__>::type,Eigen::Dynamic,1> &x) const {
    for(int i=0;i<n_;i++) x.coeffRef(i) = b.coeff(i)/sqrt(D_.coeff(i));
    ldl_ltsolve< typename boost::math::tools::promote_args<var,T__>::type , var  >(n_,x.data(),Lp_.data(),Li_.data(),Lx_.data());
  }

  template <class T__>
  Eigen::Matrix<typename boost::math::tools::promote_args<var,T__>::type,Eigen::Dynamic,1>
  LT_solve(const Eigen::Matrix<T__,Eigen::Dynamic,1> &b) const {
    Eigen::Matrix<typename boost::math::tools::promote_args<var,T__>::type,Eigen::Dynamic,1> ret(n_);
    LT_solve(b,ret);
    return(ret);
  }
  /*
   *  Solves L x = b for x where LL^T = A
   *
   */
  template <class T__>
  void L_solve(const Eigen::Matrix<T__,Eigen::Dynamic,1> &b,
               Eigen::Matrix<typename boost::math::tools::promote_args<var,T__>::type,Eigen::Dynamic,1> &x) const {
    x = b;
    ldl_lsolve< typename boost::math::tools::promote_args<var,T__>::type , var >(n_,x.data(),Lp_.data(),Li_.data(),Lx_.data());
    for(int i=0;i<n_;i++) x.coeffRef(i) /= sqrt(D_.coeff(i));
  }

  template <class T__>
  Eigen::Matrix<typename boost::math::tools::promote_args<var,T__>::type,Eigen::Dynamic,1>
  L_solve(const Eigen::Matrix<T__,Eigen::Dynamic,1> &b) const {
    Eigen::Matrix<typename boost::math::tools::promote_args<var,T__>::type,Eigen::Dynamic,1> ret(n_);
    L_solve(b,ret);
    return(ret);
  }

  /*
   *  Solves A x = b for x where LL^T = A
   *
   */
  template <class T__>
  void solve(const Eigen::Matrix<T__,Eigen::Dynamic,1> &b,
             Eigen::Matrix<typename boost::math::tools::promote_args<var,T__>::type,Eigen::Dynamic,1> &x) const {
    x = b;
    ldl_lsolve< typename boost::math::tools::promote_args<var,T__>::type , var >(n_,x.data(),Lp_.data(),Li_.data(),Lx_.data());
    for(int i=0;i<n_;i++) x.coeffRef(i) /= D_.coeff(i);
    ldl_ltsolve< typename boost::math::tools::promote_args<var,T__>::type , var  >(n_,x.data(),Lp_.data(),Li_.data(),Lx_.data());
  }
  template <class T__>
  Eigen::Matrix<typename boost::math::tools::promote_args<var,T__>::type,Eigen::Dynamic,1>
  solve(const Eigen::Matrix<T__,Eigen::Dynamic,1> &b) const {
    Eigen::Matrix<typename boost::math::tools::promote_args<var,T__>::type,Eigen::Dynamic,1> ret(n_);
    solve(b,ret);
    return(ret);
  }




  var logDetA() const {
    var ret = 0.0;
    for(int i=0;i<n_;i++) ret+=log(D_.coeff(i));
    return(ret);
  }
  var logDetL() const {return 0.5*logDetA();}

  int exitFlag() const {return exitFlag_;}

  void dumpLDL() const {
    Eigen::MatrixXd denseL (n_,n_);
    denseL.setZero();
    for(int j = 0;j<n_;j++) denseL(j,j) = 1.0;
    for(int j = 0;j<n_;j++){
      for(int i=Lp_(j);i<Lp_(j+1);i++) denseL(Li_(i),j) = doubleValue(Lx_(i));
    }
    std::cout << "LDL factorization of " << n_ << " x " << n_ << " matrix, lnz = " << lnz_ << std::endl;
    std::cout << "L = " << std::endl;
    std::cout << denseL << std::endl;

    Eigen::MatrixXd denseD (n_,n_);
    denseD.setZero();
    for(int j=0;j<n_;j++) denseD(j,j) = doubleValue(D_(j));
    std::cout << "D = " << std::endl;
    std::cout << denseD << std::endl;

    std::cout << "LDL^T = " << std::endl;
    std::cout << denseL*denseD*denseL.transpose() << std::endl;
  }

  void tripletL(std::vector<int>& ii,std::vector<int>& jj) const {
    ii.empty();
    jj.empty();
    for(int j = 0;j<n_;j++){
      ii.push_back(j);
      jj.push_back(j);
    }
    for(int j = 0;j<n_;j++){
      for(int i=Lp_(j);i<Lp_(j+1);i++){
        ii.push_back(Li_(i));
        jj.push_back(j);
      }
    }
  }




};


