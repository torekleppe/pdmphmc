# pdmphmc
Piecewise deterministic processes using Hamiltonian dynamics



This is the development version of the R-package pdmphmc, which is a re-implementation of the methodology in 

The R-package has currently been tested on mac (using the clang compiler and R 4.x), linux (using native gcc and R 4.x) 
Windows testing is so far not done (R 4.x and using the mingw64 compiler that ships with Rtools4.0). 

In general, it requires that you have a working installation of R>=4.0.0 and **a working installation of the R-package rstan**.

Installation of the package is most easily done using the `devtools` function `install_github()`, e.g.: ```devtools::install_github("https://github.com/torekleppe/pdmphmc")```





