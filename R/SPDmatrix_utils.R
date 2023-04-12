#' Transform a SPD matrix to the vector representation SPDmatrix
#'
#'
#' @param mat a SPD matrix to be transformed
SPDmatrix.asvector <- function(mat){
  n <- dim(mat)[1]
  L <- t(chol(mat))
  D <- diag(L)
  L <- L%*%diag(1.0/D)
  v <- 2.0*log(D)
  for(j in 1:(n-1)){ v <- c(v,L[(j+1):n,j])}
  return(unname(v))
}

#' Transform a SPDmatrix representation vector to a SPD matrix
#'
#'
#' @param vec a SPDmatrix vector representation to be transformed to a SPD matrix.
SPDmatrix.asmatrix <- function(vec){
  n <- round(0.5*(-1+sqrt(1+8*length(vec))))
  D <- diag(exp(vec[1:n]))
  L <- diag(n)
  k <- n+1
  for(j in 1:(n-1)){
    for(i in (j+1):n){
      L[i,j] <- vec[k]
      k <- k+1
    }
  }
  return(L%*%D%*%t(L))
}
