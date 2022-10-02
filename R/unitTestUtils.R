#utilities for unit testing


check.mean<-function(fit.obj,par.name,par.mean,nstds=10.0){
  m <- as.matrix(getMonitor(fit.obj,print=FALSE)[[1]])
  obs <- (m[par.name,"mean"]-par.mean)/m[par.name,"se_mean"]
  if(abs(obs)>nstds) stop(paste0("bad posterior mean : ",par.name))
  return(0)
}
