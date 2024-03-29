#' Build and run a simple pdmphmc model
#'
#' @description Build and run a simple model in order to test the installation/available C++ compiler
#'

testSystem<-function(include=""){
  success <- FALSE
  mdl <- build(model.file=system.file('examples/testSystemModel.cpp',
                                      package = "pdmphmc"),
               model.class.name="model",
               work.folder = paste0(getwd(),"/___testSystemWorkFolder___/"),
               include=include,
               step.type = "RKBS32")
  fit <- run(mdl,Tmax=1000.0,
             store.pars = c("x"),
             data=list(dummyDataNotUsed=0L))
  if(max(abs(fit@bin.eflag[[1]]))<0.1){
    message("model ran successfully")
    success <- TRUE
  }
  clean.model(mdl,remove.all = TRUE)
  return(success)
}



