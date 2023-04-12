.get.build.id <- function(){
  ret <- date()
  ret <- gsub(" ","__",ret,fixed=TRUE)
  ret <- gsub(":","-",ret,fixed=TRUE)
  return(ret)
}


.check.build <- function(build.object){
  return(file.exists(build.object@file.name.base) ||
           file.exists(paste0(build.object@file.name.base,".exe")))
}

#' Build (i.e. parse and compile) a pdmphmc model
#'
#'
#' @param model.file a cpp file containing the model specification class
#' @param model.class.name the name of the model specification class
#' @param process.type which kind of pdmphmc-process to use
#' @param step.type which Runge Kutta method
#' @param TM.type which Transport Map
#' @param amt should amtVar or regular AD types be used
#' @param metric.tensor.type either sparse or dense storage
#' @param work.folder the folder used for storing files, created if not already existing
#' @param compiler.info information related to compiler
#' @param include additional flags passed to the compiler. E.g. \code{"-D __DEBUG__"} for integrator related debug info or \code{"-D __TENSOR_DEBUG__"} for Riemannian tensor debugging info.
#' @param clean should output files be removed once read into R?
#'
build <- function(model.file,
                  model.class.name="model",
                  process.type=c("HMCProcessConstr","HMCProcess","RMHMCProcess"),
                  step.type=c("RKDP54","RKBS32"),
                  TM.type=c("diagLinearTM_VARI","diagLinearTM_ISG","identityTM"),
                  amt=process.type=="RMHMCProcess",
                  metric.tensor.type=c("Sparse","Dense"),
                  work.folder=paste0(getwd(),"/pdmphmc_files/"),
                  compiler.info=.default.compiler.info(),
                  include="",
                  clean=TRUE){

  if(! file.exists(model.file)) stop("bad model.file")



  out <- new("build-output")
  out@model.name <- model.class.name
  out@build.id <- .get.build.id()
  out@work.folder <- work.folder
  out@file.name.base <- normalizePath(paste0(work.folder,model.class.name,"_",out@build.id),
                                      mustWork = FALSE)


  message("model name : ",model.class.name)
  #
  # make work folder, and copy the model specification file into it
  #


  if(! file.exists(normalizePath(work.folder,mustWork = FALSE))){
    dir.create(normalizePath(work.folder,mustWork = FALSE))
  }
  file.copy(model.file,
            normalizePath(paste0(work.folder,"/___model_specification.hpp"),
                          mustWork = FALSE),
            overwrite = TRUE)
  #
  # make the custom header for this project
  #
  header <- paste0("\n typedef ",model.class.name," modelSpec__;\n")
  process.type <- match.arg(process.type)
  header <- paste0(header," #define ProcessType__ ",process.type," \n")
  message("process type : ",process.type)

  step.type <- match.arg(step.type)
  header <- paste0(header," #define RKstepType__ ",step.type," \n")
  message("Runge Kutta step type : ",step.type)

  TM.type <- match.arg(TM.type)
  header <- paste0(header," #define TMType__ ",TM.type," \n")
  message("Transport map type : ",TM.type)

  if(process.type=="RMHMCProcess"){
    metric.tensor.type <- match.arg(metric.tensor.type)
    tensorType <- paste0("metricTensor",metric.tensor.type)
    message("metric tensor type : ",tensorType)

  } else {
    tensorType <- "metricTensorDummy"
  }
  header <- paste0(header," typedef ",tensorType," metricTensorType__;\n")


  if(amt){
    header <- paste0(header," typedef amt::amtVar varType;\n")
  } else {
    header <- paste0(header," typedef stan::math::var varType;\n")
  }

  # note macro FILE_NAME_BASE__ must have appropriate slashs for both unix and windows
  fnb4macro <- normalizePath(out@file.name.base,winslash="/",mustWork=FALSE)
  #if(identical(.Platform$OS.type,"windows")){
#    fnb4macro <- gsub("/","\\\\",fnb4macro)
 #}
  header <- paste0(header," #define FILE_NAME_BASE__ \"",fnb4macro,"\" \n")

  cat(header,file=normalizePath(paste0(work.folder,"/___model_typedefs.hpp"),mustWork = FALSE),append = FALSE)
  out@compiler.flag <- .compileCpp(out,compiler.info,include)

  if(clean){
    file.remove(normalizePath(paste0(work.folder,"/___model_typedefs.hpp"),mustWork = FALSE))
    file.remove(normalizePath(paste0(work.folder,"/___model_specification.hpp"),mustWork = FALSE))
  }


  return(out)

}





