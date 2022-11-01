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
#' @param amt should amtVar or regular AD types be used
#' @param metric.tensor.type either sparse or dense storage
#' @param work.folder the folder used for storing files, created if not already existing
#' @param compiler.info information related to compiler
#' @param include additional flags passed to the compiler. E.g. \code{"-D __DEBUG__"} for integrator related debug info or \code{"-D __TENSOR_DEBUG__"} for Riemannian tensor debugging info.
#' @param clean should output files be removed once read into R?
#'
build <- function(model.file,
                  model.class.name="model",
                  process.type=c("HMCProcess","RMHMCProcess"),
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

  if(process.type=="RMHMCProcess"){
    metric.tensor.type <- match.arg(metric.tensor.type)
    tensorType <- paste0("metricTensor",metric.tensor.type)
  } else {
    tensorType <- "metricTensorDummy"
  }
  header <- paste0(header," typedef ",tensorType," metricTensorType__;\n")
  message("metric tensor type : ",tensorType)


  if(amt){
    header <- paste0(header," typedef amt::amtVar varType;\n")
  } else {
    header <- paste0(header," typedef stan::math::var varType;\n")
  }

  # note macro FILE_NAME_BASE__ must have appropriate slashs for both unix and windows
  header <- paste0(header," #define FILE_NAME_BASE__ \"",
    normalizePath(out@file.name.base,winslash="\\",mustWork=FALSE),
    "\" \n")

  cat(header,file=normalizePath(paste0(work.folder,"/___model_typedefs.hpp"),mustWork = FALSE),append = FALSE)
  out@compiler.flag <- .compileCpp(out,compiler.info,include)

  if(clean){
    file.remove(normalizePath(paste0(work.folder,"/___model_typedefs.hpp"),mustWork = FALSE))
    file.remove(normalizePath(paste0(work.folder,"/___model_specification.hpp"),mustWork = FALSE))
  }


  return(out)

}





