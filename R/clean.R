
#' Remove all files and folders created during build of pdmphmc model
#'
#' @description Remove all files and folders created made using the build()-function. Note: model will not work after this call
#'
#' @param build.object model made using the build()-function.

clean.model <- function(build.object,remove.all=FALSE,dry.run=FALSE){
  wf <- normalizePath(build.object@work.folder)
  if(file.exists(wf)){
    files <- list.files(wf)
    for(i in 1:length(files)){
      if(remove.all || length(grep(build.object@build.id,files[i],fixed=TRUE))){
        full.path <- normalizePath(paste0(build.object@work.folder,"/",files[i]),mustWork = TRUE)
        if(!dry.run){
          file.remove(full.path)
        } else {
          print(full.path)
        }
      }
    }
    files <- list.files(wf)
    if(!dry.run && length(files)==0) file.remove(wf)
  } else {
    message("clean.model: workfolder missing")
  }
}
