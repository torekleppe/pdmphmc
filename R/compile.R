
.default.compiler.info<- function(flags="-O3"){
  if(identical(.Platform$OS.type,"windows")){
    if(pkgbuild::has_rtools()){
      full.path <- strsplit(Sys.getenv("PATH"),split=";")[[1]]
      rtools.paths <- grep("rtools",full.path,ignore.case = TRUE,fixed=TRUE)
      if(length(rtools.paths)==0) stop("rtools not in path!")
      rtools.paths <- full.path[rtools.paths]
      mingw.path <- grep("mingw",rtools.paths,ignore.case = TRUE,fixed=TRUE)
      if(length(mingw.path)==0) stop("cannot find mingw-directory in path!")
      compiler.path <- paste0(rtools.paths[mingw.path],"\\g++.exe")
      #rtools.path <- gsub("/","\\",pkgbuild::rtools_path(),fixed=TRUE) # note, this is not the path to the compiler binary!!
      #print(rtools.path)
      #compiler.path <- paste0(strsplit(rtools.path,"usr")[[1]][1],"mingw64\\bin\\g++.exe")
      #print(compiler.path)
      #if(file.exists(normalizePath(compiler.path,mustWork=FALSE))){
      #  return(list(compiler=normalizePath(compiler.path),flags=flags))
      #}
      #compiler.path <- paste0(strsplit(rtools.path,"usr")[[1]][1],"mingw_64\\bin\\g++.exe")
      print(compiler.path)
      if(file.exists(normalizePath(compiler.path,mustWork=FALSE))){
        return(list(compiler=normalizePath(compiler.path),flags=flags))
      } else {
        stop("unknown rtools directory format")
      }
    } else {
      stop("requires a working c++ compiler, get the rtools package")
    }
  } else if(identical(.Platform$OS.type,"unix")) {
    flags.ext <- paste0(flags," -Wno-unknown-pragmas -Wno-deprecated-declarations")
    return(list(compiler="g++",flags=flags.ext))
  } else {
    stop("Unknown OS.type")
  }

}

.compileCpp <- function(bo,
                        compiler.info=.default.compiler.info(),
                        include=""){

  package.includes <- paste0(
    " -I",normalizePath(system.file('include', package = "StanHeaders")),
    " -I",normalizePath(system.file('include', package = "RcppEigen")),
    " -I",normalizePath(system.file('include', package = "BH")),
    " -I",normalizePath(system.file('include', package = "RcppProgress")),
    " -I",normalizePath(system.file('include', package = "pdmphmc"))
  )

  cpp.file <- normalizePath(system.file("main_template/main_template.cpp",package = "pdmphmc"))

  model.includes <- paste0("-I",bo@work.folder)

  compilerArgs <- paste0(cpp.file,
                         " -o ",bo@file.name.base,
                         " -std=c++17 ",compiler.info$flags," ",
                         model.includes," ",
                         package.includes," ",
                         include)


  ret <- system2(command=compiler.info$compiler,
                 args=compilerArgs,
                 stdout = TRUE,
                 stderr = TRUE)

  eflag <- attr(ret,"status")

  if(is.null(eflag)){
    message("compilation exited successfully")
    return(0L)
  } else {
    message("problem with compilation: ")
    cat(paste0(command=compiler.info$compiler," ",compilerArgs))
    cat(ret,file=normalizePath(paste0(bo@file.name.base,"_compiler_out.txt"),mustWork = FALSE))
    print(ret,quote = FALSE)
    return(1L)
  }
}
