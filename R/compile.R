#
# .default.compiler.info<- function(flags="-O3"){
#   if(identical(.Platform$OS.type,"windows")){
#     if(pkgbuild::has_rtools()){
#       #rtools.path <- strsplit(pkgbuild::rtools_path(),"usr")[[1]][1]
#
#
#       #full.path <- strsplit(Sys.getenv("PATH"),split=";")[[1]]
#       #rtools.paths <- grep("rtools",full.path,ignore.case = TRUE,fixed=TRUE)
#       #if(length(rtools.paths)==0) stop("rtools not in path!")
#       #rtools.paths <- full.path[rtools.paths]
#       #mingw.path <- grep("mingw",rtools.paths,ignore.case = TRUE,fixed=TRUE)
#       #if(length(mingw.path)==0) stop("cannot find mingw-directory in path!")
#       #compiler.path <- paste0(rtools.paths[mingw.path],"\\g++.exe")
#       #rtools.path <- gsub("/","\\",pkgbuild::rtools_path(),fixed=TRUE) # note, this is not the path to the compiler binary!!
#       #print(rtools.path)
#       #compiler.path <- paste0(strsplit(rtools.path,"usr")[[1]][1],"mingw64\\bin\\g++.exe")
#       #print(compiler.path)
#       #if(file.exists(normalizePath(compiler.path,mustWork=FALSE))){
#       #  return(list(compiler=normalizePath(compiler.path),flags=flags))
#       #}
#       #compiler.path <- paste0(strsplit(rtools.path,"usr")[[1]][1],"mingw_64\\bin\\g++.exe")
#       #print(compiler.path)
#       flags.ext <- paste0(flags," -DRCPP_PARALLEL_USE_TBB=1 -Wno-unused-result -Wno-deprecated-declarations -Wno-unknown-pragmas -Wno-ignored-attributes" )
#       return(list(compiler="g++",flags=flags.ext,bld.tools=TRUE))
#     } else {
#       stop("requires a working c++ compiler, get the rtools package for your R version")
#     }
#   } else if(identical(.Platform$OS.type,"unix")) {
#     flags.ext <- paste0(flags," -Wno-unknown-pragmas -Wno-deprecated-declarations")
#     return(list(compiler="g++",flags=flags.ext,bld.tools=FALSE))
#   } else {
#     stop("Unknown OS.type")
#   }
#
# }
#
#
# plugin.compiler.info<- function(flags="-O3"){
#
#   pl <- inline::getPlugin("rstan")
#
#   if(identical(.Platform$OS.type,"windows")){
#     if(pkgbuild::has_rtools()){
#       flags.ext <- paste0(flags,pl$env$PKG_LIBS,pl$env$PKG_CPPFLAGS )
#       return(list(compiler="g++",flags=flags.ext,bld.tools=TRUE))
#     } else {
#       stop("requires a working c++ compiler, get the rtools package for your R version")
#     }
#   } else if(identical(.Platform$OS.type,"unix")) {
#     #flags.ext <- paste0(flags,pl$env$PKG_LIBS,pl$env$PKG_CPPFLAGS," -Wno-unknown-pragmas -Wno-deprecated-declarations")
#     flags.ext <- paste0(pl$env$PKG_CPPFLAGS," -Wno-unknown-pragmas -Wno-deprecated-declarations")
#     return(list(compiler="g++",flags=flags.ext,bld.tools=FALSE))
#   } else {
#     stop("Unknown OS.type")
#   }
#
# }


default.compiler.info<- function(flags="-O3 "){

  if(identical(.Platform$OS.type,"windows")){
    if(pkgbuild::has_rtools()){
      flags.ext <- flags #paste0(flags,StanHeaders:::CxxFlags(TRUE))
      return(list(compiler="g++",flags=flags.ext,bld.tools=TRUE))
    } else {
      stop("requires a working c++ compiler, get the rtools package for your R version")
    }
  } else if(identical(.Platform$OS.type,"unix")) {

    flags.ext <- paste0(flags," -Wno-unknown-pragmas -Wno-deprecated-declarations")
    return(list(compiler="g++",flags=flags.ext,bld.tools=FALSE))
  } else {
    stop("Unknown OS.type")
  }

}




.compileCpp <- function(bo,
                        compiler.info=default.compiler.info(),
                        include=""){

  stan.includes <- paste0(StanHeaders:::CxxFlags(TRUE)," ",
                          StanHeaders:::LdFlags(TRUE))
  # note include of RcppParallel happens in line above
  package.includes <- paste0(stan.includes,
    " -I",normalizePath(system.file('include', package = "StanHeaders")),
    " -I",normalizePath(system.file('include', package = "RcppEigen")),
    " -I",normalizePath(system.file('include', package = "BH")),
    " -I",normalizePath(system.file('include', package = "RcppProgress")),
    " -I",normalizePath(system.file('include', package = "pdmphmc"))
  )

  #" -I",normalizePath(system.file('include', package = "RcppParallel"))

  cpp.file <- normalizePath(system.file("main_template/main_template.cpp",package = "pdmphmc"))

  model.includes <- paste0("-I",bo@work.folder)

  compilerArgs <- paste0(cpp.file,
                         " -o ",bo@file.name.base,
                         " -std=c++17 ",compiler.info$flags," ",
                         model.includes," ",
                         package.includes," ",
                         include)

  if(compiler.info$bld.tools){
    pkgbuild::with_build_tools(ret <- system2(command=compiler.info$compiler,
                                      args=compilerArgs,
                                      stdout = TRUE,
                                      stderr = TRUE))
    print(ret)
    eflag <- attr(ret,"status")
    print(eflag)
  } else {

    ret <- system2(command=compiler.info$compiler,
                   args=compilerArgs,
                   stdout = TRUE,
                   stderr = TRUE)

    eflag <- attr(ret,"status")
  }
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
