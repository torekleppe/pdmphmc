.pdmphmc_valid_control_fields <- c("absTol",
                                   "relTol",
                                   "csvPrec",
                                   "lambda",
                                   "bash.parallel",
                                   "fixedEps")

.pdmphmc_default_control <- function(){
  return(list(absTol=1.0e-4,
              relTol=1.0e-4,
              csvPrec=8L,
              lambda=c(1.0,2.0,1.0),
              bash.parallel=.Platform$OS.type=="unix"))
}

pdmphmc_control <-function(...){
  arg.list <- c(as.list(environment()), list(...))

  ctrl <- .pdmphmc_default_control()
  ctrl.nms <- names(ctrl)


  if(length(arg.list)>0){
    nms <- names(arg.list)
    for(i in 1:length(arg.list)){
      in.ctrl <- ctrl.nms==nms[i]
      if(any(in.ctrl)){
        j <- which(in.ctrl)[1]
        ctrl[[j]] <- arg.list[[i]]
      } else if(any(.pdmphmc_valid_control_fields==nms[i])){
        ctrl <- c(ctrl,arg.list)
        names(ctrl)[length(ctrl)] <- nms[i]

      } else if(nms[i]==""){
        message(paste0("pdmphmc_control : unnamed argument with value ",arg.list[[i]]," ignored"))
      } else {
        message(paste0("pdmphmc_control : argument ",nms[i]," ignored"))
      }
    }
  }
  return(ctrl)
}


#' Run pdmphmc model
#'
#' @description Run a model made using the build()-function
#'
#' @param build.object model made using the build()-function.
#' @param samples How many equidistant samples to record.
#' @param warmupFrac Which proportion of trajectory is to be used for warmup (tuning and reaching stationary phase).
#' @param Tmax (time-)length of trajectories
#' @param data named list of data, with names and formats corresponding to the DATA_* statements in the model struct
#' @param control named list for setting tuning parameters (integrator tolerances etc)
#' @param chains (integer>0) How many trajectories should be run?
#' @param cores (0<integer<=chains) How many cores should be used? Based on foreach-package.
#' @param seed (integer) random number seed.
#' @param fixedMiDiag (named list) set the scaling/mass matrix diagonal to a fixed quantity (turn of adaption), e.g. if the model contains a parameter par, then list(par=2.0) fixes the scaling so that the posterior of par is assumed to have variance around 2.0
#' @param store.pars (character vector) which parameters are to be stored. If empty, then all parameters are stored. Generated quantites are always stored, and should not be passed using this argument
#' @param clean (logical) remove output files after they have been read into R? Mainly for debugging purposes


run <- function(build.object,
                samples=2000L,
                warmupFrac=0.5,
                Tmax=10000.0,
                data=list(),
                control=pdmphmc_control(),
                chains=4L,
                cores=1L,
                seed=1L,
                fixedMiDiag = list(),
                store.pars=NULL,
                clean=TRUE){

  if(! .check.build(build.object)) stop("bad build.object")
  out <- new("run-output")
  out@build <- build.object
  out@samples <- samples
  # write data to appropriate file
  cat(jsonlite::toJSON(data),
      file=normalizePath( paste0(build.object@file.name.base,"_data.json"),mustWork = FALSE))


  control.file <- control
  if(!is.null(store.pars)){
    if(!is.character(store.pars)){
      stop("bad format of argument store.pars, should be character vector")
    }
    control.file <- c(control.file,list(store.pars=store.pars))
  }

  if(warmupFrac>=0.0 && warmupFrac<=1.0){
    control.file <- c(control.file,list(warmupFrac=warmupFrac))
  } else {
    stop("bad warmupFrac")
  }

  if(Tmax>=0.0){
    control.file <- c(control.file,list(Tmax=Tmax))
  } else {
    stop("bad Tmax")
  }

  if(samples>0){
    control.file <- c(control.file,list(samples=samples))
  } else {
    stop("bad samples")
  }

  control.file <- c(control.file,list(seed=seed))

  if(length(fixedMiDiag)>0){
    control.file <- c(control.file,list(fixedMiDiag=fixedMiDiag))
  }


  cat(jsonlite::toJSON(control.file),
      file=paste0(build.object@file.name.base,"_control.json"))




  # run binary file, possibly in parallell
  bin.eflag <- rep(0,chains)
  if(cores>1L){
    if(! control$bash.parallel){
      # parallel
      `%dopar%` <- foreach::`%dopar%`
      doParallel::registerDoParallel(cl <- parallel::makeCluster(
        min(cores,parallel::detectCores())))

      bin.eflag <- foreach::foreach(chain = 1:chains,.inorder=FALSE,.combine='c') %dopar% system2(build.object@file.name.base,args =toString(chain))
      parallel::stopCluster(cl)
    } else {
      nbatches <- ceiling(chains/cores)
      for(bat in 1:nbatches){
        c.chains <- seq(from=(bat-1)*cores+1,to=min(chains,bat*cores))
        cmd <- "(trap 'kill 0' SIGINT; "
        for(cc in c.chains){
          cmd <- paste0(cmd,build.object@file.name.base," ",cc," & ")
        }
        cmd <- paste0(cmd," wait)")
        print(cmd)
        tt <- system2("bash",input=cmd)
        bin.eflag[c.chains] <- tt
      }
    }
  } else {
    # serial mode
    for(chain in 1:chains){
      bin.eflag[chain] <- system2(build.object@file.name.base,args =toString(chain))
    }
  }

  if(max(abs(bin.eflag))>0.1){
    print(bin.eflag)
    stop("run failed")
  }

  out@warmup <- as.integer(warmupFrac * samples)
  out@chains <- as.integer(chains)
  out@samples <- as.integer(samples)
  out@bin.eflag <- list(bin.eflag)
  # collect output
  for(chain in 1:chains){


    fn.p <- normalizePath(paste0(build.object@file.name.base,"_",chain,"_point.csv"),mustWork=FALSE)
    s.point <- NULL
    if(file.exists(fn.p)){
      s.point <- as.matrix(utils::read.csv(fn.p,check.names = FALSE))
      if(clean) file.remove(fn.p)
    }

    fn.i <- normalizePath(paste0(build.object@file.name.base,"_",chain,"_int.csv"),mustWork = FALSE)
    s.int <- NULL
    if(file.exists(fn.i)){
      s.int <- as.matrix(utils::read.csv(fn.i, check.names = FALSE))
      if(clean) file.remove(fn.i)
    }

    fn.d <- normalizePath(paste0(build.object@file.name.base, "_", chain, "_diagnostics.csv"))
    s.diag <- utils::read.csv(fn.d, check.names = FALSE)
    if (clean) file.remove(fn.d)

    fn.a <- normalizePath(paste0(build.object@file.name.base,"_",chain,"_auxInfo.json"))
    s.aux <- .from.jsonFile(fn.a)
    if(clean) file.remove(fn.a)

    if(chain==1){
      #allocate space

      if(!is.null(s.point)){
        point.dims <- dim(s.point)
        out@pointSamples <- array(dim = c(point.dims[1], chains, point.dims[2]))
        dimnames(out@pointSamples) <- list(NULL, NULL, colnames(s.point))
      } else {
        message("Note: no point samples recorded")
      }
      if(!is.null(s.int)){
        int.dims <- dim(s.int)
        out@intSamples <- array(dim = c(int.dims[1], chains, int.dims[2]))
        dimnames(out@intSamples) <- list(NULL, NULL, colnames(s.int))
      } else {
        message("Note: no integrated samples recorded")
      }
    }

    if(!is.null(s.point)) out@pointSamples[, chain, ] <- s.point
    if(!is.null(s.int)) out@intSamples[, chain, ] <- s.int
    out@diagnostics[[chain]] <- s.diag[s.diag[,"intID"]==0,]
    out@aux.info[[chain]] <- s.aux
    out@CPUtime[[chain]] <- s.aux$CPUtime

  } # done collecting output

  # clean up common files
  if(clean){
    file.remove(normalizePath( paste0(build.object@file.name.base,"_data.json"),mustWork = FALSE))
    file.remove(normalizePath( paste0(build.object@file.name.base,"_control.json"),mustWork = FALSE))
  }

  return(out)
}
