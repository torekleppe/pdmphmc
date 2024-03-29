% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/build.R
\name{build}
\alias{build}
\title{Build (i.e. parse and compile) a pdmphmc model}
\usage{
build(
  model.file,
  model.class.name = "model",
  process.type = c("HMCProcessConstr", "HMCProcess", "RMHMCProcess"),
  step.type = c("RKDP54", "RKBS32"),
  TM.type = c("diagLinearTM_VARI", "diagLinearTM_ISG", "identityTM"),
  amt = process.type == "RMHMCProcess",
  metric.tensor.type = c("Sparse", "Dense"),
  work.folder = paste0(getwd(), "/pdmphmc_files/"),
  compiler.info = .default.compiler.info(),
  include = "",
  clean = TRUE
)
}
\arguments{
\item{model.file}{a cpp file containing the model specification class}

\item{model.class.name}{the name of the model specification class}

\item{process.type}{which kind of pdmphmc-process to use}

\item{step.type}{which Runge Kutta method}

\item{TM.type}{which Transport Map}

\item{amt}{should amtVar or regular AD types be used}

\item{metric.tensor.type}{either sparse or dense storage}

\item{work.folder}{the folder used for storing files, created if not already existing}

\item{compiler.info}{information related to compiler}

\item{include}{additional flags passed to the compiler. E.g. \code{"-D __DEBUG__"} for integrator related debug info or \code{"-D __TENSOR_DEBUG__"} for Riemannian tensor debugging info.}

\item{clean}{should output files be removed once read into R?}
}
\description{
Build (i.e. parse and compile) a pdmphmc model
}
