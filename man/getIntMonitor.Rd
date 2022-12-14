% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/output.R
\name{getIntMonitor}
\alias{getIntMonitor}
\title{Obtain summary statistics from pdphmc integrated output using rstan::monitor}
\usage{
getIntMonitor(object, print = TRUE)
}
\arguments{
\item{object}{S4 class of type `run-output` typically generated by function `run`}

\item{print}{logical passed to rstan::monitor}
}
\description{
Obtain summary statistics from integrated pdphmc output using rstan::monitor
}
