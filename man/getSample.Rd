% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/output.R
\name{getSample}
\alias{getSample}
\title{extract samples from pdphmc output}
\usage{
getSample(object, which.par = NULL, include.warmup = FALSE)
}
\arguments{
\item{object}{S4 class of type `run-output` typically generated by function `run`}

\item{which.par}{character vector of parameter/generated names}

\item{include.warmup}{logical}
}
\description{
extract samples of specific parameters from pdphmc output
}
