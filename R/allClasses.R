library(methods)




setClass("build-output",
         slots = c(model.name = "character",
                   build.id = "character",
                   work.folder = "character",
                   file.name.base = "character",
                   compiler.flag = "numeric"))


setClass("run-output",
         slots = c(build = "build-output",
                   aux.info="list",
                   samples = "integer",
                   warmup = "integer",
                   chains = "integer",
                   pointSamples = "array",
                   intSamples = "array",
                   diagnostics = "list",
                   monitor = "list",
                   intMonitor = "list",
                   CPUtime = "list",
                   bin.eflag = "list"))


