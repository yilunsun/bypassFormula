# Hello, world!
#
# This is an example function named 'hello'
# which prints 'Hello, world!'.
#
# You can learn more about package authoring with RStudio at:
#
#   http://r-pkgs.had.co.nz/
#
# Some useful keyboard shortcuts for package authoring:
#
#   Install Package:           'Cmd + Shift + B'
#   Check Package:             'Cmd + Shift + E'
#   Test Package:              'Cmd + Shift + T'

npCBPS_neo <- function(y,x) {
  library(CBPS)
  invisible(capture.output(fit <- npCBPS(y~x)))
  return(1/(length(y)*unlist(fit['weights'])))
}

gam_neo <- function(a,ps,y,a_out) {
  library(mgcv)
  fit <- gam(y~s(a)+s(ps,by=a))
  pred <- predict(fit, data.frame(ps=ps,a=rep(a_out,each=length(ps))))
  optvalue <- apply(matrix(pred, nrow = length(ps)), 2, mean)
  return(optvalue)
}
