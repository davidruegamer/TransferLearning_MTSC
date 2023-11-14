library(data.table)
library(tidyverse)
library(ggplot2)
library(mlr3)
library(abind)
library(reticulate)

# reticulate::use_condaenv("mlr3keras", required = TRUE)

np = reticulate::import("numpy")


bmr <- readRDS("output/final_result_pfps.RDS")

data <- readRDS("data/trans_learn.RDS")
pfps <- data$pfps
pfps$grp <- as.factor(pfps$grp)
nclasses <- nlevels(pfps$grp)
pfps$grp <- as.numeric(pfps$grp)
y <- pfps$grp
pfps$grp <- NULL

for(i in 1:10){ 
  
  ind_tr <- bmr$resamplings$resampling[[1]]$train_set(i)
  ind_te <- bmr$resamplings$resampling[[1]]$test_set(i)
  
  x_train <- do.call("abind", lapply(pfps,  function(ts) 
    array(as.matrix(ts[ind_tr,]), 
          dim = c(length(ind_tr), 101, 1)
          )
    ))
  y_train <- y[ind_tr]
  
  x_test <- do.call("abind", lapply(pfps,  function(ts) 
    array(as.matrix(ts[ind_te,]), 
          dim = c(length(ind_te), 101, 1)
    )
  ))
  y_test <- y[ind_te]
  
  np$save(paste0("data/TL/x_train", i, "_pfps.npy"), r_to_py(x_train))
  np$save(paste0("data/TL/x_test", i, "_pfps.npy"), r_to_py(x_test))
  np$save(paste0("data/TL/y_train", i, "_pfps.npy"), r_to_py(y_train))
  np$save(paste0("data/TL/y_test", i, "_pfps.npy"), r_to_py(y_test))
  

}

