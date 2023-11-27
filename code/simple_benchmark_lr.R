library(reticulate)
library(keras)
library(data.table)
library(dplyr)
library(glmnet)
library(mlr3)
library(mlr3measures)
library(Metrics)

measures <- list (msr("classif.acc"),
                  msr("classif.logloss"),
                  msr("classif.auc"),
                  msr("classif.bbrier"))

measures_man <- function(truth, probmat)
{
  
  c(Metrics:::accuracy(truth, 1*(probmat>0.5)),
    Metrics:::logLoss(truth, probmat),
    Metrics:::auc(truth, probmat),
    Metrics:::mse(truth, probmat)
  )
  
}

metr_names <- c("Accuracy", "Log-loss", "Area under ROC", "Brier Score")

if(file.exists(".RData")) file.remove(".RData")

#---------------------------------------------------------------------
# Data to Task
bmr <- readRDS("output/final_result_pfps.RDS")

data <- readRDS("data/trans_learn.RDS")
pfps <- data$pfps
pfps$grp <- as.factor(pfps$grp)
nclasses <- nlevels(pfps$grp)
pfps$grp <- as.numeric(pfps$grp)
y <- pfps$grp
pfps$grp <- NULL

res <- do.call("rbind", lapply(1:10, function(i){ 
  
  ind_tr <- bmr$resamplings$resampling[[1]]$train_set(i)
  ind_te <- bmr$resamplings$resampling[[1]]$test_set(i)
  
  x_train <- do.call("cbind", pfps)[ind_tr,]
  y_train <- y[ind_tr]-1
  
  x_test <- do.call("cbind", pfps)[ind_te,]
  y_test <- y[ind_te]-1

  cvob1 = suppressWarnings(cv.glmnet(x_train, y_train, family = "binomial"))
  pred <- predict(cvob1, newx = x_test, s = "lambda.min", type = "response")
  
  rres <- measures_man(y_test, pred)
  
  data.frame(
    metric = metr_names,
    value = rres,
    iter = i
  )
  
}))
  
saveRDS(res, file="output/final_result_logreg_pfps.RDS")

library(ggplot2)
ggplot(res, aes(x = metric, y = value)) + geom_boxplot() + 
  theme_bw() + 
  theme(text = element_text(size = 14)) 
