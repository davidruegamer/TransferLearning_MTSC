# devtools::install_github("https://github.com/mlr-org/mlr3tuningspaces")
# devtools::install_github("mlr-org/mlr3keras")
# reticulate::use_condaenv("deepregression")
# reticulate::use_condaenv("mlr3keras", required = TRUE)

library(reticulate)
library(keras)
library(mlr3keras)
library(mlr3misc)
library(mlr3hyperband)
library(checkmate)
library(data.table)
library(paradox)
library(xgboost)
library(dplyr)
library(mlr3tuning)
library(mlr3hyperband)
library(mlr3tuningspaces)
library(mlr3learners)
library(mlr3pipelines)
library(R6)
source("code/mlr3keras.R")
source("code/xgboost.R")
source("code/PipeOpFlatFunct.R")
if(file.exists(".RData")) file.remove(".RData")

#---------------------------------------------------------------------
# Data to Task
data <- readRDS("data/trans_learn.RDS")
pfps <- data$pfps
pfps$grp <- as.factor(pfps$grp)
nclasses <- nlevels(pfps$grp)
pfps$grp <- as.numeric(pfps$grp)
y = pfps$grp
pfps$grp = NULL

# Convert to functional
gr = map(pfps, as_functional)
df = data.table(grp = as.factor(y))
map(names(gr), function(x) {set(df, j = x, value = gr[[x]]); NULL})

gait = TaskClassif$new("pfps", df, target = "grp")
#---------------------------------------------------------------------
# Learners

inception = LearnerClassifKerasFDAInception$new(id = "inception", architecture = KerasArchitectureInceptionNet$new())
inception$predict_type <- "prob"
ps <- inception$param_set$values
ps$validation_split <- 0.2
ps$augmentation_ratio <- 0
ps$filters <- 50L
ps$lr <- exp(-5.6)
ps$magwarp <- TRUE
ps$spawner <- TRUE
ps$dtwwarp <- TRUE
ps$windowslice <- TRUE
inception$param_set$values <- ps
# inception$train(gait)
# # Can change params in-between
# inception$transfer(gait)


fcnet = LearnerClassifKerasFDAFCN$new(id = "fcnet", architecture = KerasArchitectureFCN$new())
fcnet$predict_type <- "prob"
ps <- fcnet$param_set$values
ps$validation_split <- 0.2
ps$augmentation_ratio <- 0
ps$filters <- 50
ps$lr <- exp(-5.6)
ps$magwarp <- TRUE
ps$spawner <- TRUE
ps$dtwwarp <- TRUE
ps$windowslice <- TRUE
fcnet$param_set$values <- ps

inception2 = LearnerClassifKerasFDAInception$new(id = "inception", architecture = KerasArchitectureInceptionNet$new())
inception2$predict_type <- "prob"
ps <- inception$param_set$values
ps$augmentation_ratio <- 2
inception2$param_set$values <- ps

fcnet2 = LearnerClassifKerasFDAFCN$new(id = "fcnet", architecture = KerasArchitectureFCN$new())
fcnet2$predict_type <- "prob"
ps <- fcnet$param_set$values
ps$augmentation_ratio <- 2
fcnet2$param_set$values <- ps

inception4 = LearnerClassifKerasFDAInception$new(id = "inception", architecture = KerasArchitectureInceptionNet$new())
inception4$predict_type <- "prob"
ps <- inception$param_set$values
ps$augmentation_ratio <- 4
inception4$param_set$values <- ps

fcnet4 = LearnerClassifKerasFDAFCN$new(id = "fcnet", architecture = KerasArchitectureFCN$new())
fcnet4$predict_type <- "prob"
ps <- fcnet$param_set$values
ps$augmentation_ratio <- 4
fcnet4$param_set$values <- ps

inception8 = LearnerClassifKerasFDAInception$new(id = "inception", architecture = KerasArchitectureInceptionNet$new())
inception8$predict_type <- "prob"
ps <- inception$param_set$values
ps$augmentation_ratio <- 8
inception8$param_set$values <- ps

fcnet8 = LearnerClassifKerasFDAFCN$new(id = "fcnet", architecture = KerasArchitectureFCN$new())
fcnet8$predict_type <- "prob"
ps <- fcnet$param_set$values
ps$augmentation_ratio <- 8
fcnet8$param_set$values <- ps

inception12 = LearnerClassifKerasFDAInception$new(id = "inception", architecture = KerasArchitectureInceptionNet$new())
inception12$predict_type <- "prob"
ps <- inception$param_set$values
ps$augmentation_ratio <- 12
inception12$param_set$values <- ps

fcnet12 = LearnerClassifKerasFDAFCN$new(id = "fcnet", architecture = KerasArchitectureFCN$new())
fcnet12$predict_type <- "prob"
ps <- fcnet$param_set$values
ps$augmentation_ratio <- 12
fcnet12$param_set$values <- ps

#----------------------------------------------------------------------
# Define Resampling
set.seed(123)
resampling = rsmp("cv", folds = 10)

# -------------------------- Set Up Image TL  ------------------------

tlnet = LearnerClassifKerasCNN$new()
tlnet$predict_type <- "prob"
tlnet$param_set$values$unfreeze_n_last_layers <- 1L
tlnet$param_set$values$cl_layer_units <- 256L

# -------------------------- Set Up XGBOOST ------------------------

xgboostFDA = LearnerClassifXgboostFDA$new()
xgboostFDA$predict_type <- c("prob")

tune_space_xgb = mlr3tuningspaces::lts("classif.xgboost.default")
tune_space_xgb$values$nrounds = to_tune(p_int(lower = 1, upper = 50L, tags = "budget"))
xgboostFDA$param_set$values = insert_named(xgboostFDA$param_set$values,
                                           tune_space_xgb$values) # comment out for default
tuner = tnr("hyperband")

xgb_at = AutoTuner$new(
  learner = xgboostFDA,
  resampling = rsmp("cv", folds = 3),
  measure = msr("classif.logloss"),
  terminator = trm("none"),
  tuner=tuner,
  store_tuning_instance = TRUE,
  store_benchmark_result = TRUE,
  store_models = TRUE
)

# -------------------------- Set Up multinom reg ------------------------

logreg = as_learner( po("flatfunct") %>>% po("learner", lrn("classif.glmnet"), lambda = 0) )
logreg$predict_type <- c("prob")

# -------------------------- Train ------------------------
learners <- list (fcnet, fcnet2, fcnet4, fcnet8, fcnet12,
                  inception, inception2, inception4, inception8, inception12,
                  xgb_at, logreg, tlnet)

design <- benchmark_grid(tasks = gait,
                         learners = learners,
                         resamplings = resampling)

bmr <- benchmark(design,
                 store_models = FALSE,
                 store_backends = FALSE,
                 encapsulate = "none")

saveRDS(bmr, file="output/final_result_pfps.RDS")

