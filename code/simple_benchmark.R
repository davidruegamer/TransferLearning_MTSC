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
gaitrec <- data$gaitrec
gaitrec$grp <- as.factor(gaitrec$grp)
nclasses <- nlevels(gaitrec$grp)
gaitrec$grp <- as.numeric(gaitrec$grp)
y = gaitrec$grp
gaitrec$grp = NULL

# Convert to functional
gr = map(gaitrec, as_functional)
df = data.table(grp = as.factor(y))
map(names(gr), function(x) {set(df, j = x, value = gr[[x]]); NULL})

gait = TaskClassif$new("gait", df, target = "grp")

#---------------------------------------------------------------------
# Learners

inception = LearnerClassifKerasFDAInception$new(id = "inception", architecture = KerasArchitectureInceptionNet$new())
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
ps <- inception$param_set$values
ps$augmentation_ratio <- 2
inception2$param_set$values <- ps

fcnet2 = LearnerClassifKerasFDAFCN$new(id = "fcnet", architecture = KerasArchitectureFCN$new())
ps <- fcnet$param_set$values
ps$augmentation_ratio <- 2
fcnet2$param_set$values <- ps

inception4 = LearnerClassifKerasFDAInception$new(id = "inception", architecture = KerasArchitectureInceptionNet$new())
ps <- inception$param_set$values
ps$augmentation_ratio <- 4
inception4$param_set$values <- ps

fcnet4 = LearnerClassifKerasFDAFCN$new(id = "fcnet", architecture = KerasArchitectureFCN$new())
ps <- fcnet$param_set$values
ps$augmentation_ratio <- 4
fcnet4$param_set$values <- ps

inception8 = LearnerClassifKerasFDAInception$new(id = "inception", architecture = KerasArchitectureInceptionNet$new())
ps <- inception$param_set$values
ps$augmentation_ratio <- 8
inception8$param_set$values <- ps

fcnet8 = LearnerClassifKerasFDAFCN$new(id = "fcnet", architecture = KerasArchitectureFCN$new())
ps <- fcnet$param_set$values
ps$augmentation_ratio <- 8
fcnet8$param_set$values <- ps

inception12 = LearnerClassifKerasFDAInception$new(id = "inception", architecture = KerasArchitectureInceptionNet$new())
ps <- inception$param_set$values
ps$augmentation_ratio <- 12
inception12$param_set$values <- ps

fcnet12 = LearnerClassifKerasFDAFCN$new(id = "fcnet", architecture = KerasArchitectureFCN$new())
ps <- fcnet$param_set$values
ps$augmentation_ratio <- 12
fcnet12$param_set$values <- ps

#----------------------------------------------------------------------
# Define Resampling
set.seed(123)
resampling = rsmp("cv", folds = 10)

# -------------------------- Set Up Image TL  ------------------------

tlNet = LearnerClassifKerasCNN$new()
tlNet

# -------------------------- Set Up XGBOOST ------------------------

xgboostFDA = LearnerClassifXgboostFDA$new()

tune_space_xgb = mlr3tuningspaces::lts("classif.xgboost.default")
tune_space_xgb$values$nrounds = to_tune(p_int(lower = 1, upper = 50L, tags = "budget"))
xgboost$param_set$values = insert_named(xgboostFDA$param_set$values, 
                                        tune_space_xgb$values) # comment out for default
tuner = tnr("hyperband")

xgb_at = AutoTuner$new(
  learner = xgboostFDA,
  resampling = resampling,
  measure = msr("classif.logloss"),
  terminator = trm("none"),
  tuner=tuner,
  store_tuning_instance = TRUE,
  store_benchmark_result = TRUE,
  store_models = TRUE
)

# -------------------------- Set Up multinom reg ------------------------

logreg = as_learner( po("flatfunct") %>>% po("learner", lrn("classif.glmnet"), lambda = 0) )

# -------------------------- Train ------------------------
learners <- list (fcnet, fcnet2, fcnet4, fcnet8, fcnet12,
                  inception, inception2, inception4, inception8, inception12,
                  xgb_at, logreg)

design <- benchmark_grid(tasks = gait,
                         learners = learners,
                         resamplings = resampling)

bmr <- benchmark(design,
                 store_models = FALSE,
                 store_backends = FALSE,
                 encapsulate = "none")

saveRDS(bmr, file="output/final_result.RDS")

# train first model on other data
# save weights
# hyperparameter -> initial weights
inception$train(other_task)
inception$save_weights(my_path)
inception$param_set$values$initial_weights = my_path

for(i in 1:7){
  inception$transfer(gait, row_ids = bmr$resamplings$resampling[[1]]$train_set(i))
  inception$predict(gait, row_ids = bmr$resamplings$resampling[[1]]$test_set(i))
}

