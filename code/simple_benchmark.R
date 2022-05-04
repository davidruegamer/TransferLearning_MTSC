# devtools::install_github("https://github.com/mlr-org/mlr3tuningspaces")
# devtools::install_github("https://github.com/mlr-org/mlr3tuningspaces")
reticulate::use_condaenv("deepregression")
library(reticulate)
library(mlr3keras) # mlr-org/mlr3keras
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
source("code/mlr3keras.R")
source("code/xgboost.R")
source("code/xgboost_noaug.R")
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
ps$filters <- 50
ps$lr <- exp(-5.6)
ps$magwarp <- TRUE
ps$spawner <- TRUE
ps$dtwwarp <- TRUE
ps$windowslice <- TRUE
inception$param_set$values <- ps

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

set.seed(123L)
xgboost = LearnerClassifXgboostFDA$new()
no_aug = list(jitter = FALSE, center = FALSE, scale = FALSE, augmentation_ratio = 0, nrounds = 2)
xgboost$param_set$values = insert_named(xgboost$param_set$values, no_aug)
xgboost$train(gait)
xgboost$predict(gait)$score(msr("classif.ce"))

set.seed(123L)
xgboost2 = LearnerClassifXgboostNoAugFDA$new()
xgboost2$param_set$values$nrounds = 2
xgboost2$train(gait)
xgboost2$predict(gait)$score(msr("classif.ce"))

#----------------------------------------------------------------------
# Define Resampling
set.seed(123)
resampling = rsmp("cv", folds = 3)
# -------------------------- Set Up XGBOOST ------------------------
tune_space_xgb = mlr3tuningspaces::lts("classif.xgboost.default")
tune_space_xgb$values$nrounds = to_tune(p_int(lower = 1, upper = 50L, tags = "budget"))
aug_space = list(
  augmentation_ratio = to_tune(0, 4),
  jitter = to_tune(),
  scaling = to_tune(),
  permutation = to_tune(),
  randompermutation = to_tune(),
  magwarp = to_tune(),
  timewarp = to_tune(),
  windowslice = to_tune(),
  windowwarp = to_tune(),
  # rotation = to_tune(),
  spawner = to_tune(),
  dtwwarp = to_tune()
)

xgboost$param_set$values = insert_named(xgboost$param_set$values, c(tune_space_xgb$values)) # comment out for default
xgboost$param_set$values = insert_named(xgboost$param_set$values, no_aug)
tuner = tnr("hyperband", eta = 3)

xgb_at = AutoTuner$new(
  learner = xgboost,
  resampling = rsmp("cv", folds = 3),
  measure = msr("classif.ce"),
  terminator = trm("evals", n_evals = 50),
  tuner=tuner,
  store_tuning_instance = TRUE,
  store_benchmark_result = TRUE,
  store_models = TRUE
)

# smp = gait$data(cols = c("..row_id", gait$target_names))[, sample(..row_id, 30), by = "grp"][["V1"]]
# xgb_at$train(gait$clone()$filter(smp))


bmr2 = benchmark(benchmark_grid(tasks = gait, learners = list(xgb_at), resamplings = resampling), store_models = TRUE)

# Tuned Augmentation
# 1:         1 <PredictionClassif[20]>  0.7137255
# 2:         2 <PredictionClassif[20]>  0.7124183
# 3:         3 <PredictionClassif[20]>  0.7790850
# Default Augmentation
#    iteration              prediction classif.ce
# 1:         1 <PredictionClassif[20]>  0.7215686
# 2:         2 <PredictionClassif[20]>  0.7241830
# 3:         3 <PredictionClassif[20]>  0.7660131
# No AUgmentation
#  iteration              prediction classif.ce
# 1:         1 <PredictionClassif[20]>  0.8000000
# 2:         2 <PredictionClassif[20]>  0.7385621
# 3:         3 <PredictionClassif[20]>  0.7307190
# -------------------------- Train ------------------------
learners <- list (fcnet, fcnet2, fcnet4, fcnet8, 
                  inception, inception2, inception4, inception8, 
                  xgb_at)

design <- benchmark_grid(tasks = gait,
                         learners = learners,
                         resamplings = resampling)

bmr <- benchmark(design,
                 store_models = FALSE,
                 store_backends = FALSE,
                 encapsulate = "none")

saveRDS(bmr, file="output/resampling_models_simple_woxgb.RDS")
