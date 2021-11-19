# devtools::install_github("https://github.com/mlr-org/mlr3tuningspaces")
library(reticulate)
library(mlr3keras)
library(mlr3misc)
library(checkmate)
library(data.table)
library(paradox)
library(xgboost)
library(dplyr)
library(mlr3tuningspaces)
source("code/mlr3keras.R")
source("code/xgboost.R")


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
# Learner
inception = LearnerClassifKerasFDA$new(architecture = KerasArchitectureInceptionNet$new())
# inception$param_set$values$nb_epochs = 5L
# inception$train(gait)
# inception$predict(gait)

fcnet = LearnerClassifKerasFDA$new(architecture = KerasArchitectureFCN$new())
# fcnet$param_set$values$nb_epochs = 5L
# fcnet$train(gait)
# fcnet$predict(gait)

xgboost = LearnerClassifXgboostFDA$new()
# xgboost$train(gait)
# xgboost$predict(gait)

#----------------------------------------------------------------------
# Benchmark Setup
library(mlr3tuning)
library(mlr3hyperband)
resampling = rsmp("cv", folds = 3)
tuner = tnr("hyperband")

# Global Tuning Space
tune_space = list(
  lr = to_tune(1e-5, 1e-2),
  epochs = to_tune(p_int(lower = 1, upper = 512, tags = "budget")),
  augmentation_ratio = to_tune(0, 20),
  jitter = to_tune(),
  scaling = to_tune(),
  permutation = to_tune(),
  randompermutation = to_tune(),
  magwarp = to_tune(),
  timewarp = to_tune(),
  windowslice = to_tune(),
  windowwarp = to_tune(),
  rotation = to_tune(),
  spawner = to_tune(),
  dtwwarp = to_tune(),
  shapedtwwarp = to_tune(),
  wdba = to_tune(),
  discdtw = to_tune(),
  discsdtw = to_tune()
)


tune_space_fcnet = list(
  filters = to_tune(4, 128)
)
fcnet$param_set$values = insert_named(fcnet$param_set$values, c(tune_space, tune_space_fcnet))

fcnet_at = AutoTuner$new(
  learner = fcnet,
  resampling = resampling,
  measure = msr("classif.acc"),
  terminator = trm("evals"),
  tuner=tuner
)
# fcnet_at$train(gait)


tune_space_inception = list(
  filters = to_tune(4, 128),
  use_residual = to_tune(),
  use_bottleneck = to_tune(),
  kernel_size = to_tune(3, 101)
)
inception$param_set$values = insert_named(inception$param_set$values, c(tune_space, tune_space_inception))
inception_at = AutoTuner$new(
  learner = inception,
  resampling = resampling,
  measure = msr("classif.acc"),
  terminator = trm("evals"),
  tuner=tuner
)
# inception_at$train(gait)
# 
# fcnet$train(gait)
# 
# inception$train(gait)


tune_space_xgb = lts("classif.xgboost.default")
tune_space_xgb$values$nrounds = to_tune(p_int(lower = 1, upper = 2000, tags = "budget"))
xgboost$param_set$values = insert_named(xgboost$param_set$values, tune_space_xgb$values)
xgb_at = AutoTuner$new(
  learner = xgboost,
  resampling = resampling,
  measure = msr("classif.acc"),
  terminator = trm("evals"),
  tuner=tuner
)

# xgb_at$train(gait)


learners <- list (fcnet_at, inception_at, xgb_at)

design <- benchmark_grid(tasks = gait,
                         learners = learners,
                         resamplings = rsmp("holdout", ratio = 0.8))

set.seed(123456)

bmr <- benchmark(design,
                 store_models = FALSE,
                 store_backends = FALSE)

measures <- list (# msr("classif.auc"),
                  msr("classif.acc"),
                  msr("classif.sensitivity"),
                  msr("classif.specificity"),
                  msr("classif.precision"),
                  msr("classif.fbeta"))

resample_perf <- as.data.table (bmr$score(measures = measures)) %>%
  as.data.frame() %>%
  dplyr::select (nr, task_id, learner_id, resampling_id, iteration, matches ("classif."))


saveRDS(resample_perf,
        "output/resample_perf.RDS")

saveRDS(bmr,
        "output/resampling_models.RDS")

resample_perf

aggr = bmr$aggregate()
print(aggr)

learners = as.data.table(bmr)$learner
lapply(1:length(learners), function(i) learners[[i]]$tuning_result)
