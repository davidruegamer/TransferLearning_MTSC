# devtools::install_github("https://github.com/mlr-org/mlr3tuningspaces")
# reticulate::use_condaenv("deepregression")
library(reticulate)
library(mlr3keras)
library(mlr3misc)
library(mlr3hyperband)
library(checkmate)
library(data.table)
library(paradox)
library(xgboost)
library(dplyr)
library(mlr3tuningspaces)
source("code/mlr3keras.R")
source("code/xgboost.R")

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

max_epochs = 1000L


#---------------------------------------------------------------------
# Learners

inception = LearnerClassifKerasFDA$new(id = "inception", architecture = KerasArchitectureInceptionNet$new())

fcnet = LearnerClassifKerasFDA$new(id = "fcnet", architecture = KerasArchitectureFCN$new())

xgboost = LearnerClassifXgboostFDA$new()


#----------------------------------------------------------------------
# Benchmark Setup
library(mlr3tuning)
library(mlr3hyperband)
resampling = rsmp("holdout", ratio = 0.8)
tuner = tnr("hyperband")
seed = 123456L

# Global Tuning Space fcnet / inception
tune_space = list(
  lr = to_tune(1e-5, 1e-2),
  epochs = to_tune(p_int(lower = 1, upper = max_epochs, tags = "budget")),
  augmentation_ratio = to_tune(0, 20),
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
  dtwwarp = to_tune(),
  # shapedtwwarp = to_tune(),
  # wdba = to_tune(),
  # discdtw = to_tune(),
  # discsdtw = to_tune(),
  seed = seed
)


tune_space_fcnet = list(
  filters = to_tune(4, 128)
)
fcnet$param_set$values = insert_named(
  fcnet$param_set$values, 
  c(tune_space, tune_space_fcnet)
)

fcnet_at = AutoTuner$new(
  learner = fcnet,
  resampling = resampling,
  measure = msr("classif.ce"),
  terminator = trm("none"),
  tuner=tuner,
  store_tuning_instance = TRUE,
  store_benchmark_result = TRUE,
  store_models = FALSE
)



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
  measure = msr("classif.ce"),
  terminator = trm("none"),
  tuner=tuner,
  store_tuning_instance = TRUE,
  store_benchmark_result = TRUE,
  store_models = FALSE
)
# inception_at$train(gait)
# 
# fcnet$train(gait)
# 
# inception$train(gait)


tune_space_xgb = lts("classif.xgboost.default")
tune_space_xgb$values$nrounds = to_tune(p_int(lower = 1, upper = max_epochs, tags = "budget"))
xgboost$param_set$values = insert_named(xgboost$param_set$values, tune_space_xgb$values)
xgb_at = AutoTuner$new(
  learner = xgboost,
  resampling = resampling,
  measure = msr("classif.ce"),
  terminator = trm("none"),
  tuner=tuner,
  store_tuning_instance = TRUE,
  store_benchmark_result = TRUE,
  store_models = FALSE
)
# xgb_at$train(gait)

learners <- list (fcnet_at, inception_at, xgb_at)

design <- benchmark_grid(tasks = gait,
                         learners = learners,
                         resamplings = rsmp("holdout", ratio = 0.8))

set.seed(seed)

bmr <- benchmark(design,
                 store_models = TRUE,
                 store_backends = FALSE)

measures <- list (msr("classif.acc"),
                  msr("classif.bacc"))

resample_perf <- as.data.table (bmr$score(measures = measures)) %>%
  as.data.frame() %>%
  dplyr::select (nr, task_id, learner_id, resampling_id, iteration, matches ("classif."))


saveRDS(resamprle_perf,
        "output/resample_perf.RDS")
# resample_perf <- readRDS("output/resample_perf.RDS")

saveRDS(bmr,
        "output/resampling_models.RDS")
# bmr <- readRDS("output/resampling_models.RDS")





readRDS("output/resample_perf.RDS")


resample_perf

aggr = bmr$aggregate()
print(aggr)

learners = as.data.table(bmr)$learner
lapply(1:length(learners), function(i) learners[[i]]$tuning_result)

tune_res <- extract_inner_tuning_results(bmr)


hyperband_schedule = function(r_min, r_max, eta, integer_budget = TRUE) {
  r = r_max / r_min
  s_max = floor(log(r, eta))
  b = (s_max + 1) * r

  map_dtr(s_max:0, function(s) {
    nb = ceiling((b / r) * ((eta^s) / (s + 1)))
    rb = r * eta^(-s)
    map_dtr(0:s, function(i) {
      ni = floor(nb * eta^(-i))
      ri = r_min * rb * eta^i
      if (integer_budget) ri = round(ri)
      data.table(bracket = s, stage = i, budget = ri, n = ni)
    })
  })
}

hyperband_schedule(1, 1000, 2)

bmr$resample_results$resample_result[[1]]
