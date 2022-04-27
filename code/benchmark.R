# devtools::install_github("https://github.com/mlr-org/mlr3tuningspaces")
reticulate::use_condaenv("deepregression")
library(reticulate)
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


validation_run <- FALSE
#---------------------------------------------------------------------
# Switch to false to run real experimentvalidation_run = FALSE
seed = 123456L
if (validation_run) {
  max_epochs = 4L
  sample_rows = c(2187L, 2218L, 2080L, 533L, 328L,329L:332L, 152L, 1825L, 2294L, 617L, 533L, 2295L, 396L, 1448L, 532L, 378L, 675L, 878L, 2L, 830L, 207L, 38L, 15L, 1406L, 1279L, 1532L, 1L)
} else {
  max_epochs = 250L
  sample_rows = gait$row_ids
}



#---------------------------------------------------------------------
# Learners

inception = LearnerClassifKerasFDAInception$new(id = "inception", architecture = KerasArchitectureInceptionNet$new())

fcnet = LearnerClassifKerasFDAFCN$new(id = "fcnet", architecture = KerasArchitectureFCN$new())

xgboost = LearnerClassifXgboostFDA$new()

#----------------------------------------------------------------------
# Global Tuning settings
resampling = rsmp("holdout", ratio = 0.8)

# Global Tuning Space fcnet / inception
tune_space = list(
  lr = to_tune(1e-5, 1e-2, logscale = TRUE),
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

# -------------------------- Set Up FCNET ------------------------
tune_space_fcnet = list(
  filters = to_tune(4, 128)
)
fcnet$param_set$values = insert_named(
  fcnet$param_set$values, 
  c(tune_space, tune_space_fcnet)
)
search_space = fcnet$param_set$search_space()$clone()
search_space$subset(setdiff(search_space$ids(), "epochs"))
sampler_fcnet = SamplerUnifwDefault$new(search_space, fcnet_default)
tuner_fcnet = tnr("hyperband",
  sampler = sampler_fcnet
)


fcnet_at = AutoTuner$new(
  learner = fcnet,
  resampling = resampling,
  measure = msr("classif.ce"),
  terminator = trm("none"),
  tuner=tuner_fcnet,
  store_tuning_instance = TRUE,
  store_benchmark_result = TRUE,
  store_models = TRUE
)



# -------------------------- Set Up Inception ------------------------
tune_space_inception = list(
  filters = to_tune(4, 128),
  use_residual = to_tune(),
  use_bottleneck = to_tune(),
  kernel_size = to_tune(3, 101)
)
inception$param_set$values = insert_named(inception$param_set$values, c(tune_space, tune_space_inception))
search_space = inception$param_set$search_space()$clone()
search_space$subset(setdiff(search_space$ids(), "epochs"))
sampler_inception = SamplerUnifwDefault$new(search_space, inception_default)
tuner_inception = tnr("hyperband",
  sampler = sampler_inception
)

inception_at = AutoTuner$new(
  learner = inception,
  resampling = resampling,
  measure = msr("classif.ce"),
  terminator = trm("none"),
  tuner=tuner_inception,
  store_tuning_instance = TRUE,
  store_benchmark_result = TRUE,
  store_models = TRUE
)


# -------------------------- Set Up XGBOOST ------------------------
tune_space_xgb = mlr3tuningspaces::lts("classif.xgboost.default")
aug_space = list(
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
  dtwwarp = to_tune()
)
tune_space_xgb$values$nrounds = to_tune(p_int(lower = 1, upper = max_epochs, tags = "budget"))
xgboost$param_set$values = insert_named(xgboost$param_set$values, c(aug_space, tune_space_xgb$values)) # comment out for default

tuner = tnr("hyperband")

xgb_at = AutoTuner$new(
  learner = xgboost,
  resampling = resampling,
  measure = msr("classif.ce"),
  terminator = trm("none"),
  tuner=tuner,
  store_tuning_instance = TRUE,
  store_benchmark_result = TRUE,
  store_models = TRUE
)


xgb_at$train(gait)

# -------------------------- Train ------------------------
set.seed(seed)
learners <- list (fcnet_at, inception_at, xgb_at)

design <- benchmark_grid(tasks = gait$clone()$filter(rows = sample_rows),
                         learners = learners,
                         resamplings = resampling)

# task, resample, xgboost learner

# DR: commented out
# bmr <- benchmark(design,
#                  store_models = TRUE,
#                  store_backends = FALSE,
#                  encapsulate = "none")
# 
# saveRDS(bmr, file="output/resampling_models.RDS")
# 
bmr <- readRDS("output/resampling_models.RDS")

measures <- list (msr("classif.acc"),
                  msr("classif.bacc"))

resample_perf <- as.data.table (bmr$score(measures = measures)) %>%
  as.data.frame() %>%
  dplyr::select (nr, task_id, learner_id, resampling_id, iteration, matches ("classif."))

# 
# saveRDS(resample_perf,
#         "output/resample_perf.RDS")
# # resample_perf <- readRDS("output/resample_perf.RDS")
# 
resample_perf

aggr = bmr$aggregate() # bmr$score()
print(aggr)

learners = as.data.table(bmr)$learner
best_tuning_res <- lapply(1:length(learners), function(i) learners[[i]]$tuning_result)

tune_res <- extract_inner_tuning_results(bmr)
# 
# 
# # hyperband_schedule = function(r_min, r_max, eta, integer_budget = TRUE) {
# #   r = r_max / r_min
# #   s_max = floor(log(r, eta))
# #   b = (s_max + 1) * r
# 
# #   map_dtr(s_max:0, function(s) {
# #     nb = ceiling((b / r) * ((eta^s) / (s + 1)))
# #     rb = r * eta^(-s)
# #     map_dtr(0:s, function(i) {
# #       ni = floor(nb * eta^(-i))
# #       ri = r_min * rb * eta^i
# #       if (integer_budget) ri = round(ri)
# #       data.table(bracket = s, stage = i, budget = ri, n = ni)
# #     })
# #   })
# # }
# 
# # hyperband_schedule(1, 250, 2)
# 
# # bmr$resample_results$resample_result[[1]]
# 
# 
# inception_default$lr <- exp(inception_default$lr)
# fcnet_default$lr <- exp(fcnet_default$lr)
# inception_default$epochs <- 300
# fcnet_default$epochs <- 300
# 
# # # No augmentation
# inception$param_set$values = insert_named(inception$param_set$values, inception_default)
# fcnet$param_set$values = insert_named(fcnet$param_set$values, fcnet_default)
# 
# noaug_design = benchmark_grid(
#   tasks = gait,
#   learners = c(inception, fcnet),
#   resamplings = bmr$resamplings$resampling
# )
# 
# bmr2 = benchmark(noaug_design)
# saveRDS(bmr2, file="output/resampling_models_noaug.RDS")
# 
# aggr2 = bmr2$aggregate()
# print(aggr2)
# sampler_fcnet$sample(1)
# sampler_inception$sample(1)

############# Re-run best setting ##############
library(mlr3misc)

params <- as.list(as.data.frame(best_tuning_res[[2]])[,-1*c(18:20)])
params$lr <- exp(params$lr)

inception$param_set$values <- 
  insert_named(inception$param_set$values, params)

inception$param_set$values$epochs <- 1000L
inception$param_set$values$val_idx <- bmr$resamplings$resampling[[1]]$test_set(1)

inception$train(gait)

# rerun_res <- resample(gait, inception, bmr$resamplings$resampling[[1]])

