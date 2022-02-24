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

xgboost = LearnerClassifXgboostFDA$new()

#----------------------------------------------------------------------
# Define Resampling
set.seed(123)
resampling = rsmp("cv", folds = 10)

# -------------------------- Set Up XGBOOST ------------------------
tune_space_xgb = mlr3tuningspaces::lts("classif.xgboost.default")
tune_space_xgb$values$nrounds = to_tune(p_int(lower = 1, upper = 50L, tags = "budget"))
xgboost$param_set$values = insert_named(xgboost$param_set$values, tune_space_xgb$values) # comment out for default
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

bmr <- readRDS("output/resampling_models_simple_woxgb.RDS")

measures <- list (msr("classif.acc"),
                  msr("classif.bacc"))

resample_perf <- as.data.table (bmr$score(measures = measures)) %>%
  as.data.frame() %>%
  dplyr::select (nr, task_id, learner_id, resampling_id, iteration, matches ("classif."))

resample_perf <- resample_perf %>% mutate(learner_id = replace(learner_id, learner_id=="fcnet" & nr==2, "fcnet4"),
                         learner_id = replace(learner_id, learner_id=="fcnet" & nr==3, "fcnet8"),
                         learner_id = replace(learner_id, learner_id=="inception" & nr==5, "inception4"),
                         learner_id = replace(learner_id, learner_id=="inception" & nr==6, "inception8"),
)
  

par(mfrow=c(1,2))
boxplot(resample_perf$classif.acc ~ resample_perf$learner_id)
boxplot(resample_perf$classif.bacc ~ resample_perf$learner_id)

# aggr = bmr$aggregate() # bmr$score()
# print(aggr)
# 
# learners = as.data.table(bmr)$learner
# best_tuning_res <- lapply(1:length(learners), function(i) learners[[i]]$tuning_result)
# 
# tune_res <- extract_inner_tuning_results(bmr)
