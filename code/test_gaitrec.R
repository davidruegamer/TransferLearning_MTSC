rm(list=ls())

library(caret)
library(abind)
library(keras)
library(reticulate)
library(xgboost)
library(glmnet)
library (mltest)
library (FDboost)
library(purrr)

np <- import ("numpy")

######################################### Gaitrec data ################################################

# load data------------------------------------------------------------------------------
#data <- readRDS("../data/trans_learn.RDS")
data <- readRDS("data/trans_learn.RDS")
gaitrec <- data$gaitrec
gaitrec$grp <- as.factor(gaitrec$grp)
nclasses <- nlevels(gaitrec$grp)
gaitrec$grp <- as.numeric(gaitrec$grp) - 1

# Scale data ----------------------------------------------------------------------------

# Caution, scaling on whole data will result in data leak from train to test
# Trial for now
gaitrec[grepl ("f_", names (gaitrec))] <- map (gaitrec[grepl ("f_", names (gaitrec))],
                                               apply, 2, scale, center = T, scale = F)


pfps <- data$pfps
pfps[grepl ("f_", names (pfps))] <- map (pfps[grepl ("f_", names (pfps))],
                                         apply, 2, scale, center = T, scale = F)

# transform and split data
set.seed(2021-7-5)
train.index <- createDataPartition(gaitrec$grp, p = .8, list = FALSE)
y_train <- to_categorical(gaitrec$grp[train.index], num_classes = nclasses)
y_test <- to_categorical(gaitrec$grp[-train.index], num_classes = nclasses)

x_gaitrec <- abind(gaitrec[-1], along=3)
x_train <- x_gaitrec[train.index,,]
x_test  <- x_gaitrec[-train.index,,]


######################################### Augmentation ################################################

# Clone https://github.com/uchidalab/time_series_augmentation
# for augmentation and wrap

aug <- import_from_path("augmentation", "code/")
source("code/list_to_args.R")
args <- list_to_args(
  augmentation_ratio = 4L, # how often to apply augmentation
  jitter=TRUE,
  scaling=FALSE,
  permutation=FALSE,
  randompermutation=FALSE,
  magwarp=FALSE,
  timewarp=FALSE,
  windowslice=FALSE,
  windowwarp=FALSE,
  rotation=FALSE,
  spawner=FALSE,
  dtwwarp=FALSE,
  shapedtwwarp=FALSE,
  wdba=FALSE,
  discdtw=FALSE,
  discsdtw=FALSE
)

res = aug$run_augmentation(x_train, y_train, args)
x_train_aug <- res[[1]]
y_train_aug <- res[[2]]


############################# Inception Time ##########################################################

# Helpful review for time series classification in deep learning: https://arxiv.org/pdf/1809.04356.pdf
# Most are implemented here: https://github.com/hfawaz/dl-4-tsc
# => clone and wrap (after some modifications)

inceptionNet <- import_from_path("inceptionnet", path = "code/")
mod <- inceptionNet$Classifier_INCEPTION(output_directory = "output/inception/",
                                         input_shape = dim(x_gaitrec)[-1],
                                         nb_classes = nclasses,
                                         verbose = TRUE,
                                         depth = 3L,
                                         nb_filters = 8L,
                                         batch_size = 128L,
                                         lr = 0.00001,
                                         nb_epochs = 750L)

y_pred_in <- mod$fit(x_train, y_train, x_test, y_test, y_test)

# check accuracy

# pred_in <- np$load ("output/inception/y_pred.npy")
# pred_in <- apply(y_pred_in,1,which.max)-1
obs <- apply(y_test,1,which.max)-1

metrics <- ml_test (y_pred_in,
                    obs,
                    output.as.table = TRUE)

f1_wo_aug <- mean(metrics$F1)
acc_wo_aug <- mean(metrics$balanced.accuracy)

## with augmentation

mod <- inceptionNet$Classifier_INCEPTION(output_directory = "output/inception/",
                                         input_shape = dim(x_gaitrec)[-1],
                                         nb_classes = nclasses,
                                         verbose = TRUE,
                                         depth = 3L,
                                         nb_filters = 8L,
                                         batch_size = 128L,
                                         lr = 0.00001,
                                         nb_epochs = 750L)

y_pred_in <- mod$fit(x_train_aug, y_train_aug, x_test, y_test, y_test)

# check accuracy

# pred_in <- np$load ("output/inception/y_pred.npy")
# pred_in <- apply(y_pred_in,1,which.max)-1
obs <- apply(y_test,1,which.max)-1

metrics <- ml_test (y_pred_in,
                    obs,
                    output.as.table = TRUE)

f1_w_aug <- mean(metrics$F1)
acc_w_aug <- mean(metrics$balanced.accuracy)

cbind(c("w/o Aug", "w/ Aug"), F1=round(c(f1_wo_aug, f1_w_aug),3), 
      Accuracy=round(c(acc_wo_aug, acc_w_aug),3))

############################# Full convolutional network #############################################
fcn <- import_from_path("fcn", path = "code/")
mod <- fcn$Classifier_FCN(output_directory = "output/fcn/",
                          input_shape = dim(x_gaitrec)[-1],
                          nb_classes = nclasses,
                          verbose = TRUE,
                          filters = 32,
                          lr = 0.00001)

y_pred_fcn <- mod$fit(x_train, y_train, x_test, y_test, y_test, batch_size = 64)

# check accuracy
mod_fcn <- load_model_hdf5("output/fcn/best_model.hdf5")
pred_fcn <- mod_fcn$predict(x_test)
pred_fcn <- apply(pred_fcn,1,which.max)-1

ml_test (pred_fcn,
         obs,
         output.as.table = TRUE)

############################# Xgboost #############################################

d_flat <- do.call("cbind", gaitrec[-1])
d_train <- d_flat[train.index,]
d_test <- d_flat[-train.index,]
y_d_train <- gaitrec$grp[train.index]
y_d_test <- gaitrec$grp[-train.index]

xgb_data <- xgb.DMatrix(data = model.matrix(~ ., data = d_train),
                        label = matrix(y_d_train, ncol=1))
xgb_data_test <- xgb.DMatrix(data = model.matrix(~ ., data = d_test),
                             label = matrix(y_d_test, ncol=1))


xgb_mod <- xgb.train(data = xgb_data,
                     early_stopping_rounds = 20,
                     nrounds = 5000,
                     params = list(objective = "multi:softprob",
                                   # eval_metric = c("auc"),
                                   num_class = nclasses),
                     watchlist = list(eval = xgb_data_test),
                     verbose = TRUE)

# extract fit and prediction
xgb_pred <- matrix(predict(xgb_mod, newdata = xgb_data_test), ncol = nclasses, byrow = T)
pred_xgb <- apply(xgb_pred,1,which.max)-1

ml_test (pred_xgb,
         obs,
         output.as.table = TRUE)