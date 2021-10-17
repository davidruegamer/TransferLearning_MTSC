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


######################################### Gaitrec data ################################################

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

################################# PFPS data ###############################################
pfps$grp <- as.factor(pfps$grp)
nclasses <- nlevels(pfps$grp)
pfps$grp <- as.numeric(pfps$grp) - 1

# transform and split data (same as before)
set.seed(2021-7-5)
train.index <- createDataPartition(pfps$grp, p = .5, list = FALSE)
y_train <- to_categorical(pfps$grp[train.index], num_classes = nclasses)
y_test <- to_categorical(pfps$grp[-train.index], num_classes = nclasses)
x_pfps <- abind(pfps[-7], along=3)
x_train <- x_pfps[train.index,,]
x_test  <- x_pfps[-train.index,,]
############################## InceptionTime ##############################################

mod <- inceptionNet$Classifier_INCEPTION(output_directory = "output/inception/pfps/",
                                         input_shape = dim(x_pfps)[-1],
                                         nb_classes = 2,
                                         verbose = TRUE,
                                         depth = 3L,
                                         nb_filters = 8L,
                                         batch_size = 128L,
                                         lr = 0.00001)

y_pred_in <- mod$fit(x_train, y_train, x_test, y_test, y_test)

# check accuracy

pred_in <- np$load ("output/inception/pfps/y_pred.npy")
pred_in <- apply(pred_in,1,which.max)-1
obs <- apply(y_test,1,which.max)-1

MLmetrics::Accuracy(pred_in, obs)

##################################### transfer learn network ##############################


# transer learn the previous model
tl <- import_from_path("transfer_net", path = "code/")
mod <- tl$load_freeze_transfer("output/fcn/", nclasses, lr = 0.01)
mod$fit(x = x_train, y = y_train, epochs = 500L, verbose = TRUE)

y_pred <- mod$predict(x_test)
cbind(y_test[,2], y_pred[,2])
# accuracy
y_pred  <- apply(y_pred,1,which.max)-1

MLmetrics::Accuracy(y_pred, y_test[,2])
MLmetrics::AUC(y_pred, y_test[,2])


# accuracy on all data points
y_pred_all <- mod$predict(abind(list(x_train, x_test), along=1))

y_pred_all  <- apply(y_pred_all,1,which.max)-1

MLmetrics::Accuracy(y_pred_all, c(y_train[,2], y_test[,2]))
MLmetrics::AUC(y_pred_all, c(y_train[,2], y_test[,2]))

mean(c(y_train[,2], y_test[,2]) ==
       (mod$predict(abind(list(x_train, x_test), along=1))[,2]>0.5))

######################################## LASSO ###########################################

d_flat <- do.call("cbind", pfps[-7])
y_d <- pfps$grp

set.seed(42)
cvl1 <- cv.glmnet(x = d_flat, y = y_d, family="binomial")
plot(cvl1)

# this is in-sample!
y_pred <- plogis(predict(cvl1, newx = d_flat, s = "lambda.min"))
# accuracy on all data points
mean(y_d==(y_pred[,1]>0.5))
# on the previous test set
mean(y_d[train.index]==(y_pred[train.index,1]>0.5))

################################## FDboost ############################################


funvars <- grep ("f_", names(pfps), value = T)

train <- pfps %>%
  map_if (~is.null(dim(.)) & length(.) != 101, ~ .[train.index ]) %>%
  map_if (is.matrix,  ~ as.matrix(.[train.index ,]))

test <-  pfps %>%
  map_if (~is.null(dim(.)) & length(.) != 101, ~ .[-train.index ]) %>%
  map_if (is.matrix,  ~ as.matrix(.[-train.index ,]))

train$Sample = c(1:101) # index of time


lhs = "grp ~ "

fvars = funvars # or funvars or msvars

rhs_fun <-  paste(paste0("bsignal(", funvars, ", s = Sample, differences = 1, df = 1)", collapse = " + "))

rhs = paste(rhs_fun,
            sep = "+")
# transform the whole string to a formula
form = as.formula( paste0(lhs, rhs))

mod <- FDboost(
  formula = form,
  data = train ,
  timeformula = NULL,
  family = Binomial(),
  # define the boosting specific parameters
  # mstop:  set to a large number, such as 1000.
  #         We will choose the optimal stoppting
  #         iteration in the next step via cross-validation
  # nu:     learning rate: 0.1 is a good default, which
  #         works for most cases
  control = boost_control(mstop = 10000,
                          nu = .1))

## Tuning to get mstop

set.seed(2019-11-22) # better set seed for repoducibility
folds = cv(rep(1, length(unique(mod$id))),
           type = "kfold", B = 10, strata = train$grp)

# DR>
cvr = cvrisk(mod,folds , grid = 1:20000)
plot (cvr)

(best_iteration = mstop(cvr))
mod[best_iteration]

pd <- predict(mod, newdata = test, type = "class")
prob <- predict(mod, newdata = test, type = "response")
MLmetrics::Accuracy(pd, test$grp)
MLmetrics::AUC(prob , as.numeric (test$grp) -1)
