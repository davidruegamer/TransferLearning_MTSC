library(caret)
library(abind)
library(keras)
library(reticulate)
library(xgboost)
library(glmnet)

# load data
data <- readRDS("../data/trans_learn.RDS")
gaitrec <- data$gaitrec
gaitrec$grp <- as.factor(gaitrec$grp)
nclasses <- nlevels(gaitrec$grp)
gaitrec$grp <- as.numeric(gaitrec$grp) - 1
pfps <- data$pfps

# transform and split data
set.seed(2021-7-5)
train.index <- createDataPartition(gaitrec$grp, p = .8, list = FALSE)
y_train <- to_categorical(gaitrec$grp[train.index], num_classes = nclasses)
y_test <- to_categorical(gaitrec$grp[-train.index], num_classes = nclasses)

x_gaitrec <- abind(gaitrec[-1], along=3)
x_train <- x_gaitrec[train.index,,]
x_test  <- x_gaitrec[-train.index,,]

# Helpful review for time series classification in deep learning: https://arxiv.org/pdf/1809.04356.pdf
# Most are implemented here: https://github.com/hfawaz/dl-4-tsc
# => clone and wrap (after some modifications)

inceptionNet <- import("inceptionnet")
mod <- inceptionNet$Classifier_INCEPTION(output_directory = "../output/inception/", 
                                         input_shape = dim(x_gaitrec)[-1], 
                                         nb_classes = nclasses, 
                                         verbose = TRUE,
                                         depth = 3L,
                                         nb_filters = 8L,
                                         batch_size = 128L,
                                         lr = 0.00001)

y_pred_in <- mod$fit(x_train, y_train, x_test, y_test, y_test)

# test another model
fcn <- import("fcn")
mod <- fcn$Classifier_FCN(output_directory = "../output/fcn/", 
                          input_shape = dim(x_gaitrec)[-1], 
                          nb_classes = nclasses, 
                          verbose = TRUE, 
                          filters = 32,
                          lr = 0.00001)

y_pred_fcn <- mod$fit(x_train, y_train, x_test, y_test, y_test, batch_size = 64)

# check accuracy
tt <- table(y_pred_fcn, apply(y_test,1,which.max)-1)
sum(diag(tt))/length(y_pred_fcn)

### compare to a strong ML baseline

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
tp <- table(apply(xgb_pred, 1, which.max)-1, y_d_test)
# is it really so bad?
sum(diag(tp))/length(xgb_pred)

### transfer learn network

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

# transer learn the previous model
tl <- import("transfer_net")
mod <- tl$load_freeze_transfer("../output/fcn/", nclasses, lr = 0.01) 
mod$fit(x = x_train, y = y_train, epochs = 500L, verbose = TRUE)

y_pred <- mod$predict(x_test)
cbind(y_test[,2], y_pred[,2])
# accuracy 
mean(y_test[,2] == (y_pred[,2]>0.5))
# accuracy on all data points
mean(c(y_train[,2], y_test[,2]) == 
       (mod$predict(abind(list(x_train, x_test), along=1))[,2]>0.5))

### check again against strong baseline

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
