rm(list=ls())
library(ParBayesianOptimization)
library(caret)
library(abind)
library(reticulate)
library(xgboost)
library (mltest)
library(purrr)
to_categorical <- function(x, num_classes)
{
  
  model.matrix(~ -1 + ., data = data.frame(x = factor(x, levels=1:num_classes)))
  
}

np <- import ("numpy")

######################################### Gaitrec data ################################################

# load data------------------------------------------------------------------------------
#data <- readRDS("../data/trans_learn.RDS")
data <- readRDS("data/trans_learn.RDS")
gaitrec <- data$gaitrec
gaitrec$grp <- as.factor(gaitrec$grp)
nclasses <- nlevels(gaitrec$grp)
gaitrec$grp <- as.numeric(gaitrec$grp)

# Scale data ----------------------------------------------------------------------------

# Caution, scaling on whole data will result in data leak from train to test
# Trial for now
gaitrec[grepl ("f_", names (gaitrec))] <- map (gaitrec[grepl ("f_", names (gaitrec))],
                                               apply, 2, scale, center = T, scale = F)

# transform and split data
set.seed(2021-7-5)
train.index <- createDataPartition(gaitrec$grp, p = .8, list = FALSE)[,1]
val.index  <- createDataPartition(gaitrec$grp[train.index], p = .8, list = FALSE)[,1]
y_train <- to_categorical(gaitrec$grp[train.index], num_classes = nclasses)
y_test <- to_categorical(gaitrec$grp[-train.index], num_classes = nclasses)

x_gaitrec <- abind(gaitrec[-1], along=3)
x_train <- x_gaitrec[train.index,,]
x_test  <- x_gaitrec[-train.index,,]

y_train_train <- to_categorical(gaitrec$grp[train.index][val.index], num_classes = nclasses)
y_train_val <- to_categorical(gaitrec$grp[train.index][-val.index], num_classes = nclasses)

x_train_train <- x_train[val.index,,]
x_train_val <- x_train[-val.index,,]


############################# Inception Time ##########################################################

# Helpful review for time series classification in deep learning: https://arxiv.org/pdf/1809.04356.pdf
# Most are implemented here: https://github.com/hfawaz/dl-4-tsc
# => clone and wrap (after some modifications)

scoringFunction <- function(
  augmentation_ratio = 4L, # how often to apply augmentation
  jitter=1L,
  scaling=0L,
  permutation=0L,
  randompermutation=0L,
  magwarp=0L,
  timewarp=0L,
  windowslice=0L,
  windowwarp=0L,
  rotation=0L,
  spawner=0L,
  dtwwarp=0L,
  shapedtwwarp=0L,
  wdba=0L,
  discdtw=0L,
  discsdtw=0L,
  depth = 3L,
  nb_filters = 8L,
  x_data_train = x_train_train,
  x_data_val = x_train_val,
  y_data_train = y_train_train,
  y_data_val = y_train_val,
  inp_shape = dim(x_gaitrec)[-1]
){
  
  # args <- mget(names(formals()),sys.frame(sys.nframe()))
  # args <- args[1:16]
  # args[2:16] <- as.logical(args[2:16])
  source("code/list_to_args.R")
  # argsparsed <- list_to_args(as.data.frame(args))
  
  args <- list_to_args(
    augmentation_ratio = as.integer(augmentation_ratio),
    seed=2L,
    jitter=as.logical(jitter),
    scaling=as.logical(scaling),
    permutation=as.logical(permutation),
    randompermutation=as.logical(randompermutation),
    magwarp=as.logical(magwarp),
    timewarp=as.logical(timewarp),
    windowslice=as.logical(windowslice),
    windowwarp=as.logical(windowwarp),
    rotation=as.logical(rotation),
    spawner=as.logical(spawner),
    dtwwarp=as.logical(dtwwarp),
    shapedtwwarp=as.logical(shapedtwwarp),
    wdba=as.logical(wdba),
    discdtw=as.logical(discdtw),
    discsdtw=as.logical(discsdtw),
    extra_tag="",
    dataset="test"
  )
  
  # Clone https://github.com/uchidalab/time_series_augmentation
  # for augmentation and wrap
  
  aug <- import_from_path("augmentation", "code/")
  
  res = aug$run_augmentation(x_data_train, y_data_train, args)
  x_train_aug <- res[[1]]
  y_train_aug <- res[[2]]
  
  inceptionNet <- import_from_path("inceptionnet", path = "code/")
  mod <- inceptionNet$Classifier_INCEPTION(output_directory = "output/inception/",
                                           input_shape = as.integer(inp_shape),
                                           nb_classes = as.integer(nclasses),
                                           verbose = FALSE,
                                           depth = as.integer(depth),
                                           nb_filters = as.integer(nb_filters),
                                           batch_size = 128L,
                                           lr = 0.00001,
                                           nb_epochs = 1000L)
  
  y_pred_in <- mod$fit(x_train_aug, y_train_aug, x_data_val, y_data_val, y_data_val)
  
  # check accuracy
  
  # pred_in <- np$load ("output/inception/y_pred.npy")
  # pred_in <- apply(y_pred_in,1,which.max)-1
  obs <- apply(y_data_val,1,which.max)-1
  
  metrics <- try(ml_test (y_pred_in,
                      obs,
                      output.as.table = TRUE))
  
  if(inherits(metrics, "try-error")){
    return(list(Score = 0))
  }else{
    f1_aug <- mean(metrics$F1)
    acc_aug <- mean(metrics$balanced.accuracy)
    
    return(list(Score = acc_aug)) 
  }

}
    

bounds <- list( 
  augmentation_ratio = c(0L,10L), # how often to apply augmentation
  jitter=c(0L,1L),
  scaling=c(0L,1L),
  permutation=c(0L,1L),
  randompermutation=c(0L,1L),
  magwarp=c(0L,1L),
  timewarp=c(0L,1L),
  windowslice=c(0L,1L),
  windowwarp=c(0L,1L),
  rotation=c(0L,1L),
  spawner=c(0L,1L),
  dtwwarp=c(0L,1L),
  shapedtwwarp=c(0L,1L),
  wdba=c(0L,1L),
  discdtw=c(0L,1L),
  discsdtw=c(0L,1L),
  depth = c(1L,5L),
  nb_filters = c(1L,30L)
)


nrClusters <- 10
nrEpochs <- 10

library(doParallel)
cl <- makeCluster(nrClusters)
registerDoParallel(cl)
clusterExport(cl,c('x_train_train','x_train_val', 'y_train_train', 'y_train_val',
                   'x_gaitrec', 'nclasses'))
clusterEvalQ(cl,expr= {
  library(mltest)
  library(reticulate)
})

set.seed(42)

tWithPar <- system.time(
  optObj <- bayesOpt(
    FUN = scoringFunction,
    bounds = bounds,
    initPoints = nrClusters * 2,
    iters.n = nrClusters * nrEpochs,
    iters.k = nrClusters,
    parallel = TRUE,
    verbose = 2,
    errorHandling = "continue"
  )
)
stopCluster(cl)
registerDoSEQ()

saveRDS(optObj, file="output/gaitrec_.RDS")
print(tWithPar)

best_args <- as.list(getBestPars(optObj)[1,])
best_args$x_data_train = x_train
best_args$x_data_val = x_test
best_args$y_data_train = y_train
best_args$y_data_val = y_test

do.call("scoringFunction", best_args)
