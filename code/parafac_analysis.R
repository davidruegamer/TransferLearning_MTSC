# Import library ------------------------------------------------

rm (list = ls())
library (tidyverse)
library (multiway)
library (abind)
library (doParallel)
library (foreach)

# Import data ---------------------------------------------------

train <- readRDS("output/train.RDS")
test <- readRDS("output/test.RDS")

# Data restructure ----------------------------------------------

subj_lvls <- as.character(train$subject_id)

X <- train  %>%
  keep (is.matrix) %>%
  abind (along = 3)  %>%
  aperm (c(2,1,3))

attr(X, "dimnames")[[2]] <- interaction(train$subject_id, train$class_label, sep = "_")


# Preprocess data -----------------------------------------------

Xs <- nscale(X, mode=3, ssnew = prod(dim(X)[-3])) # Remove SD of waveform

dimnames(Xs) <- dimnames(X)

# Parafac analysis -----------------------------------------------

n <-  6

# fit model for different numbers of factors
doParallel::registerDoParallel(5)

pfaclist <- foreach (k = 1:n, .packages = "multiway") %dopar%{

  parafac(Xs, nfac = k, const = c("uncons","uncons","orthog"), nstart = 500)

}

saveRDS(pfaclist, "output/pfaclist.RDS")

# Performance -----------------------------------------------------


vaf_list = c()
for (k in 1:n){
  s = sumsq(Xs)
  pfac_fit = fitted (pfaclist[[k]])
  s1 = sumsq(Xs-pfac_fit)
  vaf = (s-s1)/s *100
  vaf_list[k] = vaf
}
vaf_list
plot(1:n,
     vaf_list,
     type = "b",
     xlab = "nFactors",
     ylab = "VAF")

# Results ----------------------------------------------------------

f <- 3

pfac = pfaclist[[f]]

pfac = rescale (pfac, mode = "C", absorb = "B")
pfac = resign (resign(pfac, mode="B", absorb="A"), mode="C", absorb="A")

pfacB <- data.frame (
  subject_id = train$subject_id,
  class_label = train$class_label,
  pfac$B)

# Predict B weights test from the pfac of train dataset----------------------------------------------

Xt <- test  %>%
  keep (is.matrix) %>%
  abind (along = 3)  %>%
  aperm (c(2,1,3))

attr(Xt, "dimnames")[[2]] <- interaction(test$subject_id, test$class_label, sep = "_")


Xts <- nscale(Xt, mode=3, ssnew = prod(dim(Xt)[-3])) # Remove SD of waveform

dimnames(Xts) <- dimnames(Xt)


pfacC = as.matrix (pfac$C[,])
pfacA = as.matrix (pfac$A[,])

Z = krprod (pfacC,pfacA)

output <- matrix(ncol=f, nrow= length (test$subject_id))

for (i in seq_along (test$subject_id)){
  subj <- as.vector (Xts[,i,])
  a <- t(subj) %*% Z %*% solve(t(Z) %*% Z)

  output[i,] <- a
}

colnames (output) = c("fac1", "fac2")
test = as.data.frame(output)
