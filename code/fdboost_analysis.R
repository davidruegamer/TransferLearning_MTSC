rm (list = ls())

# Load packages -----------------------------------------------------------------
# Helper
library(tidyverse)
# Modelling
library(MLmetrics)
library (FDboost)
# Plotting
library (cowplot)
library(gridExtra)
# Parallel
library (furrr)

# Import data ---------------------------------------------------

train <- readRDS("output/train.RDS")
test <- readRDS("output/test.RDS")


# Preprocess ---------------------------------------------------

funvars <- grep ("pro", names(train), value = T)

# Find highly correlated variables
dat <- map_dfc(train[funvars ], ~ matrix (., ncol = 1) %>% as.numeric)
corM = cor (dat)
hc_fun <- caret::findCorrelation(corM, cutoff = 0.9) # make it high to keep more variables.
hc_fun <- names (dat)[hc_fun]

funvars <- setdiff(funvars, hc_fun)
scalvars <- grep ("age", names(train), value = T)
all_vars <- c(funvars, scalvars)

train[all_vars] <- map (train[all_vars], scale, center = T, scale = F)
train$class_label <- factor (train$class_label, levels = c("HC", "C", "A", "K", "H"))
train$Sample = c(1:101) # index of time
train$age = as.numeric (train$age)


# Boosting ---------------------------------------------------

## Modelling

### Formula
train$class <- factor(levels(train$class_label)[-1]) # make HC as control
lhs = "class_label ~ "

fvars = funvars # or funvars or msvars

rhs_fun <-  paste(paste0("bsignal(", fvars, ", s = Sample, differences = 1, df = 2)", "%O%",
                       "bols(class, df = 2, contrasts.arg = 'contr.dummy')", collapse = " + "))

rhs = paste(rhs_fun,
            sep = "+")
# transform the whole string to a formula
form = as.formula( paste0(lhs, rhs))

mod <- FDboost(
  formula = form,
  data = train ,
  timeformula = NULL,
  family = Multinomial(),
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
           type = "kfold", B = 10, strata = train$class_label)

# DR>
cvr = cvrisk(mod,folds , grid = 1:20000)
plot (cvr)

(best_iteration = mstop(cvr))
mod[best_iteration]


## Print variable importance

print (plot(mboost::varimp(mod)))

## OOB model performance
class_name <- c("HC", "C",  "A" ,  "K" ,  "H" )

pd <- apply(predict(mod, type = "response"), 1, which.max)
pd <- factor (pd, levels = as.character (1:5), labels = class_name)

# Performance

Accuracy (pd, train$class_label)

boxplot(pd ~ df.list$grp)
ModelMetrics::auc(actual = df.list$grp, predicted = pd)
(oob_auc <- unlist(cvrisk(mod, folds = folds, fun =
                            function(object) ModelMetrics::auc(actual = df.list$grp,
                                                               predict(object, type = "response")))))
(mean(oob_auc))

