library(data.table)
library(tidyverse)
library(ggplot2)
library(mlr3)
library(mlr3measures)
library(corrplot)

bmr <- readRDS("output/final_result.RDS")

measures <- list (msr("classif.acc"),
                  msr("classif.bacc"),
                  msr("classif.logloss"),
                  msr("classif.mauc_au1p"),
                  msr("classif.mauc_au1u"),
                  msr("classif.mauc_aunp"),
                  msr("classif.mauc_aunu"),
                  msr("classif.mbrier"))

# for manual evaluation
measures_man <- function(truth, probmat)
{
  
  c(mlr3measures::acc(truth, factor(apply(probmat, 1, which.max),
                                    levels = 1:5)),
    mlr3measures::bacc(truth, factor(apply(probmat, 1, which.max),
                                        levels = 1:5)),
    mlr3measures::logloss(truth, probmat),
    mlr3measures::mauc_au1p(truth, probmat),
    mlr3measures::mauc_au1u(truth, probmat),
    mlr3measures::mauc_aunp(truth, probmat),
    mlr3measures::mauc_aunu(truth, probmat),
    mlr3measures::mbrier(truth, probmat)
  )
  
}

resample_perf <- as.data.table (bmr$score(measures = measures)) %>%
  as.data.frame() %>%
  dplyr::select (nr, task_id, learner_id, resampling_id, iteration, matches ("classif."))

resample_perf <- resample_perf %>% 
  mutate(
    learner_id = replace(learner_id, learner_id=="classif.xgboost.tuned", "XGBoost (tuned)"),
    learner_id = replace(learner_id, learner_id=="fcnet" & nr==1, "FCNet (Augm. x0)"),
    learner_id = replace(learner_id, learner_id=="fcnet" & nr==2, "FCNet (Augm. x2)"),
    learner_id = replace(learner_id, learner_id=="fcnet" & nr==3, "FCNet (Augm. x4)"),
    learner_id = replace(learner_id, learner_id=="fcnet" & nr==4, "FCNet (Augm. x8)"),
    learner_id = replace(learner_id, learner_id=="fcnet" & nr==5, "FCNet (Augm. x12)"),
    learner_id = replace(learner_id, learner_id=="inception" & nr==6, "InceptionTime (Augm. x0)"),
    learner_id = replace(learner_id, learner_id=="inception" & nr==7, "InceptionTime (Augm. x2)"),
    learner_id = replace(learner_id, learner_id=="inception" & nr==8, "InceptionTime (Augm. x4)"),
    learner_id = replace(learner_id, learner_id=="inception" & nr==9, "InceptionTime (Augm. x8)"),
    learner_id = replace(learner_id, learner_id=="inception" & nr==10, "InceptionTime (Augm. x12)"),
    learner_id = replace(learner_id, learner_id=="flatfunct.classif.glmnet", "Multinomial Logistic Regression"),
    learner_id = replace(learner_id, learner_id=="classif.keras", "Transfer Learned Imagenet")
)

resample_perf <- resample_perf %>% pivot_longer(classif.acc:classif.mbrier)
resample_perf <- resample_perf %>% mutate(metric = recode(name,
                                                          classif.acc = "Accuracy",
                                                          classif.bacc = "Balanced Accuracy",
                                                          classif.logloss = "Log-loss",   
                                                          classif.mauc_au1p = "Weighted Multiclass AUC (1vs1)",
                                                          classif.mauc_au1u = "Average Multiclass AUC (1vs1)",
                                                          classif.mauc_aunp = "Weighted Multiclass AUC (1vsAll)",
                                                          classif.mauc_aunu = "Average Multiclass AUC (1vsAll)", 
                                                          classif.mbrier = "Multiclass Brier Score"
                                                          )
)

resample_perf$learner_id <- factor(resample_perf$learner_id,
                                   levels =
                                     c("FCNet (Augm. x0)", "FCNet (Augm. x2)", "FCNet (Augm. x4)",
                                       "FCNet (Augm. x8)", "FCNet (Augm. x12)", "InceptionTime (Augm. x0)",
                                       "InceptionTime (Augm. x2)", "InceptionTime (Augm. x4)",
                                       "InceptionTime (Augm. x8)", "InceptionTime (Augm. x12)",
                                       "Transfer Learned Imagenet",
                                       "XGBoost (tuned)", "Multinomial Logistic Regression"
                                     )
)

### load TL results
if(!file.exists("output/resultTL_AUG.RDS")){
  lf <- list.files("output/TL_AUG", full.names = T)
  res_TL <- do.call("rbind", lapply(lf, function(fln){
    
    rr <- read.csv(fln)[,-1]
    probmat <- as.matrix(rr[,1:5])
    colnames(probmat) <- 1:5
    truth <- as.factor(rr$truth)
    rres <- measures_man(truth, probmat)
    return(data.frame(
      metric = unique(resample_perf$metric),
      value = rres,
      dataset = gsub("output/TL_AUG/(.*)\\_aug\\_x(.*)\\_fold\\_([0-9])\\.csv", "\\1", fln),
      iter = gsub("output/TL_AUG/(.*)\\_aug\\_x(.*)\\_fold\\_([0-9])\\.csv", "\\3", fln),
      augx = gsub("output/TL_AUG/(.*)\\_aug\\_x(.*)\\_fold\\_([0-9])\\.csv", "\\2", fln)
    ))
    
  }))
  saveRDS(res_TL, "output/resultTL_AUG.RDS")
}else{
  res_TL <- readRDS("output/resultTL_AUG.RDS")
}

res_TL$augx <- factor(res_TL$augx, levels=c("0", "2", "4", "8", "12"))

ggplot(res_TL %>% filter(
  !metric %in% c("Weighted Multiclass AUC (1vsAll)", "Average Multiclass AUC (1vsAll)")
), aes(x = dataset, y = value, fill = dataset)) +
  geom_boxplot() + facet_wrap(~ metric, scales = "free_y") +
  theme_bw() + theme() + xlab("") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  guides(fill="none")

ggsave(width=8, height=5, file="results_TL.pdf")

ggplot(res_TL %>% filter(
  metric %in% c("Log-loss")
), aes(x = dataset, y = value, fill = augx)) +
  geom_boxplot() + 
  theme_bw() + theme() + xlab("") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  guides(fill="none")

ggsave(width=8, height=5, file="results_TL_logloss.pdf")

ggplot(res_TL %>% filter(
  !metric %in% c("Weighted Multiclass AUC (1vsAll)", "Average Multiclass AUC (1vsAll)")
), aes(x = augx, y = value, fill = dataset)) +
  geom_boxplot() + facet_wrap(~metric, scales="free_y") + 
  theme_bw() + theme() + xlab("") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  guides(fill="none")

ggsave(width=8, height=5, file="results_TL_metric_per_augx.pdf")

ggplot(res_TL %>% filter(
  metric %in% c("Balanced Accuracy")
), aes(x = dataset, y = value, colour = dataset)) +
  geom_point(aes(shape = augx)) + facet_wrap(~ metric, scales = "free_y") +
  theme_bw() + theme() + xlab("") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  guides(colour="none")

res_TL %>% group_by(dataset, metric) %>% 
  summarize(value = mean(value)) %>% 
  ggplot(aes(x = metric, y = value)) + geom_boxplot()

agg_res_TL <- res_TL %>% group_by(dataset, metric, augx) %>% 
  summarize(value = mean(value))

ggplot(agg_res_TL %>% pivot_wider(names_from = c(metric), 
                                  values_from = value), 
       aes(x=`Balanced Accuracy`, y=-`Log-loss`, colour = augx)) + 
  geom_point() + theme_bw()

# bestDataset <- res_TL %>% group_by(dataset, metric, augx) %>% 
#   summarize(value = mean(value)) %>% filter(metric=="Balanced Accuracy") %>% 
#   arrange(-value) %>% pull(dataset, augx)
# 
# perf_TL <- res_TL %>% 
#   filter(dataset==bestDataset[2]) %>% 
#   dplyr::select(metric, value, iter) %>% 
#   rename(iteration = iter)

perf_TL <- res_TL %>% 
  rename(iteration = iter) %>% 
  mutate(
    learner_id = paste0("Transfer Learned UCR (Augm. x", augx, ")")
  )

perf_TL$learner_id <- factor(perf_TL$learner_id,
                                   levels =
                                     c("Transfer Learned UCR (Augm. x0)",
                                       "Transfer Learned UCR (Augm. x2)", "Transfer Learned UCR (Augm. x4)",
                                       "Transfer Learned UCR (Augm. x8)", "Transfer Learned UCR (Augm. x12)"
                                     )
)

### check correlation between metrics
resample_perf %>% 
  select(-name) %>% 
  pivot_wider(names_from = metric, values_from = value) %>% 
  select(Accuracy:`Multiclass Brier Score`) %>% 
  rename(ACC = Accuracy, 
         BACC = `Balanced Accuracy`,
         WAUC1 = `Weighted Multiclass AUC (1vs1)`,
         AAUC1 = `Average Multiclass AUC (1vs1)`,
         WAUC4 = `Weighted Multiclass AUC (1vsAll)`,
         AAUC4 = `Average Multiclass AUC (1vsAll)`,
         MBS = `Multiclass Brier Score`) %>%  
  cor %>% corrplot(
    method = 'square', order = 'AOE', addCoef.col = 'black', tl.pos = 'd',
    cl.pos = 'n' #, col = COL2('BrBG')
  )

# use MBS, ACC, WAUC1  

### check metrics
ggplot(resample_perf %>% dplyr::select(learner_id, value, iteration, metric) %>% 
         rbind(perf_TL %>% select(learner_id, value, iteration, metric)) %>% 
         filter(
           !metric %in% c("Weighted Multiclass AUC (1vsAll)", "Average Multiclass AUC (1vsAll)", 
                          "Average Multiclass AUC (1vs1)", "Accuracy")
           ), 
  aes(x = learner_id, y = value, fill = learner_id)) +
  geom_boxplot() + facet_wrap(~ metric, scales = "free_y") +
  theme_bw() + theme() + xlab("") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  guides(fill="none")

ggsave(width=6, height=5, file="results.pdf")

alpha=0.25

med_lr <- resample_perf %>% dplyr::select(learner_id, value, iteration, metric) %>% 
  filter(learner_id == "Multinomial Logistic Regression") %>% 
  filter(
    metric %in% c("Log-loss", "Balanced Accuracy")
  ) %>% 
  group_by(metric) %>% 
  summarise(median(value)) %>% c

perf_TL_best <- perf_TL

perf_TL_best[perf_TL_best$metric=="Log-loss","value"] <-
  - perf_TL_best[perf_TL_best$metric=="Log-loss","value"]

perf_TL_best <- 
  perf_TL_best %>% group_by(learner_id, metric, iteration) %>% 
  summarise(value = max(value))

perf_TL_best[perf_TL_best$metric=="Log-loss","value"] <-
  - perf_TL_best[perf_TL_best$metric=="Log-loss","value"]

cross_data <- resample_perf %>% 
                    dplyr::select(learner_id, value, iteration, metric) %>% 
  rbind(perf_TL_best) %>% 
  filter(
    metric %in% c("Log-loss", "Balanced Accuracy")
  ) %>% 
  pivot_wider(names_from = metric, values_from = value) %>% 
  group_by(learner_id) %>% 
  summarise(ymin = quantile(`Balanced Accuracy`, alpha), 
            ymax = quantile(`Balanced Accuracy`, 1-alpha),
            xmin = -quantile(`Log-loss`, alpha),
            xmax = -quantile(`Log-loss`, 1-alpha),
            ymean = median(`Balanced Accuracy`),
            xmean = -median(`Log-loss`)) %>% 
  mutate(xmax = pmax(xmax, -1.75)) 

ggplot() + 
  geom_rect(data = data.frame(x = 0, y = 0), 
            aes(xmin=-1.8, xmax=-med_lr[[2]][2], 
                ymin=0.3, ymax=med_lr[[2]][1]), alpha=0.1) + 
  geom_point(data=cross_data, aes(x = xmean, y = ymean, group = learner_id, colour = learner_id)) + 
  geom_errorbar(data=cross_data, aes(x = xmean, y = ymean, ymin = ymin, ymax = ymax, colour = learner_id)) + 
  geom_errorbarh(data=cross_data, aes(x = xmean, y = ymean, xmin = xmin, xmax = xmax, colour = learner_id)) + 
  ylab("Balanced Accuracy") + xlab("Negative Log-loss") + 
  theme_bw() +
  theme(legend.title = element_blank())

ggsave(width=6, height=5, file="results2.pdf")


### check predictions
preds <- do.call("rbind", lapply(1:length(bmr$resample_results$resample_result), function(learner){
  preds <- bmr$resample_results$resample_result[[learner]]$predictions()
  cbind(do.call("rbind", lapply(1:length(preds), function(i) cbind(as.data.table(preds[[i]]), 
                                                             fold=i))), 
        learner_id = levels(resample_perf$learner_id)[learner])
}))

preds_long <- preds %>% pivot_longer(prob.1:prob.5)
preds_long <- preds_long %>% mutate(name = recode(name,
                                                  prob.1 = "Class 1",
                                                  prob.2 = "Class 2",
                                                  prob.3 = "Class 3",
                                                  prob.4 = "Class 4",
                                                  prob.5 = "Class 5")
)
                                                          
ggplot(preds_long, aes(fill=name, y=value, x=learner_id)) + 
  geom_hline(yintercept = 0.2, linetype=1, alpha=0.3) + 
  geom_boxplot(outlier.size = 0.01) +
  theme_bw() + theme(legend.title = element_blank()) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("") + ylab("Predicted Probability")

ggsave(width=8, height=5, file="preds.pdf")

ggplot(preds_long, aes(fill=learner_id, y=value, x=name)) + 
  geom_hline(yintercept = 0.2, linetype=1, alpha=0.3) + 
  geom_boxplot(outlier.size = 0.01) +
  theme_bw() + theme(legend.title = element_blank()) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("") + ylab("Predicted Probability")

ggsave(width=8, height=5, file="preds2.pdf")
