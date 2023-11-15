library(data.table)
library(tidyverse)
library(ggplot2)
library(mlr3)
library(mlr3measures)
library(Metrics)
library(corrplot)

bmr <- readRDS("output/final_result_pfps.RDS")

measures <- list (msr("classif.acc"),
                  msr("classif.logloss"),
                  msr("classif.auc"),
                  msr("classif.bbrier"))


# for manual evaluation
measures_man <- function(truth, probmat)
{
  
  c(Metrics:::accuracy(truth, 1*(probmat>0.5)),
    Metrics:::logLoss(truth, probmat),
    Metrics:::auc(truth, probmat),
    Metrics:::mse(truth, probmat)
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

resample_perf <- resample_perf %>% pivot_longer(classif.acc:classif.bbrier)
resample_perf <- resample_perf %>% mutate(metric = recode(name,
                                                          classif.acc = "Accuracy",
                                                          classif.logloss = "Log-loss",
                                                          classif.auc = "Area under ROC",
                                                          classif.bbrier = "Brier Score"
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
if(!file.exists("output/resultTL_AUG_pfps.RDS")){
  lf <- list.files("output/TL_AUG", full.names = T, pattern = ".*_pfps\\.csv")
  res_TL <- do.call("rbind", lapply(lf, function(fln){
    
    rr <- read.csv(fln)[,-1]
    rres <- measures_man(rr$truth, rr$prob)
    return(data.frame(
      metric = unique(resample_perf$metric),
      value = rres,
      dataset = gsub("output/TL_AUG/(.*)\\_aug\\_x(.*)\\_fold\\_([0-9])_pfps\\.csv", "\\1", fln),
      iter = gsub("output/TL_AUG/(.*)\\_aug\\_x(.*)\\_fold\\_([0-9])_pfps\\.csv", "\\3", fln),
      augx = gsub("output/TL_AUG/(.*)\\_aug\\_x(.*)\\_fold\\_([0-9])_pfps\\.csv", "\\2", fln)
    ))
    
  }))
  saveRDS(res_TL, "output/resultTL_AUG_pfps.RDS")
}else{
  res_TL <- readRDS("output/resultTL_AUG_pfps.RDS")
}

res_TL$augx <- factor(res_TL$augx, levels=c("0", "2", "4", "8", "12"))

ggplot(res_TL, aes(x = dataset, y = value, fill = dataset)) +
  geom_boxplot() + facet_wrap(~ metric, scales = "free_y") +
  theme_bw() + theme() + xlab("") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  guides(fill="none")

ggsave(width=8, height=5, file="results_TL_pfps.pdf")

ggplot(res_TL %>% filter(
  metric %in% c("Log-loss")
), aes(x = dataset, y = value, fill = augx)) +
  geom_boxplot() + 
  theme_bw() + theme() + xlab("") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  guides(fill="none")

ggsave(width=8, height=5, file="results_TL_logloss_pfps.pdf")

ggplot(res_TL, aes(x = augx, y = value, fill = dataset)) +
  geom_boxplot() + facet_wrap(~metric, scales="free_y") + 
  theme_bw() + theme() + xlab("") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  guides(fill="none")

ggsave(width=8, height=5, file="results_TL_metric_per_augx_pfps.pdf")

ggplot(res_TL %>% filter(
  metric %in% c("Accuracy")
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
       aes(x=`Accuracy`, y=-`Log-loss`, colour = augx)) + 
  geom_point() + theme_bw()


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
# resample_perf %>% 
#   select(-name) %>% 
#   pivot_wider(names_from = metric, values_from = value) %>% 
#   select(Accuracy:`Brier Score`) %>% 
#   rename(ACC = Accuracy, 
#          AUC = `Area under ROC`,
#          MBS = `Brier Score`) %>%  
#   cor %>% corrplot(
#     method = 'square', order = 'AOE', addCoef.col = 'black', tl.pos = 'd',
#     cl.pos = 'n' #, col = COL2('BrBG')
#   )

### check metrics
ggplot(resample_perf %>% dplyr::select(learner_id, value, iteration, metric) %>% 
         rbind(perf_TL %>% select(learner_id, value, iteration, metric)), 
       aes(x = learner_id, y = value, fill = learner_id)) +
  geom_boxplot() + facet_wrap(~ metric, scales = "free_y") +
  theme_bw() + theme() + xlab("") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  guides(fill="none")

ggsave(width=9, height=5, file="results_pfps.pdf")

# alpha=0.25
# 
# med_lr <- resample_perf %>% dplyr::select(learner_id, value, iteration, metric) %>% 
#   filter(learner_id == "Multinomial Logistic Regression") %>% 
#   filter(
#     metric %in% c("Log-loss", "Accuracy")
#   ) %>% 
#   group_by(metric) %>% 
#   summarise(median(value)) %>% c
# 
# perf_TL_best <- perf_TL
# 
# perf_TL_best[perf_TL_best$metric=="Log-loss","value"] <-
#   - perf_TL_best[perf_TL_best$metric=="Log-loss","value"]
# 
# perf_TL_best <- 
#   perf_TL_best %>% group_by(learner_id, metric, iteration) %>% 
#   summarise(value = max(value))
# 
# perf_TL_best[perf_TL_best$metric=="Log-loss","value"] <-
#   - perf_TL_best[perf_TL_best$metric=="Log-loss","value"]
# 
# cross_data <- resample_perf %>% 
#   dplyr::select(learner_id, value, iteration, metric) %>% 
#   rbind(perf_TL_best) %>% 
#   filter(
#     metric %in% c("Log-loss", "Accuracy")
#   ) %>% 
#   pivot_wider(names_from = metric, values_from = value) %>% 
#   group_by(learner_id) %>% 
#   summarise(ymin = quantile(`Accuracy`, alpha), 
#             ymax = quantile(`Accuracy`, 1-alpha),
#             xmin = -quantile(`Log-loss`, alpha),
#             xmax = -quantile(`Log-loss`, 1-alpha),
#             ymean = median(`Accuracy`),
#             xmean = -median(`Log-loss`)) %>% 
#   mutate(xmax = pmax(xmax, -1.75)) 
# 
# ggplot() +
#   geom_rect(data = data.frame(x = 0, y = 0),
#             aes(xmin=-1.8, xmax=-med_lr[[2]][2],
#                 ymin=0.3, ymax=med_lr[[2]][1]), alpha=0.1) +
#   geom_point(data=cross_data, aes(x = xmean, y = ymean, group = learner_id, colour = learner_id)) +
#   geom_errorbar(data=cross_data, aes(x = xmean, y = ymean, ymin = ymin, ymax = ymax, colour = learner_id)) +
#   geom_errorbarh(data=cross_data, aes(x = xmean, y = ymean, xmin = xmin, xmax = xmax, colour = learner_id)) +
#   ylab("Accuracy") + xlab("Negative Log-loss") +
#   theme_bw() +
#   theme(legend.title = element_blank())
# 
# ggsave(width=6, height=5, file="results2_pfps.pdf")

