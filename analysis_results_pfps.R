library(data.table)
library(tidyverse)
library(ggplot2)
library(mlr3)
library(mlr3measures)
library(corrplot)

bmr <- readRDS("output/final_result_pfps.RDS")

measures <- list (msr("classif.acc"),
                  msr("classif.logloss"),
                  msr("classif.auc"),
                  msr("classif.mbrier"))

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
                                                          classif.logloss = "Log-loss",
                                                          classif.auc = "Area under ROC",
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

### check metrics
ggplot(resample_perf %>% dplyr::select(learner_id, value, iteration, metric),
       aes(x = learner_id, y = value, fill = learner_id)) +
  geom_boxplot() + facet_wrap(~ metric, scales = "free_y") +
  theme_bw() + theme() + xlab("") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  guides(fill="none")

ggsave(width=6, height=5, file="results_pfps.pdf")

