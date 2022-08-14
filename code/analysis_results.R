library(data.table)
library(tidyverse)
library(ggplot2)
library(mlr3)

reticulate::use_condaenv("mlr3keras", required = TRUE)

bmr <- readRDS("output/final_result.RDS")

measures <- list (msr("classif.acc"),
                  msr("classif.bacc"),
                  msr("classif.logloss"),
                  msr("classif.mauc_au1p"),
                  msr("classif.mauc_au1u"),
                  msr("classif.mauc_aunp"),
                  msr("classif.mauc_aunu"),
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
    learner_id = replace(learner_id, learner_id=="flatfunct.classif.glmnet", "Multinomial Logistic Regression")
)

resample_perf <- resample_perf %>% pivot_longer(classif.acc:classif.bacc)
resample_perf <- resample_perf %>% mutate(metric = recode(name, 
                                                          classif.acc = "Accuracy",
                                                          classif.bacc = "Balanced Accuracy")
)

resample_perf$learner_id <- factor(resample_perf$learner_id, 
                                   levels = 
                                     c("FCNet (Augm. x0)", "FCNet (Augm. x2)", "FCNet (Augm. x4)", 
                                       "FCNet (Augm. x8)", "FCNet (Augm. x12)", "InceptionTime (Augm. x0)", 
                                       "InceptionTime (Augm. x2)", "InceptionTime (Augm. x4)", 
                                       "InceptionTime (Augm. x8)", "InceptionTime (Augm. x12)", 
                                       "XGBoost (tuned)", "Multinomial Logistic Regression"
                                     )
)

ggplot(resample_perf, aes(x = learner_id, y = value, fill = learner_id)) + 
  geom_boxplot() + facet_grid(~ metric) + 
  theme_bw() + theme() + xlab("") + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) + 
  guides(fill="none")

ggsave(width=8, height=5, file="results.pdf")

## summary in numbers:

resample_perf %>% filter(metric=="Accuracy") %>% 
  group_by(learner_id) %>% 
  summarise(mean_value = mean(value),
            median_value = median(value)) %>% 
  arrange(-mean_value)

resample_perf %>% filter(metric=="Balanced Accuracy") %>% 
  group_by(learner_id) %>% 
  summarise(mean_value = mean(value),
            median_value = median(value)) %>% 
  arrange(-mean_value)
