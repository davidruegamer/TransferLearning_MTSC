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

### check metrics
ggplot(resample_perf %>% filter(
  !metric %in% c("Weighted Multiclass AUC (1vsAll)", "Average Multiclass AUC (1vsAll)")
  ), aes(x = learner_id, y = value, fill = learner_id)) +
  geom_boxplot() + facet_wrap(~ metric, scales = "free_y") +
  theme_bw() + theme() + xlab("") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  guides(fill="none")

ggsave(width=8, height=5, file="results.pdf")

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

ggplot(preds_long, aes(fill=learner_id, y=value, x=name)) + 
  geom_hline(yintercept = 0.2, linetype=1, alpha=0.3) + 
  geom_boxplot(outlier.size = 0.01) +
  theme_bw() + theme(legend.title = element_blank()) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  xlab("") + ylab("Predicted Probability")

ggsave(width=8, height=5, file="preds.pdf")
