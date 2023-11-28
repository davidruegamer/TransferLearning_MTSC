library(data.table)
library(tidyverse)
library(ggplot2)
library(mlr3)
library(mlr3measures)
library(corrr)
library(cowplot)

# Custom function --------------------------------------------------------------
measures_man <- function(truth, probmat)
{

  c(Metrics:::accuracy(truth, 1*(probmat>0.5)),
    Metrics:::logLoss(truth, probmat),
    Metrics:::auc(truth, probmat),
    Metrics:::mse(truth, probmat)
  )

}


# Load data (no time-series) ---------------------------------------------------

bmr <- readRDS("output/final_result_pfps.RDS")

## Load in additional PFPS lasso results
res_lasso <- readRDS("output/final_result_logreg_pfps.RDS") %>%
  mutate (learner_id = "lasso")

measures <- list (msr("classif.acc"),
                  msr("classif.logloss"),
                  msr("classif.auc"),
                  msr("classif.bbrier"))


# Tidy data -------------------------------------------------------------------

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
) %>%
  dplyr::select (-c(nr, task_id, resampling_id, name))

# Load TL results (time series only)--------------------------------------------

bmr_time_aug <- readRDS("output/resultTL_AUG_pfps.RDS")
bmr_time_aug <- bmr_time_aug %>%
  mutate (value = ifelse (is.nan (value), NA, value))%>%
  group_by(metric, iter, augx) %>%
  summarise (value = mean (value, na.rm = TRUE))%>%
  rename(iteration = iter,
         learner_id = augx) %>%
  mutate (learner_id = case_when(
    learner_id == "0" ~ "Transfer Learned UCR (Augm. x0)",
    learner_id == "2" ~ "Transfer Learned UCR (Augm. x2)",
    learner_id == "4" ~ "Transfer Learned UCR (Augm. x4)",
    learner_id == "8" ~ "Transfer Learned UCR (Augm. x8)",
    TRUE ~ "Transfer Learned UCR (Augm. x12)",
  )) %>%
  mutate (iteration = as.numeric (iteration)) %>%
  mutate (value = ifelse (is.nan (value), NA, value),
          value = ifelse (is.infinite (value), NA, value))

# Merge two datasets -----------------------------------------------------------

df1 <- bind_rows (resample_perf, bmr_time_aug, res_lasso) %>%
  filter (learner_id != "Multinomial Logistic Regression")

new_lvls <- c("lasso",
              "XGBoost (tuned)",
              "FCNet (Augm. x0)",
              "FCNet (Augm. x2)",
              "FCNet (Augm. x4)",
              "FCNet (Augm. x8)",
              "FCNet (Augm. x12)",
              "InceptionTime (Augm. x0)",
              "InceptionTime (Augm. x2)",
              "InceptionTime (Augm. x4)",
              "InceptionTime (Augm. x8)",
              "InceptionTime (Augm. x12)",
              "Transfer Learned UCR (Augm. x0)",
              "Transfer Learned UCR (Augm. x2)",
              "Transfer Learned UCR (Augm. x4)",
              "Transfer Learned UCR (Augm. x8)",
              "Transfer Learned UCR (Augm. x12)",
              "Transfer Learned Imagenet")

new_lvls_names <- c("LASSO",
                    "XG",
                    "FCN(x0)",
                    "FCN(x2)",
                    "FCN(x4)",
                    "FCN(x8)",
                    "FCN(x12)",
                    "IncTime(x0)",
                    "IncTime(x2)",
                    "IncTime(x4)",
                    "IncTime(x8)",
                    "IncTime(x12)",
                    "TL UCR (x0)",
                    "TL UCR (x2)",
                    "TL UCR (x4)",
                    "TL UCR (x8)",
                    "TL UCR (x12)",
                    "TL Imagenet")

df3 <- df1 %>%
  #mutate (learner_id = str_remove (learner_id, " (Augm. x0)")) %>%
  mutate (metric = factor (metric))%>%
  mutate (learner_id = factor (learner_id, levels = new_lvls, labels = new_lvls_names))

# Figure 1 (correlation between metrics) ---------------------------------------
df.plot <- df3 %>%
  pivot_wider(names_from = metric, values_from = value) %>%
  rename(ACC = Accuracy,
         AUC = `Area under ROC`,
         BS = `Brier Score`) %>%
  select (`Log-loss`, ACC, BS, AUC) %>%
  mutate (AUC = ifelse (is.nan (AUC), NA, AUC))

c.plot <- df.plot %>%
  correlate() %>%    # Create correlation data frame (cor_df)
  rearrange() %>%  # rearrange by correlations
  shave()


# Figure 3 ---------------------------------------------------------------------

ll_plot <- df3 %>%
  filter (metric %in% c("Log-loss")) %>%
  filter (!is.na(value))  %>%
  ggplot () +
  geom_boxplot(aes (x =  learner_id, y = value)) +
  cowplot::theme_cowplot() +
  theme(axis.text.x = element_text(angle = 60, vjust = 1, hjust=1)) +
  xlab ("Algorithms") +
  ylab (paste0("Log-loss ", '\u2192', " (worse)"))

br_plot <- df3 %>%
  filter (metric %in% c("Brier Score")) %>%
  filter (!is.na(value))  %>%
  ggplot () +
  geom_boxplot(aes (x =  learner_id, y = value)) +
  cowplot::theme_cowplot() +
  theme(axis.text.x = element_text(angle = 60, vjust = 1, hjust=1)) +
  xlab ("Algorithms") +
  ylab (paste0("Brier Score ", '\u2192', " (worse)"))

auc_plot <- df3 %>%
  filter (metric %in% c("Area under ROC")) %>%
  ggplot () +
  geom_boxplot(aes (x =  learner_id, y = value)) +
  geom_hline(aes(yintercept = 0.7), linetype = 3) +
  cowplot::theme_cowplot() +
  theme(axis.text.x = element_text(angle = 60, vjust = 1, hjust=1)) +
  xlab ("Algorithms") +
  ylab (paste0("AUC ", '\u2192', " (better)")) +
  lims (y = c(0, 1))

acc_plot <- df3 %>%
  filter (metric %in% c("Accuracy")) %>%
  ggplot () +
  geom_boxplot(aes (x =  learner_id, y = value)) +
  geom_hline(aes(yintercept = 0.5), linetype = 3) +
  cowplot::theme_cowplot() +
  theme(axis.text.x = element_text(angle = 60, vjust = 1, hjust=1)) +
  xlab ("Algorithms") +
  ylab (paste0("Accuracy ", '\u2192', " (better)")) +
  lims (y = c(0, 1.0))

p <- plot_grid(
  ll_plot + theme(axis.text.x = element_blank(),
                 axis.title.x = element_blank()),
  br_plot + theme(axis.text.x = element_blank(),
                  axis.title.x = element_blank()),
  auc_plot, acc_plot, labels = "auto", ncol = 2, label_x =0

)

## Plot
tiff ("../manuscript/fig3.tiff", units = "in", height = 10, width = 15, res = 200)
  p
dev.off()

# Probe the best performance----------------------------------------------------

## log loss

temp <- df3 %>%
  filter (metric == "Log-loss") %>%
  group_by(learner_id) %>%
  summarise (Median = median (value, na.rm = TRUE) %>% round (3)) %>%
  arrange (desc (Median))

## Brier

temp <- df3 %>%
  filter (metric == "Brier Score") %>%
  group_by(learner_id) %>%
  summarise (Median = median (value, na.rm = TRUE) %>% round (3)) %>%
  arrange (desc (Median))

## AUC

temp <- df3 %>%
  filter (metric == "Area under ROC") %>%
  group_by(learner_id) %>%
  summarise (Median = median (value, na.rm = TRUE) %>% round (3)) %>%
  arrange (Median)

## Accuracy

temp <- df3 %>%
  filter (metric == "Accuracy") %>%
  group_by(learner_id) %>%
  summarise (Median = median (value) %>% round (3)) %>%
  arrange (desc (Median))

## Compare FCN vs Inception

## log loss

temp <- df3 %>%
  filter (metric == "Log-loss") %>%
  filter (grepl("FCN|Inc", learner_id)) %>%
  group_by(learner_id) %>%
  summarise (Median = median (value))

temp[1:5,2] - temp[6:10,2]

## Weighted Multiclass AUC (1vs1)

temp <- df3 %>%
  filter (metric == "Weighted Multiclass AUC (1vs1)") %>%
  filter (grepl("FCN|Inc", learner_id)) %>%
  group_by(learner_id) %>%
  summarise (Median = median (value))
temp[1:5,2] - temp[6:10,2]

## Compare TL image vs TL UCR

## log loss

temp <- df3 %>%
  filter (metric == "Log-loss") %>%
  filter (grepl("TL", learner_id)) %>%
  group_by(learner_id) %>%
  summarise (Median = median (value))

## Weighted Multiclass AUC (1vs1)

temp <- df3 %>%
  filter (metric == "Weighted Multiclass AUC (1vs1)") %>%
  filter (grepl("TL", learner_id)) %>%
  group_by(learner_id) %>%
  summarise (Median = median (value))
