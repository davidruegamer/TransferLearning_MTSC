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

# Load data (no time-series) ---------------------------------------------------

bmr <- readRDS("output/final_result.RDS")

measures <- list (msr("classif.acc"),
                  msr("classif.bacc"),
                  msr("classif.logloss"),
                  msr("classif.mauc_au1p"),
                  msr("classif.mauc_au1u"),
                  msr("classif.mauc_aunp"),
                  msr("classif.mauc_aunu"),
                  msr("classif.mbrier"))

# Tiday data -------------------------------------------------------------------

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

# Load TL results (time series only)--------------------------------------------

res_TL <- readRDS("output/resultTL_AUG.RDS")

res_TL$augx <- factor(res_TL$augx, levels=c("0", "2", "4", "8", "12"))


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

# Merge two datasets -----------------------------------------------------------

df1 <- resample_perf %>%
  dplyr::select (-c(nr, task_id, name, resampling_id))

df2 <- perf_TL %>%
  group_by(learner_id, metric, iteration) %>%
  summarize (value = mean (value)) %>%
  mutate(iteration = as.numeric (iteration))

new_lvls <- c("Multinomial Logistic Regression",
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

new_lvls_names <- c("mLogReg",
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
  bind_rows(df2) %>%
  #mutate (learner_id = str_remove (learner_id, " (Augm. x0)")) %>%
  mutate (metric = factor (metric))%>%
  mutate (learner_id = factor (learner_id, levels = new_lvls, labels = new_lvls_names))

# Figure 1 (correlation between metrics) ---------------------------------------
df.plot <- df3 %>%
  pivot_wider(names_from = metric, values_from = value) %>%
  select(Accuracy:`Multiclass Brier Score`) %>%
  rename(ACC = Accuracy,
         BACC = `Balanced Accuracy`,
         WAUC1 = `Weighted Multiclass AUC (1vs1)`,
         AAUC1 = `Average Multiclass AUC (1vs1)`,
         WAUC4 = `Weighted Multiclass AUC (1vsAll)`,
         AAUC4 = `Average Multiclass AUC (1vsAll)`,
         MBS = `Multiclass Brier Score`) %>%
  select (`Log-loss`, ACC, MBS, WAUC1, AAUC1)

c.plot <- df.plot %>%
  correlate() %>%    # Create correlation data frame (cor_df)
  rearrange() %>%  # rearrange by correlations
  shave()

# Custom plot function ---------------------------------------------------------

my_rplot <- function (rdf, legend = TRUE, shape = 16, colours = c("indianred2",
                                                      "white", "skyblue1"), print_cor = FALSE, colors, .order = c("default",
                                                                                                                  "alphabet"))
{
  .order <- match.arg(.order)
  if (!missing(colors)) {
    colours <- colors
  }
  row_order <- rdf$term
  pd <- stretch(rdf, na.rm = TRUE)
  pd$size <- abs(pd$r)
  pd$label <- as.character(fashion(pd$r))
  if (.order == "default") {
    pd$x <- factor(pd$x, levels = row_order)
    pd$y <- factor(pd$y, levels = rev(row_order))
  }
  plot_ <- list(geom_point(shape = shape, size = 10),
                if (print_cor) geom_text(color = "black", size = 3, alpha = 1,show.legend = TRUE),
                scale_colour_gradientn(limits = c(-1,1), colors = colours),
                theme_classic(), labs(x = "", y = ""), guides(size = "none", alpha = "none"),
                if (legend) labs(colour = NULL),
                if (!legend) theme(legend.position = "none"))
  ggplot(pd, aes_string(x = "x", y = "y", color = "r",
                        alpha = "size", label = "label")) + plot_
}

p <- my_rplot(c.plot, print_cor = TRUE, legend = TRUE)
p

tiff ("../manuscript/fig1.tiff", units = "in", height = 3, width = 5, res = 300)
p
dev.off()

# Figure 2 ---------------------------------------------------------------------

ll_plot <- df3 %>%
  filter (metric %in% c("Log-loss")) %>%
  ggplot () +
  geom_boxplot(aes (x =  learner_id, y = value)) +
  cowplot::theme_cowplot() +
  theme(axis.text.x = element_text(angle = 60, vjust = 1, hjust=1)) +
  xlab ("Algorithms") +
  ylab (paste0("Log-loss ", '\u2192', " (worse)")) +
  lims (y = c(1, 3))

br_plot <- df3 %>%
  filter (metric %in% c("Multiclass Brier Score")) %>%
  ggplot () +
  geom_boxplot(aes (x =  learner_id, y = value)) +
  cowplot::theme_cowplot() +
  theme(axis.text.x = element_text(angle = 60, vjust = 1, hjust=1)) +
  xlab ("Algorithms") +
  ylab (paste0("Brier Score ", '\u2192', " (worse)")) +
  lims (y = c(0.5, 1.2))

wauc_plot <- df3 %>%
  filter (metric %in% c("Weighted Multiclass AUC (1vs1)")) %>%
  ggplot () +
  geom_boxplot(aes (x =  learner_id, y = value)) +
  geom_hline(aes(yintercept = 0.7), linetype = 3) +
  cowplot::theme_cowplot() +
  theme(axis.text.x = element_text(angle = 60, vjust = 1, hjust=1)) +
  xlab ("Algorithms") +
  ylab (paste0("Weighted AUC ", '\u2192', " (better)")) +
  lims (y = c(0.5, 0.9))

aauc_plot <- df3 %>%
  filter (metric %in% c("Average Multiclass AUC (1vs1)")) %>%
  ggplot () +
  geom_boxplot(aes (x =  learner_id, y = value)) +
  geom_hline(aes(yintercept = 0.7), linetype = 3) +
  cowplot::theme_cowplot() +
  theme(axis.text.x = element_text(angle = 60, vjust = 1, hjust=1)) +
  xlab ("Algorithms") +
  ylab (paste0("Average AUC ", '\u2192', " (better)")) +
  lims (y = c(0.5, 0.9))

acc_plot <- df3 %>%
  filter (metric %in% c("Accuracy")) %>%
  ggplot () +
  geom_boxplot(aes (x =  learner_id, y = value)) +
  geom_hline(aes(yintercept = 0.5), linetype = 3) +
  cowplot::theme_cowplot() +
  theme(axis.text.x = element_text(angle = 60, vjust = 1, hjust=1)) +
  xlab ("Algorithms") +
  ylab (paste0("Accuracy ", '\u2192', " (better)")) +
  lims (y = c(0.2, 0.7))

p <- plot_grid(
  ll_plot + theme(axis.text.x = element_blank(),
                 axis.title.x = element_blank()),
  br_plot + theme(axis.text.x = element_blank(),
                  axis.title.x = element_blank()),
  wauc_plot+ theme(axis.text.x = element_blank(),
                   axis.title.x = element_blank()),
  aauc_plot, acc_plot, labels = "auto", ncol = 3, label_x =0

)

## Plot
tiff ("../manuscript/fig2.tiff", units = "in", height = 10, width = 15, res = 200)
  p
dev.off()

# Probe the best performance----------------------------------------------------

## log loss

temp <- df3 %>%
  filter (metric == "Log-loss") %>%
  group_by(learner_id) %>%
  summarise (Median = median (value) %>% round (3)) %>%
  arrange (desc (Median))

## Weighted  Multiclass AUC (1vs1)

temp <- df3 %>%
  filter (metric == "Weighted Multiclass AUC (1vs1)") %>%
  group_by(learner_id) %>%
  summarise (Median = median (value) %>% round (3)) %>%
  arrange (desc (Median))

## Multiclass Brier Score

temp <- df3 %>%
  filter (metric == "Multiclass Brier Score") %>%
  group_by(learner_id) %>%
  summarise (Median = median (value) %>% round (3)) %>%
  arrange (Median)

## Balanced Accuracy

temp <- df3 %>%
  filter (metric == "Balanced Accuracy") %>%
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
