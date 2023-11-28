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

# Check correlation between metrics

tiff ("../manuscript/fig1.tiff", units = "in", height = 4, width = 4, res = 300)
  df3 %>%
    pivot_wider(names_from = metric, values_from = value) %>%
    select(Accuracy:`Multiclass Brier Score`) %>%
    rename(ACC = Accuracy,
           BACC = `Balanced Accuracy`,
           WAUC1 = `Weighted Multiclass AUC (1vs1)`,
           AAUC1 = `Average Multiclass AUC (1vs1)`,
           WAUC4 = `Weighted Multiclass AUC (1vsAll)`,
           AAUC4 = `Average Multiclass AUC (1vsAll)`,
           MBS = `Multiclass Brier Score`) %>%
    select (`Log-loss`, ACC, MBS, WAUC1, AAUC1) %>%
    cor %>% corrplot(
      method = 'square', order = 'AOE', addCoef.col = 'black', tl.pos = 'd',
      cl.pos = 'n' #, col = COL2('BrBG')
    )
dev.off()

# Probe the best performance

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


## Plot
tiff ("../manuscript/fig2.tiff", units = "in", height = 7, width = 10, res = 200)

  df3 %>%
    filter (metric %in% c("Log-loss", "Weighted Multiclass AUC (1vs1)",
                          "Average Multiclass AUC (1vsAll)",
                          "Balanced Accuracy", "Multiclass Brier Score")) %>%
    mutate (metric = factor (metric, levels = c("Log-loss", "Multiclass Brier Score",
                                                "Weighted Multiclass AUC (1vs1)",
                                                "Average Multiclass AUC (1vsAll)",
                                                "Balanced Accuracy"))) %>%
    ggplot () +
      geom_boxplot(aes (x =  learner_id, y = value)) +
      facet_wrap(~ metric, ncol = 2, scales = "free_y") +
      cowplot::theme_cowplot() +
      theme(axis.text.x = element_text(angle = 60, vjust = 1, hjust=1)) +
      xlab ("Algorithms") +
      ylab ("Value")

dev.off()

