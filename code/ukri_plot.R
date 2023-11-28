resample_perf %>%
  filter(metric =="Average Multiclass AUC (1vs1)"  )%>%
ggplot(aes(x = learner_id, y = value, fill = learner_id)) +
  geom_boxplot() +
  cowplot::theme_cowplot() + theme() + xlab("") + ylab ("Average Multiclass AUC (1vs1)" ) +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1),
        axis.text = element_text(size = 14, face = "bold")) +
  guides(fill="none")

ggsave(width=8, height=5, file="results.png")
