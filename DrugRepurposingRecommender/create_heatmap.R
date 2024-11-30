rm(list=ls())
library(dplyr)
library(pheatmap)
library(stringr)
library(tidyr)
library(dplyr)
library(tibble)
library(ggplot2)
library(RColorBrewer)

getwd()
setwd("Documents/CS_Courses/Needle/Project/")

repodb = read.csv("repodb_for_final_visualization_medical_profile.csv", row.names = 1)
repodb[repodb==""] <- "Not Tested"

drugs = factor(colnames(repodb), levels = colnames(repodb))
diseases = factor(rownames(repodb),levels = rownames(repodb))
long = data.frame(pivot_longer(repodb %>%  rownames_to_column("Disease"), cols = c(-"Disease"), 
                               names_to = "Drug"))


colors = c( "darkgreen","seashell", "lightpink1", "lightpink2", "lightpink3",
           "lightpink3", "lightpink4", "lightpink4", "brown1","brown2", "brown3", "brown3",
           "brown4", "brown4", "steelblue1", "steelblue2","steelblue3", "steelblue3",
           "steelblue4", "steelblue4")
names(colors) = sort(unique(long$value))


long_arranged  = long %>% 
  arrange(factor(Disease, levels = levels(diseases)),
          factor(Drug, levels = levels(drugs)))

ggplot(long_arranged, aes(Disease, Drug)) + 
  geom_tile(aes(fill = value))+
  theme(axis.text.x=element_blank(),
        axis.text.y=element_blank()) + 
  scale_fill_manual(values=colors) +
  labs(fill = "Drug Status")

ggsave("drugs_diseases_map_medical_profiles_3.png")

