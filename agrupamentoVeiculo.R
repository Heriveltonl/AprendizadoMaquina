
#install.packages("mlbench")
library(mlbench)

#install.packages("mice") 
library(mice)

## para o kmodes
#install.packages("klaR")
library(klaR)

library(ggplot2)


setwd("/home/herivelton/Documentos/IAA/")

dados <- read.csv("Veiculos.csv")
#View(dados)

set.seed(202483)

dados$a <- NULL

cluster.results <- kmodes(dados, 10, iter.max = 10, weighted = FALSE ) 
cluster.results

resultado <- cbind(dados, cluster.results$cluster)
resultado

write.csv(resultado,"saida_Cluster_Agrupamento.csv")
