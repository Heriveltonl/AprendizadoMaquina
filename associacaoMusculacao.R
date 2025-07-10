### Instalação dos pacotes necessários
#install.packages('arules', dep=T)
library(arules)
library(datasets)

### Leitura dos dados
setwd("/home/herivelton/Documentos/IAA/")

dados <- read.csv("Musculacao.csv", sep=";")
#View(dados)

#summary(dados)

set.seed(202483)

rules <- apriori(dados, parameter = list(supp = 0.001, conf = 0.7, minlen=2))
#summary(rules)

options(digits=2)
#inspect(sort(rules, by=c("confidence","support")))

rules_df <- as(rules,"data.frame")
write.csv(rules_df,"regras_Associacao.csv")

