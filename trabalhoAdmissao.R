#Regressão Admissao
#install.packages("mlbench")
#install.packages("mice")
#install.packages("Metrics")
#install.packages("kernlab")
#install.packages("e1071")
#install.packages("ggplot2")
#install.packages("caret")

library(Metrics)
library("caret")
library(mlbench)
library(caret)
library(mice)
library(Metrics)
library(kernlab)
library(stats)
library(ggplot2)

# Definição da Função que mede o Rquadrado
r2 <- function(predito, observado) {
  return(1 - (sum((predito-observado)^2) / sum((observado-mean(observado))^2)))
}

syx <- function(predito, observado,data,p){
  n <- nrow(data)
  return(sqrt(sum((observado-predito)^2))/(n-p))
}


### Leitura dos dados
setwd("/home/herivelton/Documentos/IAA/")

dados <- read.csv("Admissao_Dados.csv")

### Retira o atributo ID da Base de dados
dados$num <- NULL

### Cria um arquivo com 80% das linhas para treino e 20% para teste
set.seed(202483)
ind <- createDataPartition(dados$ChanceOfAdmit, p=0.80, list = FALSE)
treino <- dados[ind,]
teste <- dados[-ind,]

### Prepara um grid com os valores de k que
### serão usados
tuneGrid <- expand.grid(k = c(1,3,5,7,9))

### Executa o Knn com esse grid

knn <- train(ChanceOfAdmit ~ ., data = treino, method = "knn",tuneGrid=tuneGrid)
knn
predicoes.knn <- predict(knn, teste)
### Calcula as métricas para verificar a qualidade do Modelos
r2(predicoes.knn,teste$ChanceOfAdmit)
rmse(teste$ChanceOfAdmit, predicoes.knn)
MAE <- mean(abs(teste$ChanceOfAdmit - predicoes.knn))
MAE 
Pearson <- cor(teste$ChanceOfAdmit, predicoes.knn)
Pearson
syx(teste$ChanceOfAdmit, predicoes.knn,dados,8)

#################################################################################
##### Treinamento de Redes Neurais com HoldOut
rna <- train(ChanceOfAdmit~., data=treino, method="nnet", linout=T, trace=FALSE)
rna
predicoes.rna <- predict(rna, teste)
### Calcula as métricas para verificar a qualidade do Modelo RNA com HoldOut
r2(predicoes.rna, teste$ChanceOfAdmit)
rmse(teste$ChanceOfAdmit, predicoes.rna)
MAE <- mean(abs(teste$ChanceOfAdmit - predicoes.rna))
MAE 
Pearson <- cor(teste$ChanceOfAdmit, predicoes.rna)
Pearson
syx(teste$ChanceOfAdmit, predicoes.rna,dados,8)
##### Treinamento de Redes Neurais com CrossValidation
control <- trainControl(method = "cv", number = 10)
tuneGrid_rna_cv <- expand.grid(size = seq(from = 1, to = 10, by = 1), decay = seq(from = 0.1, to = 0.9, by = 0.3))
rna_cv <- train(ChanceOfAdmit~., data=treino, method="nnet", trainControl=control, tuneGrid=tuneGrid_rna_cv, linout=T,MaxNWts=10000, maxit=2000, trace=F)
rna_cv
predicoes.rna_cv <- predict(rna_cv, teste)
### Calcula as métricas para verificar a qualidade do Modelo RNA com CrossValidation
r2(predicoes.rna_cv, teste$ChanceOfAdmit)
rmse(teste$ChanceOfAdmit, predicoes.rna_cv)
MAE <- mean(abs(teste$ChanceOfAdmit - predicoes.rna_cv))
MAE 
Pearson <- cor(teste$ChanceOfAdmit, predicoes.rna_cv)
Pearson
syx(teste$ChanceOfAdmit,predicoes.rna_cv,dados,8)
#################################################################################
##### Treinamento de SVM com HoldOut
svm <- train(ChanceOfAdmit~., data=treino, method="svmRadial")
svm
predicoes.svm <- predict(svm, teste)
### Calcula as métricas para verificar a qualidade do Modelo SVM com HoldOut
rmse(teste$ChanceOfAdmit, predicoes.svm)
r2(predicoes.svm,teste$ChanceOfAdmit)
MAE <- mean(abs(teste$ChanceOfAdmit - predicoes.svm))
MAE 
Pearson <- cor(teste$ChanceOfAdmit, predicoes.svm)
Pearson
syx(teste$ChanceOfAdmit, predicoes.svm,dados,8)
##### Treinamento de SVM com CrossValidation
ctrl <- trainControl(method = "cv", number = 10)
svm_cv <- train(ChanceOfAdmit~., data=treino, method="svmRadial", trControl=ctrl)
svm_cv
predicoes.svm_cv <- predict(svm_cv, teste)
### Calcula as métricas para verificar a qualidade do Modelo SVM com CrossValidation
r2(predicoes.svm_cv,teste$ChanceOfAdmit)
rmse(teste$ChanceOfAdmit, predicoes.svm_cv)
MAE <- mean(abs(teste$ChanceOfAdmit - predicoes.svm_cv))
MAE 
Pearson <- cor(teste$ChanceOfAdmit, predicoes.svm_cv)
Pearson
syx(teste$ChanceOfAdmit, predicoes.svm_cv,dados,8)
#################################################################################
##### Treinamento de Random Forest com HoldOut
rf <- train(ChanceOfAdmit~., data=treino, method="rf")
rf
predicoes.rf <- predict(rf, teste)
### Calcula as métricas para verificar a qualidade do Modelo Random Forrest com HoldOut 
r2(predicoes.rf,teste$ChanceOfAdmit)
rmse(teste$ChanceOfAdmit, predicoes.rf)
MAE <- mean(abs(teste$ChanceOfAdmit - predicoes.rf))
MAE 
Pearson <- cor(teste$ChanceOfAdmit, predicoes.rf)
Pearson
syx(teste$ChanceOfAdmit, predicoes.rf,dados,8)
##### Treinamento de Random Forest com CrossValidation
ctrl <- trainControl(method = "cv", number = 10)
rf_cv <- train(ChanceOfAdmit~., data=treino, method="rf", trControl=ctrl)
rf_cv
predicoes.rf_cv <- predict(rf_cv, teste)
### Calcula as métricas para verificar a qualidade do Modelo Random Forrest com CrossValidation
r2(predicoes.rf_cv ,teste$ChanceOfAdmit)
rmse(teste$ChanceOfAdmit, predicoes.rf_cv)
MAE <- mean(abs(teste$ChanceOfAdmit - predicoes.rf_cv))
MAE 
Pearson <- cor(teste$ChanceOfAdmit, predicoes.rf_cv)
Pearson
syx(teste$ChanceOfAdmit, predicoes.rf_cv,dados,8)


#### Casos novos
dados_novos <- read.csv("Admissao_Dados_novos.csv")
dados_novos$num <- NULL

predicoes.rf_cv_novos <- round(predict(rf_cv, dados_novos),2)
result <- cbind(dados_novos, predicoes.rf_cv_novos)
result$ChanceOfAdmit <- NULL

write.csv(result,"saida_Admissao_novos.csv")

# Calcular R2
r2_knn <- r2(predicoes.knn, teste$ChanceOfAdmit)
r2_rna <- r2(predicoes.rna, teste$ChanceOfAdmit)
r2_rna_cv <- r2(predicoes.rna_cv, teste$ChanceOfAdmit)
r2_svm <- r2(predicoes.svm, teste$ChanceOfAdmit)
r2_svm_cv <- r2(predicoes.svm_cv, teste$ChanceOfAdmit)
r2_rf <- r2(predicoes.rf, teste$ChanceOfAdmit)
r2_rf_cv <- r2(predicoes.rf_cv, teste$ChanceOfAdmit)

#Tabela com os resultados
r2_values <- c(r2_knn, r2_rna, r2_rna_cv, r2_svm, r2_svm_cv, r2_rf, r2_rf_cv)
names(r2_values) <- c("KNN", "RNA", "RNA_CV", "SVM", "SVM_CV", "RF", "RF_CV")

#Ttécnica com maior R2
best_technique <- names(which.max(r2_values))
best_r2 <- max(r2_values)
best_technique
best_r2

# Calcular resíduos e valores preditos com base na melhor técnica
if (best_technique == "KNN") {
  predicoes_best <- predicoes.knn
} else if (best_technique == "RNA") {
  predicoes_best <- predicoes.rna
} else if (best_technique == "RNA_CV") {
  predicoes_best <- predicoes.rna_cv
} else if (best_technique == "SVM") {
  predicoes_best <- predicoes.svm
} else if (best_technique == "SVM_CV") {
  predicoes_best <- predicoes.svm_cv
} else if (best_technique == "RF") {
  predicoes_best <- predicoes.rf
} else if (best_technique == "RF_CV") {
  predicoes_best <- predicoes.rf_cv
}

residuos <- teste$ChanceOfAdmit - predicoes_best

# Plotar gráfico de resíduos
plot(predicoes_best, residuos, 
     main = paste("Gráfico de Resíduos -", best_technique, "com R² =", round(best_r2, 2)), 
     xlab = "Valores Preditos", 
     ylab = "Resíduos", 
     pch = 20, col = "blue")
abline(h = 0, col = "red", lwd = 2)
