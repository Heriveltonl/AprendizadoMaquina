#Regressao Biomassa

library(mlbench)
library(caret)
library(mice)
library(Metrics)
library(kernlab)

r2 <- function(predito, observado) {
  return(1 - (sum((predito-observado)^2) / sum((observado-mean(observado))^2)))
}

syx <- function(predito, observado,data,p){
  n <- nrow(data)
  return(sqrt(sum((observado-predito)^2))/(n-p))
}

### Leitura dos dados
setwd("/home/herivelton/Documentos/IAA/")

dados <- read.csv("Biomassa.csv")
#View(dados)

### Cria um arquivo com 80% das linhas para treino e 20% para teste
set.seed(202483)
ind <- createDataPartition(dados$biomassa, p=0.80, list = FALSE)
treino <- dados[ind,]
teste <- dados[-ind,]

####################################################################################
##### Treinamento KNN
tuneGrid <- expand.grid(k = c(1,3,5,7,9))
knn <- train(biomassa~ ., data = treino, method = "knn",tuneGrid=tuneGrid)
knn
predicoes.knn <- predict(knn, teste)
### Metricas
r2(predicoes.knn,teste$biomassa)
rmse(teste$biomassa, predicoes.knn)
MAE <- mean(abs(teste$biomassa - predicoes.knn))
MAE 
Pearson <- cor(teste$biomassa, predicoes.knn)
Pearson
syx(teste$biomassa, predicoes.knn,dados,3)

####################################################################################
#### Treinamento de Redes Neurais com HoldOut
rna <- train(biomassa~., data=treino, method="nnet", linout=T, trace=FALSE)
rna
predicoes.rna <- predict(rna, teste)
## Metricas
r2(predicoes.rna,teste$biomassa)
rmse(teste$biomassa, predicoes.rna)
MAE <- mean(abs(teste$biomassa - predicoes.rna))
MAE 
Pearson <- cor(teste$biomassa, predicoes.rna)
Pearson
syx(teste$biomassa, predicoes.rna,dados,3)

#### Treinamento de Redes Neurais com CrossValidation
control <- trainControl(method = "cv", number = 10)
tuneGrid <- expand.grid(size = seq(from = 1, to = 10, by = 1), decay = seq(from = 0.1, to = 0.9, by = 0.3))
rna_cv <- train(biomassa~., data=treino, method="nnet", trainControl=control, tuneGrid=tuneGrid, linout=T,MaxNWts=10000, maxit=2000, trace=F)
rna_cv
predicoes.rna_cv <- predict(rna_cv, teste)
r2(predicoes.rna_cv,teste$biomassa)
rmse(teste$biomassa, predicoes.rna_cv)
MAE <- mean(abs(teste$biomassa - predicoes.rna_cv))
MAE 
Pearson <- cor(teste$biomassa, predicoes.rna_cv)
Pearson
syx(teste$biomassa, predicoes.rna_cv,dados,3)

####################################################################################
#### Treinamento de SVM com HoldOut
svm <- train(biomassa~., data=treino, method="svmRadial")
svm
predicoes.svm <- predict(svm, teste)
# Calcular metricas
r2(predicoes.svm,teste$biomassa)
rmse(teste$biomassa, predicoes.svm)
MAE <- mean(abs(teste$biomassa - predicoes.svm))
MAE 
Pearson <- cor(teste$biomassa, predicoes.svm)
Pearson
syx(teste$biomassa, predicoes.svm,dados,3)

#### Treinamento de SVM com Cross-validation
ctrl <- trainControl(method = "cv", number = 10)
svm_cv <- train(biomassa~., data=treino, method="svmRadial", trControl=ctrl)
svm_cv
predicoes.svm_cv <- predict(svm_cv, teste)
##Metricas
r2(predicoes.svm_cv,teste$biomassa)
rmse(teste$biomassa, predicoes.svm_cv)
MAE <- mean(abs(teste$biomassa - predicoes.svm_cv))
MAE 
Pearson <- cor(teste$biomassa, predicoes.svm_cv)
Pearson
syx(teste$biomassa, predicoes.svm_cv,dados,3)

####################################################################################
### Treinar Random Forest com HoldOut
rf <- train(biomassa~., data=treino, method="rf")
rf
predicoes.rf <- predict(rf, teste)

## Metricas
r2(predicoes.rf,teste$biomassa)
rmse(teste$biomassa, predicoes.rf)
MAE <- mean(abs(teste$biomassa - predicoes.rf))
MAE 
Pearson <- cor(teste$biomassa, predicoes.rf)
Pearson
syx(teste$biomassa, predicoes.rf,dados,3)


### Treinar Random Forest com HoldOut Cross-validation
ctrl <- trainControl(method = "cv", number = 10)
rf_cv <- train(biomassa~., data=treino, method="rf", trControl=ctrl)
rf_cv
predicoes.rf_cv <- predict(rf_cv, teste)
##Metricas
r2(predicoes.rf_cv,teste$biomassa)
rmse(teste$biomassa, predicoes.rf_cv)
MAE <- mean(abs(teste$biomassa - predicoes.rf_cv))
MAE 
Pearson <- cor(teste$biomassa, predicoes.rf_cv)
Pearson
syx(teste$biomassa, predicoes.rf_cv,dados,3)



#### Casos novos
dados_novos <- read.csv("Biomassa_novos.csv")
#View(dados_novos)

predicoes.svm_novos <- round(predict(svm, dados_novos),2)
resultado <- cbind(dados_novos, predicoes.svm_novos)
resultado$biomassa <- NULL
#View(resultado)
write.csv(resultado,"saida_Bioamassa_novos.csv")

###Tabela de residuos 

r2_knn <- r2(predicoes.knn, teste$biomassa)
r2_rna <- r2(predicoes.rna, teste$biomassa)
r2_rna_cv <- r2(predicoes.rna_cv, teste$biomassa)
r2_svm <- r2(predicoes.svm, teste$biomassa)
r2_svm_cv <- r2(predicoes.svm_cv, teste$biomassa)
r2_rf <- r2(predicoes.rf, teste$biomassa)
r2_rf_cv <- r2(predicoes.rf_cv, teste$biomassa)

r2_values <- c(r2_knn, r2_rna, r2_rna_cv, r2_svm, r2_svm_cv, r2_rf, r2_rf_cv)
names(r2_values) <- c("KNN", "RNA", "RNA_CV", "SVM", "SVM_CV", "RF", "RF_CV")

# Técnica com maior R2
best_technique <- names(which.max(r2_values))
best_r2 <- max(r2_values)
best_technique
best_r2

# Calcular resíduos
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

residuos <- teste$biomassa - predicoes_best

# Plotar gráfico de resíduos
plot(predicoes_best, residuos, 
     main = paste("Gráfico de Resíduos -", best_technique, "com R² =", round(best_r2, 2)), 
     xlab = "Valores Preditos", 
     ylab = "Resíduos", 
     pch = 20, col = "blue")
abline(h = 0, col = "red", lwd = 2)