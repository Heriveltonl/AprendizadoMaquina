#Classificação Veiculos

### Pacotes necessários:
#install.packages("e1071")
#install.packages("caret")
library("caret")

### Leitura dos dados
setwd("/home/herivelton/Documentos/IAA/")

dados <- read.csv("Veiculos_Dados.csv")
#View(dados)

### Retira o atributo ID
dados$a <- NULL

### Cria um arquivo com 80% das linhas para treino e 20% para teste
set.seed(202483)
ind <- createDataPartition(dados$tipo, p=0.80, list = FALSE)
treino <- dados[ind,]
teste <- dados[-ind,]

### Cria um grid com vários valores para K e faz o
### treinamento
tuneGrid <- expand.grid(k = c(1,3,5,7,9))
knn <- train(tipo ~ ., data = treino, method = "knn",tuneGrid=tuneGrid)
knn
predict.knn <- predict(knn, teste)
confusionMatrix(predict.knn, as.factor(teste$tipo))
##################################################################################
##### Treinamento de Redes Neurais com Hold Out
rna <- train(tipo~., data=treino, method="nnet",trace=FALSE)
rna
predicoes.rna <- predict(rna, teste)
confusionMatrix(predicoes.rna,as.factor(teste$tipo))

##### Treinamento de Rede Neurais com CrossValidation
ctrl <- trainControl(method = "cv", number = 10)
rna_cv <- train(tipo~., data=treino, method="nnet",trace=FALSE, trControl=ctrl)
rna_cv
predict.rna_cv <- predict(rna_cv, teste)
confusionMatrix(predict.rna_cv, as.factor(teste$tipo))
##################################################################################
### Treinamento de SVM com HoldOut
svm <- train(tipo~., data=treino, method="svmRadial")
svm
predicoes.svm <- predict(svm, teste)
confusionMatrix(predicoes.svm, as.factor(teste$tipo))

#### Treinamento de SVM com Cross-validation
ctrl <- trainControl(method = "cv", number = 10)
svm_cv <- train(tipo~., data=treino, method="svmRadial", trControl=ctrl)
svm_cv
predict.svm_cv <- predict(svm_cv, teste)
confusionMatrix(predict.svm_cv, as.factor(teste$tipo))
##################################################################################
### Treinamento Random Forest com HoldOut
rf <- train(tipo~., data=treino, method="rf")
rf
predicoes.rf <- predict(rf, teste)
confusionMatrix(predicoes.rf, as.factor(teste$tipo))

#### Treinamento de Random Forest com Cross-validation
ctrl <- trainControl(method = "cv", number = 10)
rf_cv <- train(tipo~., data=treino, method="rf", trControl=ctrl)
rf_cv
predict.rf_cv <- predict(rf_cv, teste)
confusionMatrix(predict.rf_cv, as.factor(teste$tipo))


#### Casos novos
dados_novos <- read.csv("Veiculos_Dados_novos.csv")
dados_novos$a <- NULL
View(dados_novos)

predict.svm_novos <- predict(svm_cv, dados_novos)
resultado <- cbind(dados_novos, predict.svm_novos)
resultado$tipo <- NULL
View(resultado)
write.csv(resultado, "saida_Veiculos_novos.csv")

