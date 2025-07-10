#Classificação Diabetes

### Pacotes necessários:
#install.packages("e1071")
#install.packages("caret")
library("caret")

### Leitura dos dados
setwd("/home/herivelton/Documentos/IAA/")

dados <- read.csv("Diabetes.csv")
#View(dados)

### Retira o atributo ID
dados$num <- NULL

### Cria um arquivo com 80% das linhas para treino e 20% para teste
set.seed(202483)
ran <- createDataPartition(dados$diabetes, p=0.80, list = FALSE)
treino <- dados[ran,]
teste <- dados[-ran,]

####  Treinamento KNN
tuneGrid <- expand.grid(k = c(1,3,5,7,9))
knn <- train(diabetes ~ ., data = treino, method = "knn",tuneGrid=tuneGrid)
knn
predict.knn <- predict(knn, teste)
confusionMatrix(predict.knn, as.factor(teste$diabetes))

###########################################################################
##### Treinamento de Rede Neurais com Hold Out
rna <- train(diabetes~., data=treino, method="nnet",trace=FALSE)
rna
predicoes.rna <- predict(rna, teste)
confusionMatrix(predicoes.rna,as.factor(teste$diabetes))

##### Treinamento de Redes Neurais com Cross-Validation
ctrl <- trainControl(method = "cv", number = 10)
rna_cv <- train(diabetes~., data=treino, method="nnet",trace=FALSE, trControl=ctrl)
rna_cv
predict.rna_cv <- predict(rna_cv, teste)
confusionMatrix(predict.rna_cv, as.factor(teste$diabetes))
###########################################################################
### Treinamento SVM com HoldOut
svm <- train(diabetes~., data=treino, method="svmRadial")
svm
predicoes.svm <- predict(svm, teste)
confusionMatrix(predicoes.svm, as.factor(teste$diabetes))

#### Treinamento SVM com Cross-validation
ctrl <- trainControl(method = "cv", number = 10)
svm_cv <- train(diabetes~., data=treino, method="svmRadial", trControl=ctrl)
svm_cv
predict.svm_cv <- predict(svm_cv, teste)
confusionMatrix(predict.svm_cv, as.factor(teste$diabetes))
###########################################################################
### Treinamento Random Forest com HoldOut
rf <- train(diabetes~., data=treino, method="rf")
rf
predicoes.rf <- predict(rf, teste)
confusionMatrix(predicoes.rf, as.factor(teste$diabetes))

#### Treinamento Random Forest com Cross-validation
ctrl <- trainControl(method = "cv", number = 10)
rf_cv <- train(diabetes~., data=treino, method="rf", trControl=ctrl)
rf_cv
predict.rf_cv <- predict(rf_cv, teste)
confusionMatrix(predict.rf_cv, as.factor(teste$diabetes))

#### Casos novos
dados_novos <- read.csv("Diabetes_novos.csv")
dados_novos$a <- NULL
#View(dados_novos)

predict.svm <- predict(svm_cv, dados_novos)
resultado <- cbind(dados_novos, predict.svm)
resultado$diabetes <- NULL
#View(resultado)
write.csv(resultado,"saida_Diabetes_novos.csv")
