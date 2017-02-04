rm(list=ls())
library(ROSE)
library(rpart)
library(rpart.plot)
library(randomForest)

# data from Kaggle "Credit Fraud Detection"
d<-read.csv("~/Dropbox/ML/data/creditcard.csv")

# standardize variables
d.st<-d
for (i in 1:(ncol(d)-1)){
  mu<-mean(d[[i]])
  sigma<-sd(d[[i]])
  d.st[[i]]<-(d[[i]]-mu)/sigma
  
}

# sample size
s_size<-floor(0.75*nrow(d))
set.seed(184)
train_d<-sample(seq_len(nrow(d)),size=s_size)
train<-d.st[train_d,]
test<-d.st[-train_d,]

# Check if data balanced
prop.table(table(train$Class))
# Data is severely imbalanced! 
# 0.998221015 % belongs to Class=0

# Decision Tree Model
fit<-rpart(Class ~ ., data=train)
predictions<-predict(fit,test[,1:(ncol(test)-1)],type="vector")
test$prob<-predictions
test$pred<-0
test$pred[test$prob>0.5]<-1
table(test$Class,test$pred)

#### 0.9993399% accuracy
roc.curve(test$Class,test$pred,plotit=T)
#### Area under the curve (AUC): 0.861

##### LOGISTIC REGRESSION

log.m<-glm(formula=Class ~ ., data=train,family=binomial)
log.p<-predict(log.m,test,type="response")
table(test$Class,log.p>0.5)
# 0.9992978% accuracy
roc.curve(test$Class,log.p>0.5)
# Area under the curve (AUC): 0.826

##### BALANCED DATA
train.rose<-ROSE(Class ~., data=train, seed=111)$data
prop.table(table(train.rose$Class)) # data is balanced!

fit<-rpart(Class ~ ., data=train.rose)
predictions<-predict(fit,test[,1:(ncol(test)-1)],type="vector")
test$prob<-predictions
test$pred<-0
test$pred[test$prob>0.5]<-1
table(test$Class,test$pred)
# 0.9741861% accuracy
# train logistic regresson on balanced data
log.rose<-glm(Class ~., data=train.rose, family=binomial)
pred.log.rose<-predict(log.rose, newdata=test,type="response")
roc.curve(test$Class,pred.log.rose)
# Area under the curve (AUC): 0.977


