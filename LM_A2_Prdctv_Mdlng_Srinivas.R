
library(car) # Loading the car library for Levene Test
#library("readxl") # Loading the readxl library to read excel data

library(tidyr)      # for data manipulation functions




library(randomForest) # For random forest model tests

library(dplyr)      # for data manipulation functions

# To load the mlbench library
library(mlbench)

# To list out the contents available in the library
#library(help = "mlbench")

# using "caTools" package to get the required function "sample.split()" to split the dataset
library(caTools)

library(pROC)

library(gplots)


# Accessing and assigning Pima Indians Diabetes Dataset to variable pid_data
data(PimaIndiansDiabetes)
pid_data <- PimaIndiansDiabetes

# To find the no. of elements in our multi-dimensional array
dim(PimaIndiansDiabetes)

# To find the repeated values in our response variable
levels(PimaIndiansDiabetes$diabetes)



# Question 1:______________________________________________________________________________
# To Display head of the Dataset
head(pid_data)

# To clean the data and check missing values
any(is.na(pid_data))

# To check any null values in my Dataset
any(is.null(pid_data))


# Question 2:______________________________________________________________________________

# To convert response variable 'diabetes' to binary format 0 and 1

pid_data$diabetes <- ifelse(pid_data$diabetes == "pos", 1, 0)
pid_data


# To Normalise all the explanatory variables using the standard score transformation

# Z -Score Transform for Pregnant variable
z_preg <- (pid_data$pregnant - mean(pid_data$pregnant))/sd(pid_data$pregnant)
round(mean(z_preg), 5)

sd(z_preg) # To find Standard Deviation

head(z_preg)


# Z -Score Transform for Glucose variable
z_gluc <- (pid_data$glucose - mean(pid_data$glucose))/sd(pid_data$glucose)
round(mean(z_gluc), 5)

sd(z_gluc) # To find Standard Deviation


# Z -Score Transform for Pressure variable
z_pres <- (pid_data$pressure - mean(pid_data$pressure))/sd(pid_data$pressure)
round(mean(z_pres), 5)

sd(z_pres) # To find Standard Deviation


# Z -Score Transform for Triceps variable
z_tric <- (pid_data$triceps - mean(pid_data$triceps))/sd(pid_data$triceps)
round(mean(z_tric), 5)

sd(z_tric) # To find Standard Deviation


# Z -Score Transform for Insulin variable
z_insu <- (pid_data$insulin - mean(pid_data$insulin))/sd(pid_data$insulin)
round(mean(z_insu), 5)

sd(z_insu) # To find Standard Deviation


# Z -Score Transform for Body mass variable
z_mass <- (pid_data$mass - mean(pid_data$mass))/sd(pid_data$mass)
round(mean(z_mass), 5)

sd(z_mass) # To find Standard Deviation


# Z -Score Transform for Pedigree variable
z_pedi <- (pid_data$pedigree - mean(pid_data$pedigree))/sd(pid_data$pedigree)
round(mean(z_pedi), 5)

sd(z_pedi) # To find Standard Deviation


# Z -Score Transform for Age variable
z_age <- (pid_data$age - mean(pid_data$age))/sd(pid_data$age)
round(mean(z_age), 5)
sd(z_age) # To find Standard Deviation


# Question 3:______________________________________________________________________________

# Splitting the dataset into Train and test data
# To split I used sample.split() and subset() function to do so.

# Syntax: sample.split(X = , SplitRatio = )
# Where: X = target variable
# SplitRatio = no of train observation divided by the total number of test observation.
# for eg. SplitRatio for 70%:30% (Train:Test) is 0.7.Here The observations are chosen randomly.

pid_split = sample.split(pid_data$pregnant, SplitRatio = 0.7)

#subsetting into Train data
pid_train = pid_data[pid_split,]

#subsetting into Test data
pid_test = pid_data[!pid_split,]

# Now, checking the dimensions of the train and test data created 
# to check whether the splitting is done correct.

dim(pid_train)
dim(pid_test)


# Fisher Score Calculation___________

fis_calc = function(X, Y, k) # X - matrix with predictors, Y - binary outcome, k top candidates
{
  J<- rep(NA, ncol(X))
  names(J)<- colnames(X)
  for (i in 1:ncol(X))
  {
    X1<- X[which(Y==0),i]
    X2<- X[which(Y==1),i]
    mu1<- mean(X1); mu2<- mean(X2); mu<- mean(X[,i])
    var1<- var(X1); var2<- var(X2)
    n1<- length(X1); n2<- length(X2)
    J[i]<- (n1*(mu1-mu)^2+n2*(mu2-mu)^2)/(n1*var1+n2*var2)
  }
  J<- sort(J, decreasing=TRUE)[1:k]
  return(list(score=J))
}

# Calling the function Fisher Score
fis_calc(pid_data[ , c(1, 2, 3, 4, 5, 6, 7, 8)], pid_data$diabetes, 3)


# Wilcoxon Scores Calculation______________

wils_calc = function(X, Y, k) # X - matrix with predictors, Y - binary outcome, k top candidates
{
  J<- rep(NA, ncol(X))
  names(J)<- colnames(X)
  for (i in 1:ncol(X))
  {
    X_rank<- apply(data.matrix(X[,i]), 2, function(c) rank(c))
    X1_rank<- X_rank[which(Y==0)]
    X2_rank<- X_rank[which(Y==1)]
    mu1<- mean(X1_rank); mu2<- mean(X2_rank); mu<- mean(X_rank)
    n1<- length(X1_rank); n2<- length(X2_rank); N<- length(X_rank)
    num<- (n1*(mu1-mu)^2+ n2*(mu2-mu)^2)
    denom<- 0
    for (j in 1:n1)
      denom<- denom+(X1_rank[j]-mu)^2
    for (j in 1:n2)
      denom<- denom+(X2_rank[j]-mu)^2
    J[i]<- (N-1)*num/denom
  }
  J<- sort(J, decreasing=TRUE)[1:k]
  return(list(score=J))
}

wils_calc(pid_data[ , c(1, 2, 3, 4, 5, 6, 7, 8)], pid_data$diabetes, 3)


## Reduced Train and Test Subsets_____________

# Accessing and assigning specific columns to fs_var1 variable
# var <- select(data, col1, col2, col3,....)

fs_var1 <- dplyr::select(pid_train, glucose, mass, age, diabetes) # selection from trained subset
fs_var2 <-  dplyr::select(pid_test, glucose, mass, age, diabetes) # selection from tested subset

pid_red_split1 <- sample.split(fs_var1$glucose, SplitRatio = 0.7)
pid_red_split2 <- sample.split(fs_var2$glucose, SplitRatio = 0.7)


# Subsetting as Reduced Train data
pid_red_train = fs_var1[pid_red_split1,]

# Subsetting as Reduced Test data
pid_red_test = fs_var2[!pid_red_split2,]

# Now, checking the dimensions of the train and test data created 
# to check whether the splitting is done correct.

dim(pid_red_train)
dim(pid_red_test)


# Question 4: Logistic Regression ________________________________________________________


# Logistic regression with all the predictors included
mylogit1 <- glm(diabetes~ glucose+mass+age, data = pid_red_train, family = "binomial")
summary(mylogit1)

# Improving the mmodel with stepAIC
library(MASS)
mylogit2 <- stepAIC(mylogit1)

summary(mylogit2)


# Question 5: Random Forest ______________________________________________________________

#________________ Random forest Model for "trained subset"_________________
rfm_train <- randomForest(factor(diabetes)~., data = pid_train, importance = TRUE,
                   ntree = 50, mtry = 3, replace = TRUE)
print(rfm_train)

importance(rfm_train)
  
# Evaluating Model Accuracy & Building Confusion Matrix
cnf_matrix <- table(predict(rfm_train), pid_train$diabetes) # conditional RF
cnf_matrix

# Calculating Accuracy

sum(diag(cnf_matrix))/sum(cnf_matrix)*100

# Calculating Sensitivity and Specifity 
# Sensitivty = TP/TP+FN and specifity = TN/TN+FN

sensitivity1 = 286/(286+73)
specificity1 = 122/(122+57)

sensitivity1
specificity1



#________________ Random forest Model for "Reduced trained subset"________________

rfm_red_train <- randomForest(factor(diabetes)~., data = pid_red_train, importance = TRUE,
                    ntree = 50, mtry = 3, replace = TRUE)
print(rfm_red_train)

importance(rfm_red_train)


# Evaluating Model Accuracy & Building Confusion Matrix
cnf_matrix2 <- table(predict(rfm_red_train), pid_red_train$diabetes) # conditional RF
cnf_matrix2

# Calculating Accuracy
sum(diag(cnf_matrix2))/sum(cnf_matrix2)*100

# Calculating Sensitivity and Specifity 
# Sensitivty = TP/TP+FN and specifity = TN/TN+FN

sensitivity2 = 203/(203+59)
specificity2 = 80/(80+41)

sensitivity2
specificity2



# Question 6: Plotting ROC Curve ______________________________________________________________

# ROC for the train data

# 1. ROC Curve for trained subset
logit_m1 = glm(formula = diabetes~., data = pid_train ,family = 'binomial')
summary(logit_m)

test_prob1 = predict(logit_m1, pid_test, type = "response")
logit_P <- ifelse(test_prob1 > 0.5, 1, 0) # Probability check

CM = table(pid_test$diabetes, logit_P)
print(CM)
# err_metric(CM)

#ROC-curve using pROC library
library(pROC)
test_roc1 = roc(pid_test$diabetes ~ test_prob1, plot = TRUE, print.auc = TRUE) # ROC Score
plot(test_roc1 , main ="ROC curve -- Logistic Regression ")

# To find AUC Value
as.numeric(roc_score$auc)


# 2. ROC Curve for reduced trained subset
logit_m2 = glm(formula = diabetes~., data = pid_red_train ,family = 'binomial')
summary(logit_m2)

test_prob2 = predict(logit_m2, pid_red_test, type = "response")
logit_P <- ifelse(test_prob2 > 0.5, 1, 0) # Probability check

CM = table(pid_red_test$diabetes, test_prob2)
print(CM)
# err_metric(CM)

#ROC-curve using pROC library
library(pROC)
test_roc2 = roc(pid_red_test$diabetes ~ test_prob2, plot = TRUE, print.auc = TRUE) # ROC Score
plot(test_roc , main ="ROC curve -- Logistic Regression ")

# To find AUC Value
as.numeric(roc_score$auc)

