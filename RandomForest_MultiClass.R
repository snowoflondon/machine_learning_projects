install.packages('randomForest')
library(randomForest)
library(tidyverse)

df <- read_csv('Iris.csv')

head(df)
summary(df)

set.seed(42)
df$label <- factor(df$label)
train_subset <- sample(nrow(df), 0.8*nrow(df), replace=FALSE)
df_train <- df[train_subset,]
df_val <- df[-train_subset,]

model <- randomForest(label ~ ., data = df_train, importance =T)
model

df_e <- data.frame(err_rate = model$err.rate[,'OOB'], 
                   n_trees=1:model$ntree)
ggplot(df_e, aes(x=n_trees, y=err_rate)) + 
  geom_line() + 
  labs(title='OOB error rate with respect to tree growth')

df_pred_val <- predict(model, df_val, type='class')
table(df_pred_val, df_val$label)

mean(df_pred_val == df_val$label)

importance(model)

varImpPlot(model, type=1, main=NULL)
varImpPlot(model, type=2, main=NULL)
