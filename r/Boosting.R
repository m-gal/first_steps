# This script does boosting models

# First deals
rm(list=ls())
gc()
opar <- par(no.readonly = TRUE)
options(scipen=9999)

# Load libraries
library(data.table)
library(Matrix)
#library(mice)
library(ggplot2)



# BEG: Load & prepare data -----------------------------------------------------
## load data
if (grepl('C:/', getwd()) == 1) {
  DT.field <- fread('~/RProjects/201706-UnderOptim/DS_FieldDescript.csv')
  DT <- fread('~/RProjects/201706-UnderOptim/UO.EDA/DT_issued.csv',
              sep = ';', encoding = 'UTF-8', na.strings = c('', 'NA'),
              stringsAsFactors = TRUE)
} else {
  DT.field <- fread('T:/RProjects/201706-UnderOptim/DS_FieldDescript.csv')
  DT <- fread('T:/RProjects/201706-UnderOptim/UO.EDA/DT_issued.csv', na.strings = '',
              sep = ';', encoding = 'UTF-8', na.strings = c('', 'NA'),
              stringsAsFactors = TRUE)
}

DT.field[, feature := gsub('_', '.', tolower(feature))]
DT.field[, .N, keyby = .(stage, group)]
table(DT.field[, .(stage, group)])
DT.field[ stage == 'system', feature]
# END: Load & prepare data -----------------------------------------------------


# BEG: Initial datasets for boosting -------------------------------------------
## 1. Coerce datasets to numeric matrix
#X <- as.matrix(DT[, lapply(.SD, as.numeric), .SDcols = !('was.ovd90.12m')])
y <- DT[, as.numeric(was.ovd90.12m)]

## 2. Set up train and test indexs
set.seed(1414)
train <- sample(1:DT[, .N], DT[, .N]*.8, replace = FALSE)
test <- seq(1:DT[, .N])[-train]
sum(y[train])/length(y[train])*100
sum(y[test])/length(y[test])*100

### 3. Features for exluding from
x.ex <- c('was.ovd90.12m', 'score.fps.retro', 'score.fico.retro')
# END: Initial datasets for boosting -------------------------------------------


# XGBOOST ######################################################################
library(xgboost)
library(caret)
library(pROC)
library(MLmetrics)

# BEG: XGBoost datasets for modeling -------------------------------------------
## 1. Create ALL features xgb.DMatrix
xgb.allXall.train <- xgb.DMatrix(
  as.matrix(DT[train, lapply(.SD, as.numeric), .SDcols = !x.ex]),
  label = y[train])
dim(xgb.allXall.train)
xgb.allXall.test <- xgb.DMatrix(
  as.matrix(DT[test, lapply(.SD, as.numeric), .SDcols = !x.ex]),
  label = y[test])
dim(xgb.allXall.test)
#xgb.DMatrix.save(xgb.allXall.train, "xgb.allXall.train")
#xgb.DMatrix.save(xgb.allXall.test, "xgb.allXall.test")
# END: XGBoost datasets for modeling -------------------------------------------


# BEG: XGBoost params settings -------------------------------------------------
## 1. Default XGboost param settings (from package)
xgb.param.def <- list(booster = 'gbtree',
                      max_delta_step = 0,
                      eta = .3,
                      gamma = 0,
                      max_depth = 6,
                      min_child_weight = 1,
                      subsample = 1,
                      colsample_bytree = 1,
                      num_parallel_tree = 1,
                      base_score = .5,
                      objective = 'binary:logistic',
                      #eval_metric = 'auc',
                      eval_metric = 'logloss'
                      )
## 2. Some default XGboost param settings (from package)
xgb.param.tuned <- list(booster = 'gbtree',
                      max_delta_step = 0,  # = 0
                      eta = .1,              # = .3 => 0.1
                      gamma = 0,             # = 0
                      max_depth = 3,         # = 6  => 3
                      min_child_weight = 20, # = 50 => 20
                      subsample = 1,         # = 1
                      colsample_bytree = 1,  # = 1
                      num_parallel_tree = 1, # = 1
                      base_score = .03,       # = .5
                      objective = 'binary:logistic',
                      #eval_metric = 'auc',
                      eval_metric = 'logloss'
                      )

# END: XGBoost params settings -------------------------------------------------


# BEG: Tuning with CARET XGBoost params on All rows X All columns --------------
## 1. Params grid for tuning XGBoost by CARET
xgb.param.grid <- expand.grid(nrounds = seq(100, 1000, by = 100),
                              max_depth = seq(3, 9, by = 3),
                              eta =  c(.001, .01, .1, .3),
                              gamma = 0,
                              min_child_weight = c(1, 10, 20),
                              subsample = 1,
                              colsample_bytree = 1
                              )
## 2. Control params for CARET
xgb.allXall.tune.ctrl <- trainControl(method = 'cv',
                                      number = 5,
                                      verboseIter = TRUE,
                                      allowParallel = TRUE,
                                      classProbs = TRUE,
                                      summaryFunction = twoClassSummary,
                                      savePredictions = TRUE
                                      )
## 3. CARET Tuning params
xgb.allXall.tune.fit <- train(x = as.matrix(DT[train, lapply(.SD, as.numeric),
                                               .SDcols = !x.ex]),
                              y = factor(y[train], labels = c("yes", "no")),
                              method = 'xgbTree',
                              trControl = xgb.allXall.tune.ctrl,
                              tuneGrid = xgb.param.grid,
                              metric = 'ROC'
                              )
xgb.allXall.tune.fit
# END: Tuning with CARET XGBoost params on All rows X All columns --------------


# BEG: Tuning XGBoost with cycle -----------------------------------------------
## 1. DT for saving results
xgb.max_delta_step <- NULL   # seq(1, 101, by = 10)
xgb.scale_pos_weight <- NULL   # seq(1, 101, by = 10)

## 2. Loop
for (i in seq(0.01, 3.1, by = .1)) {
  xgb.allXall.tune <- xgb.train(data = xgb.allXall.train,
                                params = xgb.param.def,
    max_delta_step = i,
                                nrounds = 50, # => 300
                                verbose = 1,
                                print_every_n = 50,
                                watchlist = list(train = xgb.allXall.train,
                                                 test = xgb.allXall.test)
  )
  xgb.max_delta_step <- rbind(
    xgb.max_delta_step, data.table(i, confusionMatrix(ifelse(
      predict(object = xgb.allXall.tune, newdata = xgb.allXall.test) >= .5, 1, 0),
      y[test])$byClass['Balanced Accuracy']
    ))
  print(i)
}; save.image()

## 3. Naming the columns and save result to csv
names(xgb.max_delta_step) <- c('xgb.max_delta_step', 'b_acc')
fwrite(xgb.max_delta_step, 'xgb.max_delta_step.csv')

## 4. Plot results
ggplot(data = xgb.max_delta_step, aes(x = xgb.max_delta_step, y = b_acc)) + geom_line() +
  geom_point(x = xgb.max_delta_step[b_acc == max(b_acc), ][[1]],
             y = xgb.max_delta_step[b_acc == max(b_acc), ][[2]],
             color = 'red', size = 2) +
  geom_text(x = xgb.max_delta_step[b_acc == max(b_acc), ][[1]],
            y = xgb.max_delta_step[b_acc == max(b_acc), ][[2]],
            label = paste0(xgb.max_delta_step[b_acc == max(b_acc), ][[1]], ' = ',
                           xgb.max_delta_step[b_acc == max(b_acc), ][[2]]),
            hjust = -.1, color = 'red')



### 5. Optimize the threshold for XGBoost
xgb.threshold <- NULL
for (i in seq(0.01, 0.6, by = .01)) {
  xgb.threshold <- rbind(xgb.threshold,
                         data.table(i,
    confusionMatrix(ifelse(
      predict(xgb.allXall.fit, newdata = xgb.allXall.test) >= i, 1, 0),
      y[test])$table[2, 2], # TRUE Negative
    confusionMatrix(ifelse(
      predict(xgb.allXall.fit, newdata = xgb.allXall.test) >= i, 1, 0),
      y[test])$table[2, 1], # FALSE Positive
    confusionMatrix(ifelse(
      predict(xgb.allXall.fit, newdata = xgb.allXall.test) >= i, 1, 0),
      y[test])$overall['Accuracy'],
    confusionMatrix(ifelse(
      predict(xgb.allXall.fit, newdata = xgb.allXall.test) >= i, 1, 0),
      y[test])$byClass['Balanced Accuracy'],
    confusionMatrix(ifelse(
      predict(xgb.allXall.fit, newdata = xgb.allXall.test) >= i, 1, 0),
      y[test])$byClass['Sensitivity'],
    confusionMatrix(ifelse(
      predict(xgb.allXall.fit, newdata = xgb.allXall.test) >= i, 1, 0),
      y[test])$byClass['Specificity']
  ))
print(i)
}; rm(i)

names(xgb.threshold) <- c('xgb.threshold', 'True.Negative', 'False.Positive',
                          'Accuracy', 'Balanced.Accuracy',
                          'Sensitivity', 'Specificity')
xgb.threshold[, Sens.Spec := (Sensitivity + Specificity)]
fwrite(xgb.threshold, 'xgb.threshold.csv')

ggplot(xgb.threshold, aes(x = xgb.threshold)) +
  geom_line(aes(y = True.Negative, color = 'True.Negative')) +
  geom_line(aes(y = False.Positive, color = 'False.Positive')) +
  scale_y_continuous('Quantity') +
  scale_colour_manual('', values = c('green', 'red'),
                      breaks = c('True.Negative', 'False.Positive')) +
  theme(legend.position = 'top')

ggplot(xgb.threshold, aes(x = xgb.threshold)) +
  geom_line(aes(y = Accuracy, color = 'Accuracy')) +
  geom_line(aes(y = Balanced.Accuracy, color = 'Balanced.Accuracy')) +
  geom_line(aes(y = Sensitivity, color = 'Sensitivity')) +
  geom_line(aes(y = Specificity, color = 'Specificity')) +
  geom_line(aes(y = Sens.Spec, color = 'Sens.Spec')) +
  scale_y_continuous('Value', limits = c(0,1.5)) +
  scale_colour_manual('', values = c('green', 'red', 'magenta', 'blue', 'darkgreen'),
                      breaks = c('Accuracy', 'Balanced.Accuracy',
                                 'Sensitivity', 'Specificity', 'Sens.Spec')) +
  theme(legend.position = 'top') +
  geom_hline(yintercept = max(xgb.threshold$Sens.Spec), linetype = 2) +
  geom_vline(xintercept = xgb.threshold[Sens.Spec == max(Sens.Spec), xgb.threshold], linetype = 2) +
  annotate('text', hjust = -.1, vjust = -.5,
           x = xgb.threshold[Sens.Spec == max(Sens.Spec), xgb.threshold],
           y = max(xgb.threshold$Sens.Spec),
           label = paste0(round(max(xgb.threshold$Sens.Spec), 4), ' : ',
                          xgb.threshold[Sens.Spec == max(Sens.Spec),
                                              xgb.threshold])) +
  ggtitle('XGBoost threshold (default, max_delta_step = 0)')

# END: Tuning XGBoost with cycle -----------------------------------------------


# BEG: XGBoost fitting on All rows X All columns -------------------------------
## 1. Fit the XGboost model on
xgb.allXall.fit <- xgb.train(data = xgb.allXall.train,
                            #params = xgb.param.def,
                            params = xgb.param.tuned,
                            nrounds = 300, # => 300
                            verbose = 1,
                            print_every_n = 50,
                            watchlist = list(train = xgb.allXall.train,
                                             test = xgb.allXall.test)
                            )
xgb.save(xgb.allXall.fit, 'xgb.allXall.fit')

## 2. View confusion matrix
### test set
confusionMatrix(ifelse(
  predict(object = xgb.allXall.fit, newdata = xgb.allXall.test) >= .03, 1, 0),
  y[test])
### train set()
confusionMatrix(ifelse(
  predict(object = xgb.allXall.fit, newdata = xgb.allXall.train) >= .5, 1, 0),
  y[train])


## 3. Calculate Gini
xgb.allXall.train.pred <- predict(object = xgb.allXall.fit, newdata = xgb.allXall.train)
xgb.allXall.test.pred <- predict(object = xgb.allXall.fit, newdata = xgb.allXall.test)
xgb.allXall.train.gini <- round(Gini(xgb.allXall.train.pred, y[train]), 3)
xgb.allXall.test.gini <- round(Gini(xgb.allXall.test.pred, y[test]), 3)

## 4. Draw the ROC and Feature importance
plot.roc(y[train], xgb.allXall.train.pred, grid = TRUE,
         print.auc=TRUE, print.auc.x=0.2, print.auc.y=0.2,
         col = 'darkgreen' )
plot.roc(y[test], xgb.allXall.test.pred, add = TRUE,
         print.auc=TRUE, print.auc.x=0.2, print.auc.y=0.1, col = 'darkred')
text(0.150, 0.225, paste('TRAIN:  Gini:', xgb.allXall.train.gini), col = 'darkgreen')
text(0.150, 0.125, paste('TEST :  Gini:', xgb.allXall.test.gini), col = 'darkred')
text(0.250, 0.350, paste0('Tuned & All rows:', '\n', 'was.ovd90.12m ~ All columns'))

## 5. View importantce
### list of features with 90% importance
xgb.ggplot.importance(xgb.importance(colnames(xgb.allXall.train),
                                     model = xgb.allXall.fit)[cumsum(Gain)<=.90]) +
  ggplot2::ggtitle('was.ovd90.12m ~ All features',
                   subtitle = '90% feature importance by XGBoost (tuned)') +
  ggplot2::theme(plot.title = element_text(size = 10, face = "bold"),
                 axis.text.y = element_text(size = 7.5)) +
  ggplot2::aes(width = .7)
# END: XGBoost fitting on All rows X All columns -------------------------------


# BEG: XGBoost fitting on All rows X All columns & RANDOM FOREST ---------------
## 1. Random Forest XGboost param settings
xgb.param.RF <- list(booster = 'gbtree',
                     scale_pos_weight = 1,
                     max_delta_step = 0,
                     eta = .3,
                     gamma = 0,
                     max_depth = 6,
                     min_child_weight = 1,
                     subsample = 1,
                     colsample_bytree = .06,
                     num_parallel_tree = 500,
                     objective = 'binary:logistic',
                     eval_metric = 'auc'
                     )
## 2. Fit the XGboost model on
xgb.allXall.RF.fit <- xgb.train(data = xgb.allXall.train,
                             nrounds = 100,
                             params = xgb.param.RF,
                             verbose = 1,
                             print_every_n = 50,
                             watchlist = list(train = xgb.allXall.train,
                                              test = xgb.allXall.test)
                            )
xgb.save(xgb.allXall.RF.fit, 'xgb.allXall.RF.fit')

## 3. View confusion matrix
### test set
confusionMatrix(ifelse(
  predict(object = xgb.allXall.RF.fit, newdata = xgb.allXall.test) >= .5, 1, 0),
  y[test])
### train set
confusionMatrix(ifelse(
  predict(object = xgb.allXall.RF.fit, newdata = xgb.allXall.train) >= .5, 1, 0),
  y[train])

## 4. Calculate Gini
xgb.allXall.RF.train.pred <- predict(object = xgb.allXall.RF.fit, newdata = xgb.allXall.train)
xgb.allXall.RF.test.pred <- predict(object = xgb.allXall.RF.fit, newdata = xgb.allXall.test)
xgb.allXall.RF.train.gini <- round(Gini(xgb.allXall.RF.train.pred, y[train]), 3)
xgb.allXall.RF.test.gini <- round(Gini(xgb.allXall.RF.test.pred, y[test]), 3)

## 5. Draw the ROC and Feature importance
plot.roc(y[train], xgb.allXall.RF.train.pred, grid = TRUE,
         print.auc=TRUE, print.auc.x=0.2, print.auc.y=0.2,
         col = 'darkgreen' )
plot.roc(y[test], xgb.allXall.RF.test.pred, add = TRUE,
         print.auc=TRUE, print.auc.x=0.2, print.auc.y=0.1, col = 'darkred')
text(0.155, 0.225, paste('TRAIN:  Gini:', xgb.allXall.RF.train.gini), col = 'darkgreen')
text(0.150, 0.125, paste('TEST:  Gini:', xgb.allXall.RF.test.gini), col = 'darkred')
text(0.250, 0.350, paste0('Random Forest & All rows:', '\n', 'was.ovd90.12m ~ All columns'))

## 6. View importantce
### list of features with 90% importance
xgb.ggplot.importance(xgb.importance(colnames(xgb.allXall.train),
                                     model = xgb.allXall.RF.fit)[cumsum(Gain)<.90]
) +
  ggplot2::ggtitle('was.ovd90.12m ~ All features',
                   subtitle = '90% feature importance by XGBoost Random Forest settings') +
  ggplot2::theme(plot.title = element_text(size = 10, face = "bold"),
                 axis.text.y = element_text(size = 7)) +
  ggplot2::aes(width = .7)

rm(xgb.allXall.RF.fit)
# END: XGBoost fitting on All rows X All columns & RANDOM FOREST ---------------


# BEG: XGBoost cross-validation ------------------------------------------------
xgb.allXall.cv <- xgb.cv(data = xgb.allXall.train,
                         nfold = 10,
                         nrounds = 3000,
                  eta = .1,
                         gamma = 0,
                  max_depth = 3,
                  min_child_weight = 20,
                         verbose = TRUE,
                         print_every_n = 50,
                         subsample = .8,
                         colsample_bytree = .8,
                         objective = 'binary:logistic',
                         eval_metric = 'logloss'
                  )
print(cv)
print(cv, verbose = TRUE)
bestRound = which.max(as.matrix(xgb.allXall.cv)[,3]-as.matrix(xgb.allXall.cv)[,4])
xgb.allXall.cv[bestRound,]

# END: XGBoost cross-validation ------------------------------------------------


# BEG: XGBoost on APPLICATION stage & default settings -------------------------
## 1.APPL: Create xgb.DMatrix
### names vector for application stage
cname.appl <- colnames(DT)[colnames(DT) %in% DT.field[stage == 'appl', feature]]
### construct APPLICATION features xgb.DMatrix
xgb.appl.train <- xgb.DMatrix(
  as.matrix(DT[train, lapply(.SD, as.numeric), .SDcols = (cname.appl)]),
  label = y[train])
dim(xgb.appl.train)
xgb.appl.test <- xgb.DMatrix(
  as.matrix(DT[test, lapply(.SD, as.numeric), .SDcols = (cname.appl)]),
  label = y[test])
dim(xgb.appl.test)
xgb.DMatrix.save(xgb.appl.train, 'xgb.DMatrix.appl.train')
xgb.DMatrix.save(xgb.appl.test, 'xgb.DMatrix.appl.test')

## 2.APPl: Fit the XGboost model on
xgb.appl.def.fit <- xgb.train(data = xgb.appl.train,
                             nrounds = 100,
                             params = xgb.param.def,
                             verbose = 1,
                             print_every_n = 20,
                             watchlist = list(train = xgb.appl.train,
                                              test = xgb.appl.test)
                             )

## 3. View confusion matrix
confusionMatrix(ifelse(
  predict(object = xgb.appl.def.fit, newdata = xgb.appl.test) >= .5, 1, 0),
  y[test])

## 4. Calculate Gini
xgb.appl.def.train.pred <- predict(object = xgb.appl.def.fit, newdata = xgb.appl.train)
xgb.appl.def.test.pred <- predict(object = xgb.appl.def.fit, newdata = xgb.appl.test)
xgb.appl.def.train.gini <- round(Gini(xgb.appl.def.train.pred, y[train]), 3)
xgb.appl.def.test.gini <- round(Gini(xgb.appl.def.test.pred, y[test]), 3)

## 5. Draw the ROC and Feature importance
plot.roc(y[train], xgb.appl.def.train.pred, grid = TRUE,
         print.auc=TRUE, print.auc.x=0.2, print.auc.y=0.2,
         col = 'darkgreen' )
plot.roc(y[test], xgb.appl.def.test.pred, add = TRUE,
         print.auc=TRUE, print.auc.x=0.2, print.auc.y=0.1, col = 'darkred')
text(0.150, 0.225, paste('TRAIN:  Gini:', xgb.appl.def.train.gini), col = 'darkgreen')
text(0.150, 0.125, paste('TEST :  Gini:', xgb.appl.def.test.gini), col = 'darkred')
text(0.250, 0.350, paste0('Default & All rows :', '\n', 'was.ovd90.12m ~ Application stage'))

## 6. View importantce
xgb.importance(colnames(xgb.appl.train), model = xgb.appl.def.fit)[]
xgb.ggplot.importance(xgb.importance(colnames(xgb.appl.train),
                                     model = xgb.appl.def.fit)
) +
  ggplot2::ggtitle('was.ovd90.12m ~ Application stage features',
                   subtitle = 'Feature importance by XGBoost default settings') +
  ggplot2::theme(plot.title = element_text(size = 10, face = "bold")) +
  ggplot2::aes(width = .7)
# END: XGBoost on APPLICATION stage & default settings -------------------------
# XGBOOST ######################################################################


# LIGTH GBM ####################################################################
library(lightgbm)
library(methods)
#DT <- fread('DT_issued_boost.csv')

# BEG: LigthGBM datasets for modeling ------------------------------------------
lgb.allXall.train <- lgb.Dataset(
  as.matrix(DT[train, lapply(.SD, as.numeric), .SDcols = !x.ex]),
  label = y[train],
  free_raw_data = FALSE)
dim(lgb.allXall.train)
lgb.allXall.test <- lgb.Dataset(
  as.matrix(DT[test, lapply(.SD, as.numeric), .SDcols = !x.ex]),
  label = y[test],
  free_raw_data = FALSE)
dim(lgb.allXall.test)
lgb.allXall.test.caret <- as.matrix(DT[test, lapply(.SD, as.numeric), .SDcols = !x.ex])
#lgb.Dataset.save(lgb.allXall.train, "lgb.allXall.train")
#lgb.Dataset.save(lgb.allXall.test, "lgb.allXall.test")
save.image()
# END: LigthGBM datasets for modeling ------------------------------------------


# BEG: LigthGBM model fitting --------------------------------------------------
## 1. Main params
lgb.param.default <- list(boosting = 'gbdt',
                          is_unbalance = TRUE,
                          bagging_fraction = 1.0, # subsample = 1,
                          feature_fraction = 1.0, # colsample_bytree = 1
                          objective = 'binary',
                          eval = c('binary_logloss')
                          )

## 2. Fit the model
lgb.allXall.fit <- lgb.train(data = lgb.allXall.train,
                             param = lgb.param.default,
    learning_rate = .15,   # default = .1  => .15
    nrounds = 41,          # default = 100 => 41
    min_data_in_leaf = 19, # defualt = 20  => 19
    num_leaves = 17,       # default = 31  => 17
    max_depth = -1,        # default = -1  => -1
                             eval_freq = 50,
                             verbose = 1,
                             valids = list(train = lgb.allXall.train,
                                           test = lgb.allXall.test)
  )

## 3. View confusion matrix
confusionMatrix(ifelse(predict(lgb.allXall.fit, lgb.allXall.test.caret) >= .5, 1, 0),
                y[test])
confusionMatrix(ifelse(predict(lgb.allXall.fit,
          as.matrix(DT[train, lapply(.SD, as.numeric), .SDcols = !x.ex])) >= .5, 1, 0),
          y[train])

## 4. View features importance
xgb.ggplot.importance(lgb.importance(lgb.allXall.fit)[cumsum(Gain) <= .90]) +
  ggplot2::ggtitle('was.ovd90.12m ~ All features',
                   subtitle = '90% feature importance by LigthGBM (tuned)') +
  ggplot2::theme(plot.title = element_text(size = 10, face = "bold"),
                 axis.text.y = element_text(size = 7.5),
                 panel.background = element_rect(color = 2)) +
  ggplot2::aes(width = .7)
# END: LigthGBM model fitting --------------------------------------------------


# BEG: LigthGBM params tuning --------------------------------------------------
## 1. DT for saving results
lgb.nrounds <- NULL          # seq(1, 501, by = 10)
lgb.learning_rate <- NULL    # seq(0.001, 50, by = 1)
lgb.min_data_in_leaf <- NULL # seq(1, 50, by = 1)
lgb.num_leaves <- NULL       # seq(2, 50, by = 1)
lgb.max_depth <- NULL        # seq(1, 30, by = 1)

## 2. Loop
j <- 1
for (i in seq(.001, 10.1, by = 0.1)) {
lgb.allXall.tune <- lgb.train(data = lgb.allXall.train,
                             param = lgb.param.default,
      learning_rate = .15,   # default = 0.1
      nrounds = 41,          # default = 100
      min_data_in_leaf = 19, # defualt = 20
      num_leaves = 17,       # default = 31
      max_depth = -1,        # default = -1
                             eval_freq = 50,
                             verbose = 1,
                             valids = list(train = lgb.allXall.train,
                                           test = lgb.allXall.test)
                             )
lgb.min_sum_hessian_in_leaf <- rbind(
  lgb.min_sum_hessian_in_leaf,
  data.table(i, confusionMatrix(ifelse(predict(lgb.allXall.tune,
                                               lgb.allXall.test.caret) >= .5, 1, 0),
                                y[test])$byClass['Balanced Accuracy']
             ))
print(j)
j <- j + 1
}; save.image(); rm(i, j)

## 3. Naming the columns and save result to csv
names(lgb.nrounds) <- c('lgb.nrounds', 'b_acc')
  fwrite(lgb.nrounds, 'lgb.nrounds.csv')

names(lgb.learning_rate) <- c('lgb.learning_rate', 'b_acc')
  fwrite(lgb.learning_rate, 'lgb.learning_rate.csv')

names(lgb.min_data_in_leaf) <- c('lgb.min_data_in_leaf', 'b_acc')
  fwrite(lgb.min_data_in_leaf, 'lgb.min_data_in_leaf.csv')

names(lgb.num_leaves) <- c('lgb.num_leaves', 'b_acc')
  fwrite(lgb.num_leaves, 'lgb.num_leaves.csv')

names(lgb.max_depth) <- c('lgb.max_depth', 'b_acc')
  fwrite(lgb.max_depth, 'lgb.max_depth.csv')

names(lgb.min_sum_hessian_in_leaf) <- c('lgb.min_sum_hessian_in_leaf', 'b_acc')
  fwrite(lgb.min_sum_hessian_in_leaf, 'lgb.min_sum_hessian_in_leaf.csv')

## 4. Plot results
ggplot(data = lgb.nrounds, aes(x = lgb.nrounds, y = b_acc)) + geom_line() +
  geom_point(x = lgb.nrounds[b_acc == max(b_acc), ][[1]],
             y = lgb.nrounds[b_acc == max(b_acc), ][[2]],
             color = 'red', size = 2) +
  geom_text(x = lgb.nrounds[b_acc == max(b_acc), ][[1]],
            y = lgb.nrounds[b_acc == max(b_acc), ][[2]],
            label = paste0(lgb.nrounds[b_acc == max(b_acc), ][[1]], ' = ',
                           lgb.nrounds[b_acc == max(b_acc), ][[2]]),
            hjust = -.1, color = 'red')


ggplot(data = lgb.learning_rate, aes(x = lgb.learning_rate, y = b_acc)) + geom_line() +
  geom_point(x = lgb.learning_rate[b_acc == max(b_acc), ][[1]],
             y = lgb.learning_rate[b_acc == max(b_acc), ][[2]],
             color = 'red', size = 2) +
  geom_text(x = lgb.learning_rate[b_acc == max(b_acc), ][[1]],
            y = lgb.learning_rate[b_acc == max(b_acc), ][[2]],
            label = paste0(lgb.learning_rate[b_acc == max(b_acc), ][[1]], ' = ',
                           lgb.learning_rate[b_acc == max(b_acc), ][[2]]),
            hjust = -.1, color = 'red')


ggplot(data = lgb.min_data_in_leaf, aes(x = lgb.min_data_in_leaf, y = b_acc)) + geom_line() +
  geom_point(x = lgb.min_data_in_leaf[b_acc == max(b_acc), ][[1]],
             y = lgb.min_data_in_leaf[b_acc == max(b_acc), ][[2]],
             color = 'red', size = 2) +
  geom_text(x = lgb.min_data_in_leaf[b_acc == max(b_acc), ][[1]],
            y = lgb.min_data_in_leaf[b_acc == max(b_acc), ][[2]],
            label = paste0(lgb.min_data_in_leaf[b_acc == max(b_acc), ][[1]], ' = ',
                           lgb.min_data_in_leaf[b_acc == max(b_acc), ][[2]]),
                           hjust = -.1, color = 'red')


ggplot(data = lgb.num_leaves, aes(x = lgb.num_leaves, y = b_acc)) + geom_line() +
  geom_point(x = lgb.num_leaves[b_acc == max(b_acc), ][[1]],
             y = lgb.num_leaves[b_acc == max(b_acc), ][[2]],
             color = 'red', size = 2) +
  geom_text(x = lgb.num_leaves[b_acc == max(b_acc), ][[1]],
            y = lgb.num_leaves[b_acc == max(b_acc), ][[2]],
            label = paste0(lgb.num_leaves[b_acc == max(b_acc), ][[1]], ' = ',
                           lgb.num_leaves[b_acc == max(b_acc), ][[2]]),
            hjust = -.1, color = 'red')


ggplot(data = lgb.max_depth, aes(x = lgb.max_depth, y = b_acc)) + geom_line() +
  geom_point(x = lgb.max_depth[b_acc == max(b_acc), ][[1]],
             y = lgb.max_depth[b_acc == max(b_acc), ][[2]],
             color = 'red', size = 2) +
  geom_text(x = lgb.max_depth[b_acc == max(b_acc), ][[1]],
            y = lgb.max_depth[b_acc == max(b_acc), ][[2]],
            label = paste0(lgb.max_depth[b_acc == max(b_acc), ][[1]], ' = ',
                           lgb.max_depth[b_acc == max(b_acc), ][[2]]),
            hjust = -.1, color = 'red')


ggplot(data = lgb.min_sum_hessian_in_leaf, aes(x = lgb.min_sum_hessian_in_leaf, y = b_acc)) + geom_line() +
  geom_point(x = lgb.min_sum_hessian_in_leaf[b_acc == max(b_acc), ][[1]],
             y = lgb.min_sum_hessian_in_leaf[b_acc == max(b_acc), ][[2]],
             color = 'red', size = 2) +
  geom_text(x = lgb.min_sum_hessian_in_leaf[b_acc == max(b_acc), ][[1]],
            y = lgb.min_sum_hessian_in_leaf[b_acc == max(b_acc), ][[2]],
            label = paste0(lgb.min_sum_hessian_in_leaf[b_acc == max(b_acc), ][[1]], ' = ',
                           lgb.min_sum_hessian_in_leaf[b_acc == max(b_acc), ][[2]]),
            hjust = -.1, color = 'red')


### 5. Optimize the threshold for LightGBM
lgb.threshold <- NULL
for (i in seq(0.01, 0.99, by = .01)) {
  lgb.threshold <- rbind(lgb.threshold,
                         data.table(i,
    confusionMatrix(ifelse(
      predict(lgb.allXall.fit, lgb.allXall.test.caret) >= i, 1, 0),
      y[test])$table[2, 2], # TRUE Negative
    confusionMatrix(ifelse(
      predict(lgb.allXall.fit, lgb.allXall.test.caret) >= i, 1, 0),
      y[test])$table[2, 1], # FALSE Positive
    confusionMatrix(ifelse(
      predict(lgb.allXall.fit, lgb.allXall.test.caret) >= i, 1, 0),
      y[test])$overall['Accuracy'],
    confusionMatrix(ifelse(
      predict(lgb.allXall.fit, lgb.allXall.test.caret) >= i, 1, 0),
      y[test])$byClass['Balanced Accuracy'],
    confusionMatrix(ifelse(
      predict(lgb.allXall.fit, lgb.allXall.test.caret) >= i, 1, 0),
      y[test])$byClass['Sensitivity'],
    confusionMatrix(ifelse(
      predict(lgb.allXall.fit, lgb.allXall.test.caret) >= i, 1, 0),
      y[test])$byClass['Specificity']
    ))
  print(i)
}; rm(i)
names(lgb.threshold) <- c('lgb.threshold', 'True.Negative', 'False.Positive',
                          'Accuracy', 'Balanced.Accuracy',
                          'Sensitivity', 'Specificity')
lgb.threshold[, Sens.Spec := (Sensitivity + Specificity)]
fwrite(lgb.threshold, 'lgb.threshold.csv')

ggplot(lgb.threshold, aes(x = lgb.threshold)) +
  geom_line(aes(y = True.Negative, color = 'True.Negative')) +
  geom_line(aes(y = False.Positive, color = 'False.Positive')) +
  scale_y_continuous('Quantity') +
  scale_colour_manual('', values = c('green', 'red'),
                      breaks = c('True.Negative', 'False.Positive')) +
  theme(legend.position = 'top')

ggplot(lgb.threshold, aes(x = lgb.threshold)) +
  geom_line(aes(y = Accuracy, color = 'Accuracy')) +
  geom_line(aes(y = Balanced.Accuracy, color = 'Balanced.Accuracy')) +
  geom_line(aes(y = Sensitivity, color = 'Sensitivity')) +
  geom_line(aes(y = Specificity, color = 'Specificity')) +
  geom_line(aes(y = Sens.Spec, color = 'Sens.Spec')) +
  scale_y_continuous('Value', limits = c(0,1.5)) +
  scale_colour_manual('', values = c('green', 'red', 'magenta', 'blue', 'darkgreen'),
                      breaks = c('Accuracy', 'Balanced.Accuracy',
                                 'Sensitivity', 'Specificity', 'Sens.Spec')) +
  theme(legend.position = 'top') +
  geom_hline(yintercept = max(lgb.threshold$Sens.Spec), linetype = 2) +
  geom_vline(xintercept = lgb.threshold[Sens.Spec == max(Sens.Spec), lgb.threshold], linetype = 2) +
  annotate('text', hjust = -.1, vjust = -.5,
           x = lgb.threshold[Sens.Spec == max(Sens.Spec), lgb.threshold],
           y = max(lgb.threshold$Sens.Spec),
           label = paste0(round(max(lgb.threshold$Sens.Spec), 4), ' : ',
                          lgb.threshold[Sens.Spec == max(Sens.Spec),
                                              lgb.threshold])
  )

# END: LigthGBM params tuning --------------------------------------------------
# LIGTH GBM ####################################################################
save.image()
################################################################################








table(cut(DT[, score.fps.retro], 10), DT[, was.ovd90.12m])
nz <- nearZeroVar(DT[, .SD, .SDcols = -('was.ovd90.12m')])
colnames(DT[, .SD, .SDcols = -('was.ovd90.12m')])[nz]
nearZeroVar(DT[, score.fps.retro])
nearZeroVar(DT[, c('score.fps.retro', 'score.fico.retro',
                   'job.company.line.3.code', 'score.cash'), with = FALSE])


DT[, sapply(.SD, function(x){
  a <- length(unique(na.omit(x)))/nrow(x)
  b <- sort(table(x), decreasing = TRUE)[1]/sort(table(x), decreasing = TRUE)[2]
  c <- ifelse(length(unique(na.omit(x)))>2,
              sort(table(x), decreasing = TRUE)[2]/sort(table(x), decreasing = TRUE)[3],
              0)
  return(c (a, b, c))
  }), .SDcols = c('score.fps.retro', 'score.fico.retro',
                  'job.company.line.3.code', 'score.cash')]


(t<- sort(table(DT[, score.fps.retro]), decreasing = TRUE))
rm(t)
t[1]/t[2]
t[2]/t[3]

#-------------------------------------------
DT.field[feature %in% colnames(DT), .(stage, feature)]
colnames(DT)[colnames(DT) %in% DT.field[,feature]]
colnames(DT)[colnames(DT) %in% DT.field[stage == 'retro',feature]]



