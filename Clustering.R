# This script doing Self-Organizing Kohonen Maps
# This script should be executed after <UO.EDA.R>
#
# Mikhail Galkin 14/10/2017


# save.image()
# load('.RData')

### FIRST STEP
rm(list=ls())
gc()

options(digits = 7, scipen = 0) # default
# options(digits = 7, scipen = 9999)

opar <- par(no.readonly = TRUE) # save deafault parameters
par(opar)

### LOAD LIBRARIES
library(kohonen)
library(lightgbm)
library(data.table)
library(Matrix)
library(ggplot2)
library(gridExtra)
library(cluster)
library(factoextra)
library(colorRamps)
library(pals)
library(Hmisc)
library(magrittr)
library(partykit)
library(rpart)

# BEG: LOAD DATA ---------------------------------------------------------------
## 1. Load data
if (grepl('C:/', getwd()) == 1) {
  DT <- fread('~/RProjects/201706-UnderOptim/UO.EDA/DT_all.csv',
              encoding = 'UTF-8', na.strings = c('', 'NA'), stringsAsFactors = TRUE)
  DT.f <- fread('~/RProjects/201706-UnderOptim/UO.Importance/DTf_importance.csv')
} else {
  DT <- fread('T:/RProjects/201706-UnderOptim/UO.EDA/DT_all.csv',
              encoding = 'UTF-8', na.strings = c('', 'NA'), stringsAsFactors = TRUE)
  DT.f <- fread('T:/RProjects/201706-UnderOptim/UO.Importance/DTf_importance.csv')
}
### view absent features
names(DT)[!(names(DT) %in% DT.f[, Feature])]
### view classes
unique(DT[, sapply(.SD, class)])
# END: LOAD DATA ---------------------------------------------------------------



# BEG: PREPARE DATA ------------------------------------------------------------
### some view
DT[dim.preapproved != 'Предодобренные', .N, by = dim.segment]

## STEP 1. Remove weak features & preapproved
### 1. Delete preapproved issues & zero-gain features
DT <- DT[dim.preapproved != 'Предодобренные']
DT[, DT.f[Gain.avg == 0, Feature] := NULL]
dim(DT)

### 2. Delete score.group features
DT[ , (DT.f[stage == 'score', Feature]) := NULL]
dim(DT)

### 3. Delete feature with zero-near importance less than 1%
DT.f[Feature %in% names(DT), .N, keyby = (cut(Gain.avg, 50))]
# ggplot(data = DT.f[stage != 'score' & Feature != 'was.ovd90.12m' & Gain.avg >0,
#                    .(Feature, Gain.avg)], aes(x = Gain.avg)) +
#   geom_histogram(bins = 500)
DT[, DT.f[Gain.avg < 0.01 & Gain.avg > 0, Feature] := NULL]
dim(DT)

### 4. Delete some special features
DT[, c('packettypename') := NULL]
dim(DT)

### 5. Replace credit.sum by requested.loan.amount and delete requested..
DT[, .(requested.loan.amount = sum(requested.loan.amount),
       credit.sum = sum(credit.sum)),
   by = .(dim.issued, dim.segment)]
sum(DT[!is.na(credit.sum), requested.loan.amount] -
  DT[!is.na(credit.sum), credit.sum])
DT[, credit.sum := requested.loan.amount]
DT[, requested.loan.amount := NULL]


## STEP 2.
#### Continuation step.1.
#### Without weak features and only heavy features at every group
xclude <- c('was.ovd90.12m', 'dim.issued',
            'otkaz.detail', 'otkaz.who',
            'matrix.id', 'dbr.name')
top = 10
cols <- c('Gain.avg.preapp0.grp.application', 'Gain.avg.preapp0.grp.bki',
          'Gain.avg.preapp0.grp.calculation', 'Gain.avg.preapp0.grp.furfsr',
          'Gain.avg.preapp0.grp.info', 'Gain.avg.preapp0.grp.rbo',
          'Gain.avg.preapp0.grp.rbo.behavior')

### Feature list fom SOM
f.som <- NULL
for (i in cols){
  f.som <- unique(
    rbind(f.som,
          DT.f[!is.na(get(i)), .(Feature, get(i))
               ][!(Feature %in% xclude),
                 ][Feature != 'packettypename',
                   ][order(-V2)
                     ][1:top,
                       ][!is.na(Feature), .(Feature)]
          ))
}; rm(i, top, cols)
(f.som <- unlist(f.som))

### some view
DT[, .N, by = (dim.segment)]
DT[, mean(was.ovd90.12m)]*100
DT[dim.issued == 1, .(.N, mean(was.ovd90.12m)*100), by = (dim.segment)]
# END: PREPARE DATA ------------------------------------------------------------



# BEG: CREATE TRAIN SET --------------------------------------------------------
set.seed(1414)
train <- sample(1:DT[, .N], DT[, .N]*.8, replace = FALSE)
test <- seq(1:DT[, .N])[-train]

DT.full <- copy(DT)
rm(DT); gc()

DT.test <- copy(DT.full[test, ])
DT.train <- copy(DT.full[train, ])

### view summaty info
print(
rbind(
  DT.full[, .(.N,
              dim.issued = mean(dim.issued*100, na.rm = TRUE),
              was.ovd90.12m = mean(was.ovd90.12m*100, na.rm = TRUE)),
          by = .(dim.segment)
          ][, was.ovd90.12m :=
              ifelse(was.ovd90.12m == 0, 0, was.ovd90.12m/dim.issued*100)
            ],
  DT.full[, .(dim.segment = 'ВСЕГО',
              .N,
              dim.issued = mean(dim.issued*100, na.rm = TRUE),
              was.ovd90.12m = mean(was.ovd90.12m*100, na.rm = TRUE)),
          ][, was.ovd90.12m :=
              ifelse(was.ovd90.12m == 0, 0, was.ovd90.12m/dim.issued*100)
            ]
  ))
# END: CREATE TRAIN SET --------------------------------------------------------



# BEG: SAVE DATA FOR REPORT ----------------------------------------------------
write.table(DT.full, file = 'DT_full.csv', sep = ';', na = 'NA', dec = '.',
            row.names = FALSE, fileEncoding = 'UTF-8')

write.table(DT.train, file = 'DT_train.csv', sep = ';', na = 'NA', dec = '.',
            row.names = FALSE, fileEncoding = 'UTF-8')
write.table(train, file = 'train.csv', sep = ';', na = 'NA', dec = '.',
            row.names = FALSE, fileEncoding = 'UTF-8')

write.table(DT.test, file = 'DT_test.csv', sep = ';', na = 'NA', dec = '.',
            row.names = FALSE, fileEncoding = 'UTF-8')
write.table(test, file = 'test.csv', sep = ';', na = 'NA', dec = '.',
            row.names = FALSE, fileEncoding = 'UTF-8')

write.table(f.som, file = 'f_som.csv', sep = ';', na = 'NA', dec = '.',
            row.names = FALSE, col.names = FALSE, fileEncoding = 'UTF-8')
# END: SAVE DATA FOR REPORT ----------------------------------------------------



# BEG: SOM TRAINING & SAVE -----------------------------------------------------
# ## 1. Create dataset for SOM Kohonen
# DT.som <- as.matrix(DT.train[, lapply(.SD, function(x){
#   scale(as.numeric(x))
#   }),
#   .SDcols = names(DT.train)[(names(DT.train) %in% f.som)]
#   ])
# dim(DT.som)
# (saveRDS(DT.som, paste0('DT.som.', dim(DT.som)[1], 'x', dim(DT.som)[2], '.rds')))
# 
# ## 2. View NA.fraction by rows for supersom ~ maxNA.fraction
# ### full dataset
# DT.train[, .N, keyby = (cut(apply(DT.train, 1, FUN = function(x){
#   round(sum(is.na(x))/length(x), 2)
#   }), breaks = seq(0, 1, by = .05), right = FALSE))
#   ]
# ### dataset with f.som only by row
# DT.train[, .N,
#    keyby = (cut(apply(DT.train[, .SD, .SDcols = f.som], 1,
#                       FUN = function(x){
#                         round(sum(is.na(x))/length(x), 2)
#                         }),
#                 breaks = seq(0, 1, by = .05), right = FALSE))
#    ]
# ### dataset with f.som only by
# DT.train[, lapply(.SD, function(x){
#   round(sum(is.na(x))/length(x), 2)
#   }),
#   .SDcols = names(DT.train)[(names(DT.train) %in% f.som)]
#   ] %>% t() %>% sort()
# 
# 
# ## 3. TRAIN THE SOM MODEL!
# system.time(
# som.fit <- supersom(DT.som,
#                     grid = somgrid(xdim = 40, ydim = 40, topo = 'hexagonal'),
#                     rlen = 10000,
#                     alpha = c(.05, .01),
#                     user.weights = 1,
#                     maxNA.fraction = .9,
#                     keep.data = TRUE,
#                     dist.fcts = 'sumofsquares',
#                     normalizeDataLayers = TRUE)
# )
# rm(DT.som)
# ## 4. Save the model
# (saveRDS(som.fit, paste0('som.fit.',
#                         som.fit$grid$xdim, 'x', som.fit$grid$ydim,
#                         '.', round(length(train)/dim(DT.full)[1]*100, 0), 'train',
#                         '.', length(som.fit$changes), 'rlen',
#                         '.', som.fit$maxNA.fraction*100, 'NA',
#                         '.', length(dimnames(som.fit$data[[1]])[[2]]), 'ff.',
#                         substr(Sys.Date(), 1, 4),
#                         substr(Sys.Date(), 6, 7),
#                         substr(Sys.Date(), 9, 10),
#                         '.rds'))
# )
# ### view how much rows the model excluded by maxNA.fraction
# DT.train[is.na(som.fit$unit.classif), .N]
# END: SOM TRAINING & SAVE -----------------------------------------------------



### BEG: LOAD EARLY FITTED SOM MODEL -------------------------------------------
 ## 1. Early models was trained on all of dataset
 ##    & [credit.sum != requested.loan.amount]
 # s <- 'som.fit.20x20.10000rlen.90NA.65ff'
 # s <- 'som.fit.20x20.80train.10000rlen.90NA.65ff.20170906'

 ## 2. LAST models on 80% train datasets
 ##   & credit.sum = requested.loan.amount
 # s <- 'som.fit.20x20.80train.10000rlen.90NA.65ff.20170919'
 # s <- 'som.fit.24x24.80train.10000rlen.90NA.65ff.20170922'
 # s <- 'som.fit.30x30.80train.10000rlen.90NA.65ff.20170921'
 # s <- 'som.fit.40x40.80train.10000rlen.90NA.65ff.20170920'
  s <- 'som.fit.40x40.80train.10000rlen.90NA.65ff.20171010' # to 31.08.17 & without doubled requests

 ## 3. Load model
 if (grepl('C:/', getwd()) == 1) {
   s <- readRDS(paste0('~/RProjects/201706-UnderOptim/UO.Clustering/', s, '.rds'))
 } else {
   s <- readRDS(paste0('T:/RProjects/201706-UnderOptim/UO.Clustering/', s, '.rds'))
 }

 som.fit <- s
 rm(s)
### END: LOAD EARLY FITTED SOM MODEL -------------------------------------------



# BEG: MISSING NODES -----------------------------------------------------------
## 1. Name of current model
(som <- paste0(som.fit$grid$xdim, 'x', som.fit$grid$ydim,
               '.', round(length(train)/dim(DT.full)[1]*100, 0), 'train',
               '.',length(som.fit$changes), 'rlen',
               '.',som.fit$maxNA.fraction*100, 'NA',
               '.', length(dimnames(som.fit$data[[1]])[[2]]), 'ff'))


## 2. Find missing nodes
### missing nodes
(if (length(sort(unique(som.fit$unit.classif))) < nrow(som.fit$codes[[1]])){
  nodes.na <- which(!(seq(1, nrow(som.fit$codes[[1]])) %in%
                         unique(som.fit$unit.classif)))
} else {
  nodes.na <- NULL
})
### nodes without issued credits
(nodes.denied <- DT.train[, sum(dim.issued),
                   keyby = .(som.fit$unit.classif)][V1 == 0, som.fit])
# END: MISSING NODES -----------------------------------------------------------



# BEG: NODES CLUSTERING --------------------------------------------------------
# ## 1. Clustering the nodes of SOM
# ### estimate optimal number of clusters by elbow  method
# k.max <- 30
# #### estimate with factoextra package
# fviz_nbclust(x = som.fit$codes[[1]], FUNcluster = kmeans,
#              method = 'wss', k.max = k.max) +
#   ggtitle(label = 'Optimal number of cluster : FUN = kmeans'
#           , subtitle = som
#   )
# fviz_nbclust(x = som.fit$codes[[1]], FUNcluster = pam,
#              method = 'wss', k.max = k.max) +
#   ggtitle(label = 'Optimal number of cluster : FUN = pam'
#           , subtitle = som
#   )
#
# ### estimate optimal number of clusters by average silhouette width
# fviz_silhouette(silhouette(pam(x = som.fit$codes[[1]], k = k.max)))
# fviz_nbclust(x = som.fit$codes[[1]], FUNcluster = pam,
#              method = 'silhouette', k.max = k.max) +
#   ggtitle(label = 'Optimal number of cluster : FUN = pam'
#           , subtitle = som
#   )
#
# ### ATTANTION!! estimate optimal number of clusters by Gap statistic
# fviz_gap_stat(clusGap(som.fit$codes[[1]], FUNcluster = kmeans,
#                       nstart = 100, K.max = k.max, B = 100)) +
#   ggtitle(label = 'Optimal number of cluster : FUN = kmeans'
#           , subtitle = som
#   )
# fviz_gap_stat(clusGap(som.fit$codes[[1]], FUNcluster = pam,
#                       K.max = k.max, B = 100)) +
#   ggtitle(label = 'Optimal number of cluster : FUN = pam'
#           , subtitle = som
#   )
#
# ### hierarchical clustering to cluster the codebook vectors
# som.clust <- cutree(hclust(dist(som.fit$codes[[1]])), 10); rm(k.max)
# END: NODES CLUSTERING --------------------------------------------------------



# BEG: SOM VISUALIZATION -------------------------------------------------------
### colour palette & shape definition
shape <-  'straight' # 'round'

## 1. Training progress & results
par(mfrow = c(2, 2), mar = c(5, 4, 3, 2), par(cex.main = .9))

### mean distance to closest unit
plot(som.fit, type = 'changes', main = 'Training progress')

### counts within nodes
plot(som.fit, type = 'counts', palette.name = matlab.like, shape = shape,
     main = paste('Counts of cases in nodes'
                  , '::', som
                  ))
# add.cluster.boundaries(som.fit, som.clust, col = 'red', lwd = 4)

### neighbour distances
plot(som.fit, type='dist.neighbours', palette.name=grey.colors, shape = shape,
     main = paste('Neighbour distances'
                  , '::', som
                  ))
# add.cluster.boundaries(som.fit, som.clust, col = 'red', lwd = 4)

### map quality
plot(som.fit, type = 'quality', palette.name = matlab.like, shape = shape,
     main = paste('Nodes quality'
                  , '::', som
                  ))
# add.cluster.boundaries(som.fit, som.clust, col = 'red', lwd = 4)

par(mfrow = c(1, 1), mar = c(5.1, 4.1, 3.1, 2.1))


## 2. Bads & approval plots
# par(mfrow = c(2, 2), mar = c(5, 4, 3, 2), par(cex.main = .9))
### %was.ovd90.12m with dim.issued == 0
plot(som.fit, type = 'property', palette.name = green2red,
     property = setDT(rbind(
       DT.train[, lapply(.SD, function(x){mean(x, na.rm = TRUE)}),
          .SDcols = c('dim.issued', 'was.ovd90.12m'),
          keyby = .(som.fit$unit.classif)
          ][, was.ovd90.12m.T :=
              ifelse(was.ovd90.12m == 0, 0, was.ovd90.12m/dim.issued)
            ][, .(Group.1 = som.fit, x= was.ovd90.12m.T*100)],
       if (is.null(nodes.na)){
         NULL
         } else {
           cbind(Group.1 = nodes.na, x = NA)
           })
       )[Group.1 %in% nodes.denied, x := NA
         ][order(Group.1)][, x],
     shape = shape,
     ncolors = 50,
     main = paste('Was.ovd90.12m: Avg.%'
                  , '::', som
                  ))

### %dim.issued
plot(som.fit, type = 'property', palette.name = heat.colors,
     property = setDT(rbind(
       aggregate(as.numeric(DT.train[, dim.issued*100]),
                 by = list(som.fit$unit.classif),
                 FUN = function(x){mean(x, na.rm = TRUE)}),
       if (is.null(nodes.na)){
         NULL
       } else {
         cbind(Group.1 = nodes.na, x = NA)
       })
       )[order(Group.1)][, x],
     shape = shape,
     ncolors = 50,
     main = paste('Approval: Avg.%'
                  , '::', som
                  ))

### #was.ovd90.12m (! HEAVY PLOT)
plot(som.fit, type = 'mapping', pch = 20, cex = .5, shape = shape,
     col = ifelse(DT.train[, was.ovd90.12m] == 1, 'red', 'gray77'),
     main = paste('Was.ovd90.12m,#'
                  , '::', som
                  ))

### #dim.issued (! HEAVY PLOT)
plot(som.fit, type = 'mapping', pch = 20, cex = .5, shape = shape,
     col = ifelse(DT.train[, dim.issued] == 1, 'green3', 'gray77'),
     main = paste('Issued,#'
                  , '::', som
                  ))

par(mfrow = c(1, 1), mar = c(5.1, 4.1, 3.1, 2.1))


### 3. Plot codes top of feature subset
(top <- head(DT.f[group != 'score' &
                    !(Feature %in% xclude) &
                    Feature != 'packettypename' &
                    Feature %in% dimnames(som.fit$codes[[1]])[[2]],
                  .(Feature, Gain.avg)
                  ][order(-Gain.avg)
                    ][[1]], 9))
sub.som.top <- som.fit
sub.som.top$codes <- sub.som.top$codes[[1]][, top]
plot(sub.som.top, type = 'codes',
     palette.name = rainbow,
     bgcol = 'seashell',#polychrome(max(som.clust))[som.clust],
     codeRendering = 'segments', # segments, stars, lines
     shape = shape,
     main = paste('Top-9 features codes'
                  , '::', som
                  )); rm(top, sub.som.top)


## 4. Another feature in original scale
# if (
#   (f <- select.list(sort(c(xclude, dimnames(som.fit$data[[1]])[[2]])),
#                     multiple=FALSE,
#                     graphics=TRUE,
#                     title='Choose feature')) != ''
#   ) {
#   plot(som.fit, type = 'property', palette.name = matlab.like,
#        property = setDT(rbind(
#          aggregate(as.numeric(DT.train[, f, with = FALSE][[1]]),
#                    by = list(som.fit$unit.classif),
#                    FUN = function(x){mean(x, na.rm = TRUE)}),
#          if (is.null(nodes.na)){
#            NULL
#            } else {
#              cbind(Group.1 = nodes.na, x = NA)
#              })
#          )[order(Group.1)][, x],
#        shape = shape,
#        main = paste(f, ': Mean,%'
#                     , '::', som
#                     )); rm(f)
#   }

par(mfrow = c(1, 1), mar = c(5.1, 4.1, 3.1, 2.1), par(cex.main = 1))
rm(shape)
# END: SOM VISUALIZATION -------------------------------------------------------



# BEG: GET & VISUALIZATION GOOD(high)~BAD(low) nodes----------------------------
## 1. Create set with %Approval & %Was90 in nodes
(hilo <- setDT(                                           # coerce to data.table
  rbind(                                                  # adding NA-nodes
    DT.train[, .(cases.N = .N,                            # calc count of cases
           dim.issued = mean(dim.issued, na.rm = TRUE),   # calc mean value
           was.ovd90.12m = mean(was.ovd90.12m, na.rm = TRUE)),
       by = .(node = som.fit$unit.classif)                # by every nodes
       ][, was.ovd90.12m.T :=
           ifelse(was.ovd90.12m == 0, 0, was.ovd90.12m/dim.issued)
         ][node %in% nodes.denied,                                  # if there were refuses
           c('dim.issued', 'was.ovd90.12m', 'was.ovd90.12m.T') := 0 # then assing 0
           ][!is.na(node), ],                                       # remove NA from nodes
    if (is.null(nodes.na)){                                         # adding № of nodes without request
      NULL
    } else {
      cbind(node = nodes.na,
            cases.N = 0,
            dim.issued = 0,
            was.ovd90.12m = 0,
            was.ovd90.12m.T = 0)
    }
  ))[order(node)])


## 2. Calculate zones by quantile
### approval level by quantile
probs = c(.2, .8)
hilo[, app.hilo := ifelse(dim.issued <= hilo[, quantile(dim.issued,
                                              probs = probs)][1], 1,
                    ifelse(dim.issued >= hilo[, quantile(dim.issued,
                                                     probs = probs)][2], 3, 2))]
hilo[, .(.N, mean(dim.issued)), by = (app.hilo)]

### quality (default) level by quantile
hilo[, def.hilo := ifelse(was.ovd90.12m.T <= hilo[, quantile(was.ovd90.12m.T,
                                                   probs = probs)][1], 3,
                    ifelse(was.ovd90.12m.T >= hilo[, quantile(was.ovd90.12m.T,
                                                          probs = probs)][2], 1, 2))]
hilo[, .(.N, mean(was.ovd90.12m.T)), by = (def.hilo)]

### zones with highApproval+highQuality & lowApproval+lowQuality
hilo[, zone.col := ifelse(app.hilo + def.hilo == 2, 'red4',
                       ifelse(app.hilo + def.hilo == 6, 'green4', 'snow'))]
hilo[dim.issued == 0, zone.col := 'gray0']
hilo[, zone := ifelse(app.hilo + def.hilo == 2, 'low',
                   ifelse(app.hilo + def.hilo == 6, 'high', 'any'))]
hilo[dim.issued == 0, zone := 'denied']

### view percentage
hilo[, .(N.nodes = .N, N.cases = sum(cases.N),
      dim.issued = mean(dim.issued)*100,
      was.ovd90.12m.T = mean(was.ovd90.12m.T)*100), by = .(zone)]


## 3. Adding codes from som.fit
som.codes <- setDT(data.table(som.fit$codes[[1]], keep.rownames = TRUE))
setnames(som.codes, 'rn', 'node')
som.codes[, node := as.numeric(substr(node, 2, nchar(node)))]
hilo <- merge(hilo,som.codes); rm(som.codes)

#### save the hilo.table for rmd-report
write.table(hilo, file = 'hilo.csv', sep = ';', na = 'NA', dec = '.',
            row.names = FALSE, fileEncoding = 'UTF-8')


## 4. Plot high-low nodes
### create matrix witn 'new' pseudo-codes
ab <- as.matrix(hilo[, .(dim.issued, was.ovd90.12m.T)])
dimnames(ab) <- list(paste0('V', seq(1, nrow(som.fit$codes[[1]]))),
                     c('Approval', 'Default'))

### create 'new' fitted SOM-model with pseudo-codes
hilo.ab <- som.fit
hilo.ab$codes <- ab

### mapplot our high-low nodes
plot(hilo.ab, type = 'codes',
     palette.name = green2red,
     bgcol = hilo[, zone.col],
     codeRendering = 'segments',
     shape = shape,
     main = paste0('High & Low nodes by ', probs[1], '~', probs[2], ' quantile'
                   #, ':: ', som
     ))

### boxplots by zone
grid.arrange(
  ggplot(hilo[zone != 'denied'], aes(x = zone, y = dim.issued*100, fill = zone.col)) +
    geom_boxplot() +
    labs(subtitle = paste('Approval in nodes: Avg.%:', probs[1], '~', probs[2], 'quantile')) +
    scale_fill_manual(breaks = c('high', 'low', 'any'),
                      values = c('green4', 'red4', 'snow')) +
    scale_y_continuous(name = 'Approval, %',
                       breaks = seq(0, 100, 5),
                       trans = 'sqrt'),
  ggplot(hilo[zone != 'denied'], aes(x = zone, y = was.ovd90.12m.T*100, fill = zone.col)) +
    geom_boxplot() +
    labs(subtitle = paste('Was.ovd90.12m in nodes: Avg.%:', probs[1], '~', probs[2], 'quantile')) +
    scale_fill_manual(breaks = c('high', 'low', 'any'),
                      values = c('green4', 'red4', 'snow')) +
    scale_y_continuous(name = 'Was.ovd90.12m, %',
                       breaks = seq(0, 100, 5),
                       trans = 'sqrt'),
  ncol = 2)

### dotplot with quantile
ggplot(hilo[zone != 'denied'],
       aes(x = dim.issued*100, y = was.ovd90.12m.T*100, color = zone)) +
  geom_point(aes(size = cases.N), alpha = .3) +
  labs(title = paste0('Nodes allocation: ',
                      probs[1]*100, '% ~ ', probs[2]*100, '% quantile')
       ,subtitle = som
       , x = 'Approval, %', y = 'Was.ovd90.12m, %') +
  geom_hline(yintercept = hilo[, quantile(was.ovd90.12m.T, probs = probs)][1]*100,
             linetype = 2, size = .5, color = 'red4') +
  geom_hline(yintercept = hilo[, quantile(was.ovd90.12m.T, probs = probs)][2]*100,
             linetype = 2, size = .5, color = 'red4') +
  geom_hline(yintercept = DT.full[dim.issued == 1, mean(was.ovd90.12m)]*100,
             linetype = 1, size = .5, color = 'red4') +
  geom_vline(xintercept = hilo[, quantile(dim.issued, probs = probs)][1]*100,
             linetype = 2, size = .5, color = 'green4') +
  geom_vline(xintercept = hilo[, quantile(dim.issued, probs = probs)][2]*100,
             linetype = 2, size = .5, color = 'green4') +
  geom_vline(xintercept = DT.full[, mean(dim.issued)]*100,
             linetype = 1, size = .5, color = 'green4') +
  geom_rug(sides = 'tr') +
  scale_color_manual(breaks = c('high', 'low', 'any'),
                     values = c('gray77', 'green4', 'red4')) +
  scale_y_continuous(trans = 'sqrt',
                     breaks = sort(c(seq(0, hilo[, max(was.ovd90.12m.T*100)], 5), 1, 2, 3))) +
  scale_x_continuous(breaks = sort(c(seq(0, hilo[, max(dim.issued*100)], 5))))

rm(ab, hilo.ab)
# END: GET & VISUALIZATION GOOD(high)~BAD(low) nodes----------------------------



# BEG: FINDIND FEATURES WICH HAVE A STATISTICAL DIFF ---------------------------
### 1. Result table
DT.res <- cbind(node = som.fit$unit.classif,
                som.distance = som.fit$distances,
                DT.train)
dim(DT.res)
DT.res <- merge(hilo[, .(node, zone, cases.N,
                      was.ovd90.12m.T, dim.issued.T = dim.issued)],
                DT.res,
                by.x = 'node', by.y = 'node',
                all.x = FALSE, all.y = TRUE,
                sort = FALSE)
dim(DT.res)

DT.res[, c('wave', 'top.up'):= NULL]
dim(DT.res)

DT.res <- DT.res[zone != 'denied', ]
DT.res[, zone := as.factor(zone)]
dim(DT.res)

#### replace all 'Missing' to NA for correctly calculating the test
DT.res[, .N, by = tip.klienta]
#sum(DT.res == 'Missing', na.rm = TRUE)
DT.res[, sum(sapply(.SD, function(x){sum(x == 'Missing', na.rm = TRUE)}))]
DT.res[, sum(sapply(.SD, function(x){sum(is.na(x))}))]
DT.res[, sum(sapply(.SD, function(x){sum(x == 'Missing', na.rm = TRUE) +
    sum(is.na(x))}))]
for(col in names(DT.res)) {
  set(DT.res,
      i = which(DT.res[[col]] == 'Missing'),
      j = col,
      value = NA)
}; rm(col)
DT.res[, sum(sapply(.SD, function(x){sum(is.na(x))}))]
DT.res[, sum(sapply(.SD, function(x){sum(x == 'Missing', na.rm = TRUE)}))]

#### replace mistakes at 'job.amountofworkingyears'
DT.res[, .N, keyby = job.amountofworkingyears]
DT.res[job.amountofworkingyears < 14 ,
       job.amountofworkingyears := NA]
DT.res[job.amountofworkingyears > 50 ,
       job.amountofworkingyears := NA]
DT.res[, .N, keyby = job.amountofworkingyears]

#### to low case all factors
DT.res[, (names(DT.res)[DT.res[, sapply(.SD, is.factor)]]) := lapply(.SD, function(x){
  as.factor(tolower(as.character(x)))
  }),
  .SDcols = (names(DT.res)[DT.res[, sapply(.SD, is.factor)]])
]

#### save the result table for rmd-report
write.table(DT.res, file = 'DT_res.csv', sep = ';', na = 'NA', dec = '.',
            row.names = FALSE, fileEncoding = 'UTF-8')


### 2. Some info from result table
#### average was90 amount zone\nodes
DT.res[, .(.N, approval = mean(dim.issued.T)*100,
           was90 = mean(was.ovd90.12m.T)*100),
       keyby = .(zone)]
DT.res[zone %in% c('high', 'low'),
       .(cases.N = .N,
         approval = mean(dim.issued.T)*100,
         was90 = mean(was.ovd90.12m.T)*100),
       keyby = .(zone, node)]

#### average was90 amount all credits
print(DT.res[, .(.N,
                 approval = mean(dim.issued)*100,
                 was90 = mean(was.ovd90.12m)*100),
             keyby = .(zone)
             ][, 'was90' := ifelse(was90 == 0, 0, was90/approval*100)])
print(DT.res[zone %in% c('high', 'low'), .(.N,
                 approval = mean(dim.issued)*100,
                 was90 = mean(was.ovd90.12m)*100),
             keyby = .(zone, node)
             ][, 'was90' := ifelse(was90 == 0, 0, was90/approval*100)
               ][order(zone, was90, -approval)])

print(DT.res[, .(.N,
                 shareN = .N/DT.res[, .N]*100,
                 amountDenied = .N - sum(dim.issued),
                 sumDenied = sum(ifelse(dim.issued == 0, credit.sum,  0)),
                 amountIssued = sum(dim.issued),
                 sumIssued = sum(ifelse(dim.issued == 1, credit.sum,  0)),
                 percApproval = mean(dim.issued)*100,
                 was90 = mean(was.ovd90.12m)*100),
             keyby = .(zone)
             ][, 'was90' := ifelse(was90 == 0, 0, was90/percApproval*100)])


DT.res[dim.issued == 0, sum(credit.sum), by = zone]


#### set p-value level
pv <- 0.01
### 3. Kruskal-Wallis test for difference between zones low\high\any
#kruskal.test(dim.segment ~ zone, data = DT.res)
#unique(DT.res[, sapply(.SD, class)])
#### p-value by ktuskal test
ignor <- c('matrix.id',
           'birth.region', 'reg.address.region', 'reg.address.city',
           'living.address.region', 'living.address.city',
           'emp.address.region', 'emp.address.city',
           'education',
           'job.industry', 'job.company.line.3.code', 'job.company.line.2.code',
           'packets.mean.income.4.wage',
           'bscor.days.since','bscor.days.since.last',
           'bscor.number.of.incoming',
           'bscor..account.age.max',
           'bscor.days.since.any',
           'bscor.average.credit',
           'bscor.max.credit.limit',

           'total.number.of.loans.in.sib',
           'total.amount.of.loans.in.sib',                
           'spouse.total.amount.of.loans',          
           
           'pd',
           'priziv.vozrast',
           'tip.klienta',
           
           'amount.of.approved.income',
           'other.monthly.expenses',
           'payment',
           
           'u.klienta.aktivy.kvartira',
           'car.type.of.acquisition',
           
           'org.blagonadezhnaya',
           'organization.type.by.owner',          
           'per.of.work.in.cur.org',
           
           'creditsum02',
           'credit.sum',
           
           'equifax.request.count',
           'dirrbscreditor',
           
           'ranee.u.klienta.prosr',
           'running.costs',
           'disposable.income')
write.table(ignor, file = 'ignor.csv', sep = ';', na = 'NA', dec = '.',
            row.names = FALSE, col.names = FALSE, fileEncoding = 'UTF-8')
DT.f[order(-Gain.avg)][Feature %in% ignor, .(Feature, descript)]

res.krusk <-
  DT.res[, lapply(.SD, function(x){
    kruskal.test(x ~ zone,
                 data = DT.res)
    })
   # , .SDcols = f.som                   # by only f.som features
   , .SDcols = setdiff(names(DT.res),    # by all features & not included in c()
                       c('zone', 'node', 'cases.N', 'som.distance',
                         'was.ovd90.12m', 'dim.issued',
                         'was.ovd90.12m.T', 'dim.issued.T',
                         'otkaz.detail', 'otkaz.who', 'refuse.reason.code',
                         ignor))
    ][3, ]

suppressWarnings(
res.krusk <- res.krusk[, -c('zone', 'node', 'cases.N', 'som.distance',
                            'was.ovd90.12m', 'dim.issued',
                            'was.ovd90.12m.T', 'dim.issued.T',
                            'otkaz.detail', 'otkaz.who',
                            'matrix.id', 'refuse.reason.code'
                            )]
)

res.krusk <- unlist(res.krusk)
res.krusk <- data.table(feature = names(res.krusk), p.value=res.krusk)

#### plot Kruskal-Wallis test result
ggplot(data = res.krusk[p.value >= 0 & p.value <= pv],
       aes(y = p.value, x = reorder(feature, -p.value))) +
  geom_bar(stat = 'identity', fill = 'tomato') + coord_flip() +
  ggtitle(paste0('Kruskal test between LOW, HIGH and ANY (',
                 dim(res.krusk[p.value >= 0 & p.value <= pv])[1], ')')) +
  xlab('feature') +
  theme(
    axis.text.y = element_text(size = 7),
    title = element_text(size = 9)
  )


### 4. Wilcoxon rank sum test for difference between zones low and high
#kruskal.test(dim.segment ~ zone, data = DT.res[zone %in% c('low', 'high'), ])

#### p-value by Wilcoxon test between HIGH & LOW
res.wilc.hilo <-
  DT.res[zone %in% c('low', 'high'),
         lapply(.SD, function(x){
           kruskal.test(x ~ zone,
                 data = DT.res[zone %in% c('low', 'high'), ])
  }),
  .SDcols = res.krusk[p.value >= 0 & p.value <= pv, feature]
    ][3, ]
res.wilc.hilo <- unlist(res.wilc.hilo)
res.wilc.hilo <- data.table(feature = names(res.wilc.hilo),
                            pv.hilo = res.wilc.hilo)

#### p-value by Wilcoxon test between HIGH & ANY
res.wilc.hiany <-
  DT.res[zone %in% c('any', 'high'),
         lapply(.SD, function(x){
           kruskal.test(x ~ zone,
                        data = DT.res[zone %in% c('any', 'high'), ])
         }),
         .SDcols = res.krusk[p.value >= 0 & p.value <= pv, feature]
         ][3, ]
res.wilc.hiany <- unlist(res.wilc.hiany)
res.wilc.hiany <- data.table(feature = names(res.wilc.hiany),
                             pv.hiany = res.wilc.hiany)

#### p-value by Wilcoxon test between LOW & ANY
res.wilc.loany <-
  DT.res[zone %in% c('any', 'low'),
         lapply(.SD, function(x){
           kruskal.test(x ~ zone,
                        data = DT.res[zone %in% c('any', 'low'), ])
         }),
         .SDcols = res.krusk[p.value >= 0 & p.value <= pv, feature]
         ][3, ]
res.wilc.loany <- unlist(res.wilc.loany)
res.wilc.loany <- data.table(feature = names(res.wilc.loany),
                             pv.loany = res.wilc.loany)

#### summary p-value by Wilcoxon test
identical(res.wilc.loany[, 1], res.wilc.hilo[, 1]) ==
identical(res.wilc.hilo[, 1], res.wilc.hiany[, 1])

res.wilc <- data.table(res.wilc.hilo,
                       res.wilc.hiany[, .(pv.hiany)],
                       res.wilc.loany[, .(pv.loany)])
res.wilc[, diff.zone := ifelse(pv.hilo <= pv,
                               ifelse(pv.hiany <= pv,
                                      ifelse(pv.loany <= pv, 1, 0), 0), 0)]
res.wilc[, .N, by = .(diff.zone)]
rm(res.wilc.hiany, res.wilc.hilo, res.wilc.loany)

#### plot Wilcoxon-test result between HIGH & LOW zones
ggplot(data = res.wilc[diff.zone == 1],
       aes(y = pv.hilo, x = reorder(feature, -pv.hilo))) +
  geom_bar(stat = 'identity', fill = 'skyblue') + coord_flip() +
  ggtitle(paste0('Wilcoxon test between LOW and HIGH zones by features (',
                 dim(res.wilc[diff.zone == 1])[1], ')')) +
  xlab('feature') +
  theme(
    axis.text.y = element_text(size = 7),
    title = element_text(size = 9)
  )
DT.res[, .N, by = zone]

### 5. Table
res.feat <- DT.res[,
                   lapply(.SD, function(x){
                     if (class(x) == 'factor') {
                       ux <- unique(x)
                       uxm <- ux[which.max(tabulate(match(x, ux[!is.na(ux)])))]
                       paste0(uxm, '(#', table(x)[uxm],
                              ' - %', round(table(x)[uxm]/sum(table(x))*100, 1),
                              ') All(#',  sum(table(x)),
                              ' - %NA=', round(sum(is.na(x))/length(x)*100, 1),
                              ')')
                       } else {
                         paste0('Avg=', prettyNum(mean(x, na.rm = TRUE)), ' ',
                                'Med=', prettyNum(median(x, na.rm = TRUE)), ' ',
                                'Min=', prettyNum(min(x, na.rm = TRUE)), ' ',
                                'Max=', prettyNum(max(x, na.rm = TRUE)), ' ',
                                '%NA=', round(sum(is.na(x))/length(x)*100, 1)
                                )
                         }
                     }), keyby = .(zone),
                   .SDcols = res.wilc[diff.zone == 1, feature]]
res.feat <- dcast(
  melt(res.feat, id.vars = 'zone', variable.name = 'feature'),
  feature ~ zone)
res.feat <- merge(res.wilc, res.feat,
                     by.x = 'feature', by.y = 'feature',
                     all.x = FALSE, all.y = TRUE,
                     sort = FALSE)
res.feat <- res.feat[order(pv.hilo)]
res.feat <- merge(res.feat, DT.f[,.(Feature, descript)],
                     by.x = 'feature', by.y = 'Feature',
                     all.x = TRUE, all.y = FALSE,
                     sort = FALSE)
# setnames(res.feat,
#          c('any', 'high', 'low'),
#          c(paste0('any.', DT.res[zone == 'any', .N]),
#            paste0('high.', DT.res[zone == 'high', .N]),
#            paste0('low.', DT.res[zone == 'low', .N])))
# END: FINDIND FEATURES WICH HAVE A STATISTICAL DIFF ---------------------------



# BEG: IMPORTANCE WILCOXON FEATURE BY BOOSTING ---------------------------------
## 1. LigthGBM datasets for modeling
lgb.data <- lgb.Dataset(
  as.matrix(DT.res[ , lapply(.SD, as.numeric),
                    .SDcols = res.feat[, feature]]),
  label = DT.res[, ifelse(zone == 'high', 1,
                          ifelse(zone == 'low', 2, 0))],
  free_raw_data = FALSE)
dim(lgb.data)

## 2. Main params
lgb.param <- list(boosting = 'gbdt',
                  learning_rate = .01,    # default = .1
                  min_data_in_leaf = 10,  # defualt = 20
                  num_leaves = 31,        # default = 31
                  max_depth = -1,         # default = -1
                  bagging_fraction = 1.0, # subsample = 1,
                  feature_fraction = 0.6, # colsample_bytree = 1
                  objective = 'multiclass',
                  eval = c('multi_logloss'),
                  verbose = 1,
                  save_binary = TRUE,
                  eval_freq = 50)

## 3. Fit the model
set.seed(1414)
lgb.fit <- lgb.train(data = lgb.data,
                     param = lgb.param,
                     nrounds = 1001,
                     num_class = 3)

## 4. Add importanse to final table
res.feat <- merge(res.feat, lgb.importance(lgb.fit)[, .(Feature, gain = Gain)],
                  by.x = 'feature', by.y = 'Feature',
                  all.x = TRUE, all.y = FALSE,
                  sort = FALSE)
res.feat <- res.feat[order(-gain)]
res.feat[, gain.cum := cumsum(gain)]
res.feat[, imp.95 := ifelse(gain.cum < .96, 1, 0)]
res.feat[, diff.zone := NULL]

## 5. View features importance
xgboost::xgb.ggplot.importance(res.feat[imp.95 == 1, .(Feature = feature,
                                                       Gain = gain)]) +
  ggplot2::ggtitle('High & Low & Any zones ~ Features',
                   subtitle = paste0('95% feature (# ',
                                     res.feat[imp.95 == 1, .N],
                                     ') importance by LigthGBM')) +
  ggplot2::theme(plot.title = element_text(size = 10, face = "bold"),
                 axis.text.y = element_text(size = 7.5),
                 panel.background = element_rect(color = 2)) +
  ggplot2::aes(width = .7)

## 6. Save the result.feat and lgb-model for rmd-report
write.table(res.feat, file = 'res_feat.csv', sep = ';', na = 'NA', dec = '.',
            row.names = FALSE, fileEncoding = 'UTF-8')
saveRDS.lgb.Booster(lgb.fit, 'lgb_fit.rds')

rm(lgb.data, lgb.param)
#detach('package:lightgbm', unload=TRUE)
# END: IMPORTANCE WILCOXON FEATURE BY BOOSTING ---------------------------------



for (i in res.feat[imp.95 == 1, feature]) {
  if ( !DT.res[, is.factor(get(i))] &
       DT.res[, length(unique(get(i)))] > 10 ) {
    print(paste('1', i, DT.res[, class(get(i))], DT.res[, length(unique(get(i)))]))
  } else {
    print(paste('2', i, DT.res[, class(get(i))], DT.res[, length(unique(get(i)))]))
  }
}



# BEG: IMPORTANT FEATURE VIZUALIZATION -----------------------------------------
## 1. Loop
# i <- 'family.income.rest'
# i <- 'days.of.last.offset.of.debt'
# i <- 'sum.limit.full.pki'
# i <- 'bscor.days.since.any'
# i <- 'job.industry'
# i <- 'credit.sum'
# i <- 'education'

prbs <- c(0.01, .99)
for (i in res.feat[imp.95 == 1, feature]) {
  ## separate between plots
  cat (i)

  if (!DT.res[, is.factor(get(i))] &
       DT.res[, length(unique(get(i)))] > 10){
    print(paste('1', i))

    ## prepare datasets
    dt <- DT.res[!is.na(get(i)) &
                   get(i) >= quantile(get(i), probs = prbs, na.rm = TRUE)[1] &
                   get(i) <= quantile(get(i), probs = prbs, na.rm = TRUE)[2],
                 .(zone, value = get(i))]
    d <- dt[zone %in% c('any', 'high', 'low') & !is.na(value),
            .(zone = ifelse(zone == 'high', 'h',
                            ifelse(zone == 'low', 'l', 'a')),
              value)
            ][ , zone := as.factor(as.character(zone))]


    ## grow 2 tree
    t1 <- ctree(zone ~ value, data = d,
                control = ctree_control(mincriterion = .99,
                                        minsplit = 200, #1000
                                        maxdepth = 3,
                                        multiway = FALSE)
                )
    t1log <- ctree(zone ~ log(value+1), data = d,
                control = ctree_control(mincriterion = .99,
                                        minsplit = 200, #1000
                                        maxdepth = 3,
                                        multiway = FALSE)
                )

    ## save data for rmd-report
    if (grepl('C:/', getwd()) == 1) {
      fwrite(dt,
             paste0('~/RProjects/201706-UnderOptim/UO.Clustering/Trees/dt1_',
                    i, '.csv'))
      fwrite(d,
             paste0('~/RProjects/201706-UnderOptim/UO.Clustering/Trees/d1_',
                    i, '.csv'))
      saveRDS(t1,
              paste0('~/RProjects/201706-UnderOptim/UO.Clustering/Trees/t1_',
                     i, '.rds'))
      saveRDS(t1log,
              paste0('~/RProjects/201706-UnderOptim/UO.Clustering/Trees/t1log_',
                     i, '.rds'))
    } else {
      fwrite(dt,
             paste0('T:/RProjects/201706-UnderOptim/UO.Clustering/Trees/dt1_',
                    i, '.csv'))
      fwrite(d,
             paste0('T:/RProjects/201706-UnderOptim/UO.Clustering/Trees/d1_',
                    i, '.csv'))
      saveRDS(t1,
              paste0('T:/RProjects/201706-UnderOptim/UO.Clustering/Trees/t1_',
                     i, '.rds'))
      saveRDS(t1log,
              paste0('T:/RProjects/201706-UnderOptim/UO.Clustering/Trees/t1log_',
                     i, '.rds'))
    }

    ## plot density
    grid.arrange(
      #### value
      ggplot(data = dt) +
        geom_density(aes(x = value, fill = zone), alpha = .5, adjust = 3
                     #, kernel = 'biweight'
                     ) +
        scale_fill_manual(values = c('gray', 'green', 'red')) +
        scale_y_continuous(labels = scales::
                             format_format(big.mark = ' ',
                                           decimal.mark = '.',
                                           scientific = FALSE)) +
        scale_x_continuous(labels = scales::
                             format_format(big.mark = ' ',
                                           decimal.mark = '.',
                                           scientific = FALSE
                                           )(pretty(dt[, value])),
        #trans = 'sqrt',
        breaks = pretty(dt[, value])) +
        labs(subtitle = paste0(res.feat[feature == i, descript],
                               ' (№', which(res.feat[, feature] == i),')')) +
        theme(legend.position = c(.97, .87), #'top
              legend.title = element_text(size = 9),
              legend.text = element_text(size = 8),
              legend.key.size = unit(.4, 'cm'),
              axis.title.x = element_blank(),
              axis.text.x = element_text(size = 8),
              axis.title.y = element_text(size = 9),
              axis.text.y = element_text(size = 8)),

      #### log(value + 1)
      ggplot(data = dt) +
        geom_density(aes(x = log(value + 1), fill = zone)
                     , alpha = .5, adjust = 3
                     #, kernel = 'biweight'
                     ) +
        scale_fill_manual(values = c('gray', 'green', 'red')) +
        scale_y_continuous(trans = 'sqrt') +
        scale_x_continuous(name = paste0('ln (', i, ')')) +
        theme(legend.position = 'none',
              axis.title.x = element_text(size = 9),
              axis.text.x = element_text(size = 8),
              axis.title.y = element_text(size = 9),
              axis.text.y = element_text(size = 8)),
      nrow = 2)

    ## plot decision tree
    if (!is.null(unlist(t1)$node.split.breaks)) { # check split in t1 tree
        plot(t1, type = 'extended',
             main = res.feat[feature == i, descript],
             gp = gpar(fontsize = 7.5),
             ip_args = list(gp = gpar(fontsize = 7.5), id = TRUE, pval = TRUE),
             tp_args = list(gp = gpar(fontsize = 7.5)
                            , fill = c('gray88', 'green', 'red')
                            , id = TRUE
                            , beside = TRUE
                            , gap = 0
                            , rot = 0
                            , just = c('center', 'top'))
             )
      next
    }

    if (!is.null(unlist(t1log)$node.split.breaks)) { # check split in t1 tree
      plot(t1log, type = 'extended',
           main = res.feat[feature == i, descript],
           gp = gpar(fontsize = 7.5),
           ip_args = list(gp = gpar(fontsize = 7.5), id = TRUE, pval = TRUE),
           tp_args = list(gp = gpar(fontsize = 7.5)
                          , fill = c('gray88', 'green', 'red')
                          , id = TRUE
                          , beside = TRUE
                          , gap = 0
                          , rot = 0
                          , just = c('center', 'top'))
           )
      next
    }

    plot(t2, type = 'extended',
         main = res.feat[feature == i, descript],
         gp = gpar(fontsize = 7),
         ip_args = list(gp = gpar(fontsize = 7), id = TRUE, pval = TRUE),
         tp_args = list(gp = gpar(fontsize = 7)
                        , fill = c('gray88', 'green', 'red')
                        , id = TRUE
                        , beside = TRUE
                        , gap = 0
                        , rot = 0
                        , just = c('center', 'top')
         )
    )

  } else {
    print(paste('2', i))
    ## prepare datasets
    dt <- DT.res[, .N, keyby = .(value = get(i), zone)
                 ][, share := N/sum(N), by = .(value)]
    d <- DT.res[zone %in% c('any', 'high', 'low') & !is.na(get(i)),
                .(zone = ifelse(zone == 'high', 'h',
                                 ifelse(zone == 'low', 'l', 'a')),
                  value = get(i))
                ][ , zone := as.factor(as.character(zone))]

    ## grow  tree
    t2 <- ctree(zone ~ value, data = d,
                control = ctree_control(mincriterion = .99,
                                        minsplit = 200, #1000
                                        maxdepth = Inf,
                                        multiway = FALSE)
                )

    ## save data for rmd-report
    if (grepl('C:/', getwd()) == 1) {
      fwrite(dt,
             paste0('~/RProjects/201706-UnderOptim/UO.Clustering/Trees/dt2_',
                    i, '.csv'))
      fwrite(d,
             paste0('~/RProjects/201706-UnderOptim/UO.Clustering/Trees/d2_',
                    i, '.csv'))
      saveRDS(t2,
              paste0('~/RProjects/201706-UnderOptim/UO.Clustering/Trees/t2_',
                     i, '.rds'))
    } else {
      fwrite(dt,
             paste0('T:/RProjects/201706-UnderOptim/UO.Clustering/Trees/dt2_',
                    i, '.csv'))
      fwrite(d,
             paste0('T:/RProjects/201706-UnderOptim/UO.Clustering/Trees/d2_',
                    i, '.csv'))
      saveRDS(t2,
              paste0('T:/RProjects/201706-UnderOptim/UO.Clustering/Trees/t2_',
                     i, '.rds'))
    }

    ## plot barchart
    print(
    ggplot(data = dt) +
        geom_bar(aes(x = as.factor(value), y = share, fill = zone),
                 position = 'dodge', stat = 'identity') +
        scale_fill_manual(values = c('gray', 'green', 'red')) +
        scale_x_discrete(labels = scales::wrap_format(40)) +
        scale_y_continuous(labels = scales::percent, trans = 'sqrt',
                           limits = c(0, 1.03),
                           breaks = c(0, .01, .03, .05, .1, .15, .2, .3, .4, .5,
                                      .6, .8, 1)) +
        geom_text(aes(x = as.factor(value), y = share,
                      label = scales::percent(share), group = zone),
                  position = position_dodge(width=.9),
                  size = 3, hjust = -.05) +
        coord_flip() +
        labs(subtitle = paste0(res.feat[feature == i, descript],
                               ' (№', which(res.feat[, feature] == i),')')) +
        theme(legend.position = 'top',
              legend.key.size = unit(.4, 'cm'),
              axis.title.y = element_blank(),
              axis.text.y = element_text(size = 8),
              axis.title.x = element_blank(),
              axis.text.x = element_text(size = 7.5))
    )

    ## plot decision tree
    plot(t2, type = 'extended',
         main = res.feat[feature == i, descript],
         gp = gpar(fontsize = 7),
         ip_args = list(gp = gpar(fontsize = 7), id = TRUE, pval = TRUE),
         tp_args = list(gp = gpar(fontsize = 7)
                        , fill = c('gray88', 'green', 'red')
                        , id = TRUE
                        , beside = TRUE
                        , gap = 0
                        , rot = 0
                        , just = c('center', 'top')
                        )
         )
    }
}; rm(i, dt, d, t1, t1log, t2)
# END: IMPORTANT FEATURE VIZUALIZATION -----------------------------------------



# BEG: TREE ON ALL IMPORTANT FEATURES ------------------------------------------
## prepare datasets
dall <- DT.res[zone %in% c('any', 'high', 'low'),
              c('zone', 'dim.issued', 'was.ovd90.12m',
                'otkaz.who', 'otkaz.detail',
                'credit.sum',
                res.feat[imp.95 == 1, feature]), with = FALSE
              ][ , zone := ifelse(zone == 'high', 'h',
                                  ifelse(zone == 'low', 'l', 'a'))
                 ][ , zone := as.factor(as.character(zone))]

dall.f <- as.formula(paste('zone ~',
                           paste(setdiff(names(dall),
                                         c('zone', 'credit.sum',
                                           'dim.issued', 'was.ovd90.12m',
                                           'otkaz.who', 'otkaz.detail')),
                                 collapse = '+')))


## grow tree by partykit-package
# tall5 <- partykit::ctree(dall.f, data = dall,
#                          control = ctree_control(mincriterion = .999
#                                                  , minsplit = 100
#                                                  , maxdepth = 4
#                                                  , multiway = FALSE
#                                                  )
#                          )

## grow tree by rpart-package
tall5 <-  rpart(dall.f, data = dall, method = 'class',
                control = rpart.control (cp = .001
                                         , minsplit = 100
                                         , maxdepth = 5
                                         , xval = 10
                                         )
                )

## grow  tree by partykit-package
# tall5 <- ctree(dall.f, data = dall,
#             control = ctree_control(mincriterion = .999,
#                                     minsplit = 100, #1000
#                                     maxdepth = 5,
#                                     multiway = FALSE)
#             )


## plot tree
plot(#tall5,           # for tree by partykit-package
     as.party(tall5),  # for tree by rpart-package
     type = 'extended',
     gp = gpar(fontsize = 7),
     ip_args = list(gp = gpar(fontsize = 7), id = TRUE
                    , pval = FALSE
                    , abbreviate = FALSE
     ),
     tp_args = list(gp = gpar(fontsize = 6.5),
                    fill = c('gray88', 'green', 'red'),
                    beside = TRUE, ymax = 1, ylines = 1.2
                    #, widths = 1
                    , gap = 0
                    , reverse = NULL
                    , rot = 0
                    , just = c('center', 'top')
                    , id = FALSE 
                    #, mainlab = NULL
     ))

## save data for rmd-report
if (grepl('C:/', getwd()) == 1) {
  fwrite(dall,
         '~/RProjects/201706-UnderOptim/UO.Clustering/Trees/dall.csv')
  saveRDS(tall5,
          '~/RProjects/201706-UnderOptim/UO.Clustering/Trees/tall5.rds')
} else {
  fwrite(dall,
         'T:/RProjects/201706-UnderOptim/UO.Clustering/Trees/dall.csv')
  saveRDS(tall5,
          'T:/RProjects/201706-UnderOptim/UO.Clustering/Trees/tall5.rds')
}
print(tall5)
# END: TREE ON ALL IMPORTANT FEATURES ------------------------------------------



# BEG: REJECTION REASON FOR TERMINAL LEAVES ------------------------------------
#i <- 9
top <- 5

for (i in which(tall5$frame$var == '<leaf>')) {
### Approval & Default rate
print(cbind('Leaf.Num' = i,
  dall[tall5$where == i,
           .(.N,
             approval = mean(dim.issued)*100,
             was90 = mean(was.ovd90.12m)*100)][order(was90, -approval)]))

### High-zone reasons
dall.h.reas <- dall[zone == 'h' & dim.issued == 0 & tall5$where == i,
                    .(h.N = .N,
                      h.shareN = round(.N/dall[zone == 'h'
                                               & dim.issued == 0
                                               & tall5$where == i,
                                               .N]*100, 2)),
                    keyby = .(otkaz.detail)
                    ][order(-h.shareN)
                      ][1:ifelse(length(h.N) > top, top, length(h.N)),  ]
### Low-zone reasons
dall.l.reas <- dall[zone == 'l' & dim.issued == 0 & tall5$where == i,
                    .(l.N = .N,
                      l.shareN = round(.N/dall[zone == 'l'
                                               & dim.issued == 0
                                               & tall5$where == i,
                                               .N]*100, 2)),
                    keyby = .(otkaz.detail)
                    ][order(-l.shareN)
                      ][1:ifelse(length(l.N) > top, top, length(l.N)),  ] 
### Any-zone reasons
dall.a.reas <- dall[zone == 'a' & dim.issued == 0 & tall5$where == i,
                    .(a.N = .N,
                      a.shareN = round(.N/dall[zone == 'a'
                                               & dim.issued == 0
                                               & tall5$where == i,
                                               .N]*100, 2)),
                    keyby = .(otkaz.detail)
                    ][order(-a.shareN)
                      ][1:ifelse(length(a.N) > top, top, length(a.N)),  ] 
### All reasons
dall.all.reas <- dall[dim.issued == 0 & tall5$where == i,
                      .(all.N = .N,
                        all.shareN = round(.N/dall[dim.issued == 0
                                                   & tall5$where == i,
                                                   .N]*100, 2)),
                      keyby = .(otkaz.detail)
                      ][order(-all.shareN)
                        ][1:ifelse(length(all.N) > top, top, length(all.N)),  ] 
### Pick-up sumary table
dall.ttl.reas <- merge(dall.h.reas, dall.l.reas, by = 'otkaz.detail',
                       all = TRUE, sort = FALSE)
dall.ttl.reas <- merge(dall.ttl.reas, dall.a.reas, by = 'otkaz.detail',
                       all = TRUE, sort = FALSE)
dall.ttl.reas <- merge(dall.ttl.reas, dall.all.reas, by = 'otkaz.detail',
                       all = TRUE, sort = FALSE)

dall.ttl.reas[, lapply(.SD, function(x) {sum(x, na.rm = TRUE)}),
              .SDcols = -c('otkaz.detail')]
dall.ttl.reas <- rbind(cbind(otkaz.detail = 'ВСЕ ОТКАЗЫ в таблице:', 
                             dall.ttl.reas[,
                                           lapply(.SD, function(x) {
                                             sum(x, na.rm = TRUE)
                                           }),
                                           .SDcols = -c('otkaz.detail')]),
                       dall.ttl.reas)
dall.ttl.reas <- dall.ttl.reas[otkaz.detail != 'NA', ][order(-all.shareN)]
print(dall.ttl.reas)

    ## Save data for rmd-report
    if (grepl('C:/', getwd()) == 1) {
      write.table(dall.ttl.reas,
                  file = paste0('~/RProjects/201706-UnderOptim/UO.Clustering/Trees/dall.ttl.reas_leaf_',
                                i, '.csv'),
                  sep = ';', na = '-', dec = '.', row.names = FALSE,
                  fileEncoding = 'UTF-8')
    } else {
      write.table(dall.ttl.reas,
                  file = paste0('T:/RProjects/201706-UnderOptim/UO.Clustering/Trees/dall.ttl.reas_leaf_',
                                i, '.csv'),
                  sep = ';', na = '-', dec = '.', row.names = FALSE,
                  fileEncoding = 'UTF-8')
    }
}
rm(dall.h.reas, dall.l.reas, dall.a.reas, dall.all.reas, i, top)
# END: REJECTION REASON FOR TERMINAL LEAVES ------------------------------------



# BEG: SOM PREDICT -------------------------------------------------------------
## 1. Create dataset for prediction
DT.test.som <- as.matrix(DT.test[, lapply(.SD, function(x){
  scale(as.numeric(x))
  }),
  .SDcols = names(DT.test)[(names(DT.test) %in% f.som)]
  ])
dim(DT.test.som)

## 2. Predict test set
som.pred <- predict(som.fit, newdata = DT.test.som)
rm(DT.test.som)
saveRDS(som.pred, 'som.pred.rds')


## 3. Define the predicted zone
DT.test <- DT.test[, node := som.pred$unit.classif]
DT.test <- merge(DT.test, hilo[, .(node, zone)],
                 by = 'node',
                 all.x = TRUE, all.y = FALSE,
                 sort = FALSE)

## 4. Approved & default at predicted zones amount all credits
### on trained som
print(DT.res[, .(.N,
                 shareN = .N/DT.res[, .N]*100,
                 amountIssued = sum(dim.issued),
                 percApproval = mean(dim.issued)*100,
                 was90 = mean(was.ovd90.12m)*100),
             keyby = .(zone)
             ][, 'was90' := ifelse(was90 == 0, 0, was90/percApproval*100)])

### on test set
print(DT.test[zone != 'denied',
              .(.N,
                shareN = .N/DT.test[zone != 'denied', .N]*100,
                amountIssued = sum(dim.issued),
                percApproval = mean(dim.issued)*100,
                was90 = mean(was.ovd90.12m)*100),
              keyby = .(zone)
              ][, 'was90' := ifelse(was90 == 0, 0, was90/percApproval*100)])

## 5. Plot predicted zones
### data
gpred <- DT.test[, .(cases.N = .N,
                     approval = mean(dim.issued)*100,
                     was90 = mean(was.ovd90.12m)*100),
                 keyby = .(zone, node)
                 ][, 'was90' := ifelse(was90 == 0, 0, was90/approval*100)]
## plot
ggplot(gpred[zone != 'denied'], aes(x = approval, y = was90, color = zone)) +
  geom_point(aes(size = cases.N), alpha = .3) +
  labs(title = 'On test set predicted nodes & zones'
       , x = 'Approval, %', y = 'Was.ovd90.12m, %') +
  geom_hline(yintercept = gpred[, quantile(was90, probs = probs)][1],
             linetype = 2, size = .5, color = 'red4') +
  geom_hline(yintercept = gpred[, quantile(was90, probs = probs)][2],
             linetype = 2, size = .5, color = 'red4') +
  geom_hline(yintercept = DT.full[dim.issued == 1, mean(was.ovd90.12m)]*100,
             linetype = 1, size = .5, color = 'red4') +
  geom_vline(xintercept = gpred[, quantile(approval, probs = probs)][1],
             linetype = 2, size = .5, color = 'green4') +
  geom_vline(xintercept = gpred[, quantile(approval, probs = probs)][2],
             linetype = 2, size = .5, color = 'green4') +
  geom_vline(xintercept = DT.full[, mean(dim.issued)]*100,
             linetype = 1, size = .5, color = 'green4') +
  geom_rug(sides = 'tr') +
  scale_color_manual(breaks = c('high', 'low', 'any'),
                     values = c('gray77', 'green4', 'red4')) +
  scale_y_continuous(trans = 'sqrt',
                     breaks = sort(c(seq(0, gpred[, max(was90)], 5), 1, 2, 3))) +
  scale_x_continuous(breaks = sort(c(seq(0, gpred[, max(approval)], 5))))
# END: SOM PREDICT -------------------------------------------------------------



# BEG: COMMON REJECTION REASON -------------------------------------------------
## 1. HIGH zone
### Refusal source
ref.hi.source <- DT.res[zone == 'high' & dim.issued == 0,
                        .(high.N = .N,
                          high.shareN = round(.N/DT.res[zone == 'high'
                                                  & dim.issued == 0,
                                                  .N]*100, 2)),
                        keyby = .(otkaz.who)][order(-high.shareN)]
### Refusal reason TOP-33
ref.hi.reason <- DT.res[zone == 'high' & dim.issued == 0,
                    .(high.N = .N,
                      high.shareN = round(.N/DT.res[zone == 'high'
                                                    & dim.issued == 0,
                                                    .N]*100, 2)),
                    keyby = .(otkaz.detail)][order(-high.shareN)][1:33, ]


## 2. LOW zone
### Refusal source
ref.lo.source <- DT.res[zone == 'low' & dim.issued == 0,
                        .(low.N = .N,
                          low.shareN = round(.N/DT.res[zone == 'low'
                                                  & dim.issued == 0,
                                                  .N]*100, 2)),
                        keyby = .(otkaz.who)][order(-low.shareN)]
### Refusal reason TOP-33
ref.lo.reason <- DT.res[zone == 'low' & dim.issued == 0,
                        .(low.N = .N,
                          low.shareN = round(.N/DT.res[zone == 'low' 
                                                 & dim.issued == 0,
                                                 .N]*100, 2)),
                        keyby = .(otkaz.detail)][order(-low.shareN)][1:33, ]


## 3. ANY zone
### Refusal source
ref.an.source <- DT.res[zone == 'any' & dim.issued == 0,
                        .(any.N = .N,
                          any.shareN = round(.N/DT.res[zone == 'any'
                                                 & dim.issued == 0,
                                                 .N]*100, 2)),
                        keyby = .(otkaz.who)][order(-any.shareN)]
### Refusal reason TOP-33
ref.an.reason <- DT.res[zone == 'any' & dim.issued == 0,
                        .(any.N = .N,
                          any.shareN = round(.N/DT.res[zone == 'any'
                                                 & dim.issued == 0,
                                                 .N]*100, 2)),
                        keyby = .(otkaz.detail)][order(-any.shareN)][1:33, ]


## 4. Summary table
### Refusal source
ref.all.source <- DT.res[dim.issued == 0,
                        .(all.N = .N,
                          all.shareN = round(.N/DT.res[dim.issued == 0,
                                                       .N]*100, 2)),
                        keyby = .(otkaz.who)][order(-all.shareN)]
ref.ttl.source <- merge(ref.hi.source, ref.lo.source, by = 'otkaz.who',
                        all = TRUE, sort = FALSE)
ref.ttl.source <- merge(ref.ttl.source, ref.an.source, by = 'otkaz.who',
                        all = TRUE, sort = FALSE)
ref.ttl.source <- merge(ref.ttl.source, ref.all.source, by = 'otkaz.who',
                        all = TRUE, sort = FALSE)

ref.ttl.source[, lapply(.SD, function(x) {sum(x, na.rm = TRUE)}),
               .SDcols = -c('otkaz.who')]

ref.ttl.source <- rbind(cbind(otkaz.who = 'ВСЕ ОТКАЗЫ в таблице', 
                              ref.ttl.source[,
                                             lapply(.SD, function(x) {
                                               sum(x, na.rm = TRUE)
                                               }),
                                             .SDcols = -c('otkaz.who')]),
                        ref.ttl.source)
ref.ttl.source <- ref.ttl.source[order(-all.shareN)]
rm(ref.hi.source, ref.lo.source, ref.an.source, ref.all.source)

### Refusal reason TOP-33
ref.all.reason <- DT.res[dim.issued == 0,
                  .(all.N = .N,
                    all.shareN = round(.N/DT.res[dim.issued == 0,
                                                 .N]*100, 2)),
                  keyby = .(otkaz.detail)][order(-all.shareN)][1:33, ]
ref.ttl.reason <- merge(ref.hi.reason, ref.lo.reason, by = 'otkaz.detail',
                 all = TRUE, sort = FALSE)
ref.ttl.reason <- merge(ref.ttl.reason, ref.an.reason, by = 'otkaz.detail',
                 all = TRUE, sort = FALSE)
ref.ttl.reason <- merge(ref.ttl.reason, ref.all.reason, by = 'otkaz.detail',
                 all = TRUE, sort = FALSE)

ref.ttl.reason[, lapply(.SD, function(x) {sum(x, na.rm = TRUE)}),
               .SDcols = -c('otkaz.detail')]

ref.ttl.reason <- rbind(cbind(otkaz.detail = 'ВСЕ ОТКАЗЫ', 
                              ref.ttl.reason[,
                                             lapply(.SD, function(x) {
                                               sum(x, na.rm = TRUE)
                                             }),
                                             .SDcols = -c('otkaz.detail')]),
                        ref.ttl.reason)
ref.ttl.reason <- ref.ttl.reason[order(-all.shareN)]
rm(ref.hi.reason, ref.lo.reason, ref.an.reason, ref.all.reason)

## 6. Save the result.feat and lgb-model for rmd-report
write.table(ref.ttl.source, file = 'ref_ttl_source.csv', sep = ';', na = '-',
            dec = '.', row.names = FALSE, fileEncoding = 'UTF-8')
write.table(ref.ttl.reason, file = 'ref_ttl_reason.csv', sep = ';', na = '-',
            dec = '.', row.names = FALSE, fileEncoding = 'UTF-8')

# END: COMMON REJECTION REASON -------------------------------------------------
save.image()

########################## DONE LINE ###########################################


#### some else code ---------------------------------------------------------
start.time <- Sys.time()
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken


DT.full[, .(.N, mean(was.ovd90.12m)*100), by = 'dim.segment'][order(-N)]

summary(DT.train[, as.factor(bscor.max.limit.util)])
unique(DT.train[, bscor.max.limit.util])
unique(DT.train[, na.omit(bscor.max.limit.util)])

DT.train[, summary(as.factor(svyazan.v.chs.ki))]
DT.f[feature == 'svyazan.v.chs.ki', .(svyazan.v.chs.ki, descript)]

DT.train[, summary(decision)]
DT.train[, mean(decision)]

DT.train[, .N, keyby = .(dim.preapproved, dim.issued)]

length(som.fit$unit.classif)
length(sort(unique(som.fit$unit.classif)))

DT.train[, som := som.fit$unit.classif]
DT.train[, mean(credit.sum, na.rm = TRUE), keyby = .(som)]

DT.train[, gdata::mapLevels(dim.segment)]
getAnywhere(plot.kohonen)
getAnywhere(supersom)



library(rattle)
rattle()
