# This script for analysing time cache of BKI requests
## We take requests which had

save.image()
# First deals
rm(list=ls())
gc()
#opar <- par(no.readonly = TRUE)
options(scipen=9999)

# Load libraries
library(data.table)
library(ggplot2)
time.start <- Sys.time()
# BEG: Load & view data --------------------------------------------------------
## 1. load data
if (grepl('C:/', getwd()) == 1) {
  DT <- fread('~/RProjects/201706-UnderOptim/DS_CacheBKI_20171225_notNA_(20171201).csv',
              encoding = 'UTF-8', na.strings = c('', 'NA', 'N/A')
              #, nrows = 697496
              #, skip = 0
                  , skip = 697497
              )
  field.names <- fread('~/RProjects/201706-UnderOptim/DS_CacheBKI_20171225_notNA_(20171201).csv',
              encoding = 'UTF-8', na.strings = c('', 'NA', 'N/A'), nrows = 0)

  #DT.f <- fread('~/RProjects/201706-UnderOptim/DS_FieldDescript.csv')

} else {
  DT <- fread('T:/RProjects/201706-UnderOptim/DS_CacheBKI_20171225_notNA_(20171201).csv',
              encoding = 'UTF-8', na.strings = c('', 'NA', 'N/A')
              , nrows = 697496
              , skip = 0
              #, skip = 697497
              )
  #DT.f <- fread('T:/RProjects/201706-UnderOptim/DS_FieldDescript.csv')
}

setnames(DT, names(field.names))
#DT.f[, feature := gsub('_', '.', tolower(feature))]
(names(DT) <- gsub('_', '.', tolower(names(DT))))

## 2. view min-max request date
DT[, max(as.Date(gsub('01.01.1900', NA, gsub("[-/]", ".", date.request)), '%d.%m.%Y'))]
DT[, min(as.Date(gsub('01.01.1900', NA, gsub("[-/]", ".", date.request)), '%d.%m.%Y'))]

#DT.f[feature %in% names(DT) & group == 'bki', .(feature, descript)]
rm(field.names)
# END: Load & view data --------------------------------------------------------



# BEG: Clean & Prepare dataset -------------------------------------------------
### rename PAS to ID
setnames(DT, 'pas', 'id')
## 1. Determ the fields
### numeric
x.num0 <- c('credit.sum', 'age')
x.num1 <- c(paste0('creditsum', sprintf('%02d', 1:40)))
x.num2 <- c(paste0('nextpayment', sprintf('%02d', 1:40)))
x.num3 <- c(paste0('credsumoverdue', sprintf('%02d', 1:40)))
x.num4 <- c(paste0('credsumdebt', sprintf('%02d', 1:40)))
x.num5 <- c(paste0('delay0.', sprintf('%02d', 1:40)))
x.num6 <- c(paste0('delay30.', sprintf('%02d', 1:40)))
x.num7 <- c(paste0('delay60.', sprintf('%02d', 1:40)))
x.num8 <- c(paste0('delay90.', sprintf('%02d', 1:40)))
x.num9 <- c(paste0('delay0.full.', sprintf('%02d', 1:40)))
x.num10 <- c(paste0('delay30.full.', sprintf('%02d', 1:40)))
x.num11 <- c(paste0('delay60.full.', sprintf('%02d', 1:40)))
x.num12 <- c(paste0('delay90.full.', sprintf('%02d', 1:40)))
gc()

### character
x.char <- c('id', 'id.request', 'rbo.client.id',
            paste0('credactive', sprintf('%02d', 1:40))
            )
### binary
x.bin <- c('was.ovd90.12m', 'dim.issued', 'issued1day')


## 2. Coerce fields to needed class
### date
DT[, date.request:= as.POSIXct(date.request, format = '%d.%m.%Y %H:%M:%S')]
DT[, date.of := as.Date(gsub('01.01.1900', NA, gsub("[-/]", ".", date.of)),
                        '%d.%m.%Y')]

### replace ',' to '.' in x.num and coerce to "numeric" class
DT[, (x.num0) := lapply(.SD, function(x) {
  as.numeric(sub(',', '.', x))
}), .SDcols = (x.num0)]

DT[, (x.num1) := lapply(.SD, function(x) {
  as.numeric(sub(',', '.', x))
}), .SDcols = (x.num1)]

DT[, (x.num2) := lapply(.SD, function(x) {
  as.numeric(sub(',', '.', x))
}), .SDcols = (x.num2)]

DT[, (x.num3) := lapply(.SD, function(x) {
  as.numeric(sub(',', '.', x))
}), .SDcols = (x.num3)]

DT[, (x.num4) := lapply(.SD, function(x) {
  as.numeric(sub(',', '.', x))
}), .SDcols = (x.num4)]

DT[, (x.num5) := lapply(.SD, function(x) {
  as.numeric(sub(',', '.', x))
}), .SDcols = (x.num5)]

DT[, (x.num6) := lapply(.SD, function(x) {
  as.numeric(sub(',', '.', x))
}), .SDcols = (x.num6)]

DT[, (x.num7) := lapply(.SD, function(x) {
  as.numeric(sub(',', '.', x))
}), .SDcols = (x.num7)]

DT[, (x.num8) := lapply(.SD, function(x) {
  as.numeric(sub(',', '.', x))
}), .SDcols = (x.num8)]

DT[, (x.num9) := lapply(.SD, function(x) {
  as.numeric(sub(',', '.', x))
}), .SDcols = (x.num9)]

DT[, (x.num10) := lapply(.SD, function(x) {
  as.numeric(sub(',', '.', x))
}), .SDcols = (x.num10)]

DT[, (x.num11) := lapply(.SD, function(x) {
  as.numeric(sub(',', '.', x))
}), .SDcols = (x.num11)]

DT[, (x.num12) := lapply(.SD, function(x) {
  as.numeric(sub(',', '.', x))
}), .SDcols = (x.num12)]
gc()

### to character
DT[, (x.char) := lapply(.SD, function(x) {
  as.character(x)
}), .SDcols = (x.char)]

### coerce binary to "integer" class
DT[, (x.bin) := lapply(.SD, function(x) {
  as.integer(x)
}), .SDcols = (x.bin)]

rm(x.num0, x.num1, x.num2, x.num3, x.num4, x.num5, x.num6, x.num7, x.num8,
   x.num9, x.num10, x.num11, x.num12, x.char, x.bin)
gc()

table(DT[, sapply(.SD, class), .SDcols = (names(DT)[names(DT) != 'date.request'])])
# END: Clean & Prepare dataset -------------------------------------------------

DT[1:30, .(id, date.request)]

# BEG: Calculations BKI SUMMARY INFO -------------------------------------------
### view all BKI names
#unique(unlist(DT[, paste0('bki.name', sprintf('%02d', 1:40)), with = FALSE]))
#c('НБКИ', 'ОКБ', 'Эквифакс', 'КБРС', 'РБО', 'Анкета', 'Андер-р')

gc()
## 1. By SEVERAL BKI
#i <- 'НБКИ'
for (i in c('НБКИ', 'ОКБ', 'Эквифакс', 'КБРС')) {
  print(i)
  (ii <- switch(i, НБКИ = 'nbki', ОКБ = 'okb', Эквифакс = 'equi', КБРС = 'kbrs'))

  # get the matrix with bki.name~creditNUMBER
  mx <- ifelse(DT[, paste0('bki.name', sprintf('%02d', 1:40)), with = FALSE] == i,
               TRUE, NA)

  # (1) CREDITSUM: get SUM vector
  sum.creditsum <-
    apply(DT[, paste0('creditsum', sprintf('%02d', 1:40)), with = FALSE] * mx,
          1, FUN = function(x){
            ifelse(all(is.na(x)), NA, sum(x, na.rm = TRUE))
            })
  DT[, paste0(ii, '.sum') := as.numeric(sum.creditsum)]
  print('sum.creditsum - DONE')

  # (2) CREDACTIVE: get SURVEY INFO vector
  sum.credactive <-
    apply(
      as.data.table(ifelse(as.matrix(mx),
                           as.matrix(DT[,
                                        paste0('credactive', sprintf('%02d', 1:40)),
                                        with = FALSE]),
                           NA)),
      1, FUN = function(x){
        ifelse(is.null(names(table(x))), NA,
               paste0(as.vector(table(x)), ':',
                      names(table(x)), sep = '', collapse = '*')
              )
        }
    )

  sum.active <-
    apply(DT[, paste0('creditsum', sprintf('%02d', 1:40)), with = FALSE] *
            (as.data.table(ifelse(as.matrix(mx),
                                 as.matrix(DT[,
                                              paste0('credactive', sprintf('%02d', 1:40)),
                                              with = FALSE]),
                                 NA)) == 'Активен'),
          1, FUN = function(x){
            ifelse(all(is.na(x)), NA, sum(x, na.rm = TRUE))
          }
    )
  DT[, paste0(ii, '.sum.active') := as.numeric(sum.active)]

  count.active <-
    apply(as.data.table(ifelse(as.matrix(mx),
                               as.matrix(DT[,
                                            paste0('credactive', sprintf('%02d', 1:40)),
                                            with = FALSE]),
                               NA)) == 'Активен',
          1, FUN = function(x){
            ifelse(all(is.na(x)), NA, sum(x, na.rm = TRUE))
            }
    )
  DT[, paste0(ii, '.count.active') := as.numeric(count.active)]
  print('sum.credactive - DONE')

  # (3) NEXTPAYMENT: get SUM vector
  sum.nextpayment <-
    apply(DT[, paste0('nextpayment', sprintf('%02d', 1:40)), with = FALSE] * mx,
          1, FUN = function(x){
            ifelse(all(is.na(x)), NA, sum(x, na.rm = TRUE))
          })
  DT[, paste0(ii, '.payment') := as.numeric(sum.nextpayment)]
  print('sum.nextpayment - DONE')

  # (4) CREDSUMOVERDUE: get SUM vector
  sum.credsumoverdue <-
    apply(DT[, paste0('credsumoverdue', sprintf('%02d', 1:40)), with = FALSE] * mx,
          1, FUN = function(x){
            ifelse(all(is.na(x)), NA, sum(x, na.rm = TRUE))
          })
  DT[, paste0(ii, '.sum.overdue') := as.numeric(sum.credsumoverdue)]
  print('sum.credsumoverdue - DONE')

  # (5) DELAY0: get SUM vector
  sum.delay.0 <-
    apply(DT[, paste0('delay0.', sprintf('%02d', 1:40)), with = FALSE] * mx,
          1, FUN = function(x){
            ifelse(all(is.na(x)), NA, sum(x, na.rm = TRUE))
          })
  print('sum.delay.0 - DONE')

  # (6) DELAY30: get SUM vector
  sum.delay.30 <-
    apply(DT[, paste0('delay30.', sprintf('%02d', 1:40)), with = FALSE] * mx,
          1, FUN = function(x){
            ifelse(all(is.na(x)), NA, sum(x, na.rm = TRUE))
          })
  print('sum.delay.30 - DONE')

  # (7) DELAY60: get SUM vector
  sum.delay.60 <-
    apply(DT[, paste0('delay60.', sprintf('%02d', 1:40)), with = FALSE] * mx,
          1, FUN = function(x){
            ifelse(all(is.na(x)), NA, sum(x, na.rm = TRUE))
          })
  print('sum.delay.60 - DONE')

  # (8) DELAY90: get SUM vector
  sum.delay.90 <-
    apply(DT[, paste0('delay90.', sprintf('%02d', 1:40)), with = FALSE] * mx,
          1, FUN = function(x){
            ifelse(all(is.na(x)), NA, sum(x, na.rm = TRUE))
          })
  print('sum.delay.90 - DONE')

  # (8-1) DELAY: get SUM vector
  count.delay <-
    apply(data.table(sum.delay.0, sum.delay.30, sum.delay.60, sum.delay.90),
          1, FUN = function(x){
            ifelse(all(is.na(x)), NA, sum(x, na.rm = TRUE))
          })
  DT[, paste0(ii, '.count.delay') := count.delay]
  print('count.delay - DONE')

  # (9) CREDSUMDEBT: get SUM vector
  sum.credsumdebt <-
    apply(DT[, paste0('credsumdebt', sprintf('%02d', 1:40)), with = FALSE] * mx,
          1, FUN = function(x){
            ifelse(all(is.na(x)), NA, sum(x, na.rm = TRUE))
          })
  DT[, paste0(ii, '.sum.debt') := as.numeric(sum.credsumdebt)]
  print('sum.credsumdebt - DONE')

  # (10) DELAY0_FULL: get SUM vector
  sum.delay.0.full <-
    apply(DT[, paste0('delay0.full.', sprintf('%02d', 1:40)), with = FALSE] * mx,
          1, FUN = function(x){
            ifelse(all(is.na(x)), NA, sum(x, na.rm = TRUE))
          })
  print('sum.delay.0.full - DONE')

  # (11) DELAY30_FULL: get SUM vector
  sum.delay.30.full <-
    apply(DT[, paste0('delay30.full.', sprintf('%02d', 1:40)), with = FALSE] * mx,
          1, FUN = function(x){
            ifelse(all(is.na(x)), NA, sum(x, na.rm = TRUE))
          })
  print('sum.delay.30.full - DONE')

  # (12) DELAY60_FULL: get SUM vector
  sum.delay.60.full <-
    apply(DT[, paste0('delay60.full.', sprintf('%02d', 1:40)), with = FALSE] * mx,
          1, FUN = function(x){
            ifelse(all(is.na(x)), NA, sum(x, na.rm = TRUE))
          })
  print('sum.delay.60.full - DONE')

  # (13) DELAY90_FULL: get SUM vector
  sum.delay.90.full <-
    apply(DT[, paste0('delay90.full.', sprintf('%02d', 1:40)), with = FALSE] * mx,
          1, FUN = function(x){
            ifelse(all(is.na(x)), NA, sum(x, na.rm = TRUE))
          })
  print('sum.delay.90.full - DONE')

  # (13-1) DELAY_FULL: get SUM vector
  count.delay.full <-
    apply(data.table(sum.delay.0.full, sum.delay.30.full,
                     sum.delay.60.full, sum.delay.90.full),
          1, FUN = function(x){
            ifelse(all(is.na(x)), NA, sum(x, na.rm = TRUE))
          })
  DT[, paste0(ii, '.count.delay.full') := count.delay.full]
  print('count.delay.full - DONE')

  # (14) TOTAL vector of bki info
  DT[, paste0(ii, '.ttl') := ifelse(paste(sum.creditsum,
                                          sum.credactive,
                                          sum.nextpayment,
                                          sum.credsumoverdue,
                                          sum.delay.0,
                                          sum.delay.30,
                                          sum.delay.60,
                                          sum.delay.90,
                                          sum.credsumdebt,
                                          sum.delay.0.full,
                                          sum.delay.30.full,
                                          sum.delay.60.full,
                                          sum.delay.90.full,
                                          sep = '|') ==
                                      'NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA|NA',
                                    'allNA',
                                    paste(sum.creditsum,
                                          sum.credactive,
                                          sum.nextpayment,
                                          sum.credsumoverdue,
                                          sum.delay.0,
                                          sum.delay.30,
                                          sum.delay.60,
                                          sum.delay.90,
                                          sum.credsumdebt,
                                          sum.delay.0.full,
                                          sum.delay.30.full,
                                          sum.delay.60.full,
                                          sum.delay.90.full,
                                          sep = '|'))]
  print(paste0(ii, '.ttl', ' - DONE'))

  rm(i, ii, mx,
     sum.creditsum, sum.credactive, count.active, sum.active,
     sum.nextpayment, sum.credsumoverdue,
     sum.delay.0, sum.delay.30, sum.delay.60, sum.delay.90,
     count.delay,
     sum.credsumdebt,
     sum.delay.0.full, sum.delay.30.full, sum.delay.60.full, sum.delay.90.full,
     count.delay.full)
  gc()
}
print('Loop is done!')


## 2. Summary for ALL BKI
### (1) Total: CREDITSUM
DT[, allbki.sum := apply(data.table(nbki.sum, okb.sum, equi.sum, kbrs.sum),
                         1, function(x) {sum(x, na.rm = TRUE)})]

### (2) Total: SUM OF ACTIVE
DT[, allbki.sum.active := apply(data.table(nbki.sum.active, okb.sum.active,
                                    equi.sum.active, kbrs.sum.active),
                         1, function(x) {sum(x, na.rm = TRUE)})]

### (3) Total: COUNT OF ACTIVE
DT[, allbki.count.active := apply(data.table(nbki.count.active, okb.count.active,
                                           equi.count.active, kbrs.count.active),
                                1, function(x) {sum(x, na.rm = TRUE)})]

### (4) Total: PAYMENT
DT[, allbki.payment := apply(data.table(nbki.payment, okb.payment,
                                        equi.payment, kbrs.payment),
                                  1, function(x) {sum(x, na.rm = TRUE)})]

### (5) Total: SUM OF OVERDUE
DT[, allbki.sum.overdue := apply(data.table(nbki.sum.overdue, okb.sum.overdue,
                                        equi.sum.overdue, kbrs.sum.overdue),
                             1, function(x) {sum(x, na.rm = TRUE)})]

### (6) Total: COUNT OF DELAY
DT[, allbki.count.delay := apply(data.table(nbki.count.delay, okb.count.delay,
                                            equi.count.delay, kbrs.count.delay),
                                 1, function(x) {sum(x, na.rm = TRUE)})]

### (7) Total: SUM OF DEBT
DT[, allbki.sum.debt := apply(data.table(nbki.sum.debt, okb.sum.debt,
                                            equi.sum.debt, kbrs.sum.debt),
                                 1, function(x) {sum(x, na.rm = TRUE)})]

### (8) Total: SUM OF DEBT
DT[, allbki.count.delay.full := apply(data.table(nbki.count.delay.full,
                                                 okb.count.delay.full,
                                                 equi.count.delay.full,
                                                 kbrs.count.delay.full),
                              1, function(x) {sum(x, na.rm = TRUE)})]

### (9) Total for all bki
DT[, allbki.ttl := paste('nbki(', nbki.ttl, ')::',
                         'okb(', okb.ttl, ')::',
                         'equi(', equi.ttl, ')::',
                         'kbrs(', kbrs.ttl, ')',
                         sep = '')]


## 3. Calculate OVERDUE by RBO active credit
# rbo.mx <- ifelse(DT[, paste0('bki.name', sprintf('%02d', 1:40)), with = FALSE] == 'РБО',
#                  TRUE, NA)
# rbo.sumoverdue.active <-
#   apply(DT[, paste0('credsumoverdue', sprintf('%02d', 1:40)), with = FALSE] *
#           (as.data.table(ifelse(as.matrix(rbo.mx),
#                                 as.matrix(DT[,
#                                              paste0('credactive', sprintf('%02d', 1:40)),
#                                              with = FALSE]),
#                                 NA)) == 'Активен'),
#         1, FUN = function(x){
#           ifelse(all(is.na(x)), NA, sum(x, na.rm = TRUE))
#         }
#   )
# DT[, rbo.sum.overdue.active := as.numeric(rbo.sumoverdue.active)]
# rm(rbo.mx, rbo.sumoverdue.active)
gc()
#DT[, rbo.sum.overdue.active := NULL]
# END: Calculations BKI SUMMARY INFO -------------------------------------------

DT[1:30, .(id, date.request)]

# BEG: Delete redundant fields -------------------------------------------------
x.del <- c(paste0('creditsum', sprintf('%02d', 1:40)),
           paste0('nextpayment', sprintf('%02d', 1:40)),
           paste0('credsumoverdue', sprintf('%02d', 1:40)),
           paste0('credsumdebt', sprintf('%02d', 1:40)),
           paste0('delay0.', sprintf('%02d', 1:40)),
           paste0('delay30.', sprintf('%02d', 1:40)),
           paste0('delay60.', sprintf('%02d', 1:40)),
           paste0('delay90.', sprintf('%02d', 1:40)),
           paste0('delay0.full.', sprintf('%02d', 1:40)),
           paste0('delay30.full.', sprintf('%02d', 1:40)),
           paste0('delay60.full.', sprintf('%02d', 1:40)),
           paste0('delay90.full.', sprintf('%02d', 1:40)),
           paste0('credactive', sprintf('%02d', 1:40)),
           paste0('bki.name', sprintf('%02d', 1:40))
)
names(DT)[!names(DT) %in% x.del]
DT[, (x.del) := NULL]
rm(x.del)
# END: Delete redundant fields -------------------------------------------------



# BEG: Fill 'allNA' isssues beetwen same issues --------------------------------
## STEP 1: Create temporary field [bin.bki] - bins between results from BKI
### NBKI:
DT[nbki.ttl != 'allNA', nbki.bin := seq_len(.N), by = .(id)
   ][, nbki.bin := zoo::na.locf(nbki.bin, na.rm = FALSE, fromLast = FALSE),
     by = .(id)]
### OKB:
DT[okb.ttl != 'allNA', okb.bin := seq_len(.N), by = .(id)
   ][, okb.bin := zoo::na.locf(okb.bin, na.rm = FALSE, fromLast = FALSE),
     by = .(id)]
### EQUI:
DT[equi.ttl != 'allNA', equi.bin := seq_len(.N), by = .(id)
   ][, equi.bin := zoo::na.locf(equi.bin, na.rm = FALSE, fromLast = FALSE),
     by = .(id)]
### KBRS:
DT[kbrs.ttl != 'allNA', kbrs.bin := seq_len(.N), by = .(id)
   ][, kbrs.bin := zoo::na.locf(kbrs.bin, na.rm = FALSE, fromLast = FALSE),
     by = .(id)]
### ALL BKI:
DT[allbki.ttl != 'nbki(allNA)::okb(allNA)::equi(allNA)::kbrs(allNA)',
   allbki.bin := seq_len(.N), by = .(id)
   ][, allbki.bin := zoo::na.locf(allbki.bin, na.rm = FALSE, fromLast = FALSE),
     by = .(id)]
# DT[id %in% DT[, .N, by = id][order(-N)][c(2), id], .(id, date.request, nbki.ttl, nbki.bin)]


## STEP 2: Shift up the next bki info
### NBKI:
DT[nbki.ttl != 'allNA', nbki.ttl.shift := shift(nbki.ttl, n = 1, type = 'lead'),
   by = .(id)]
### OKB:
DT[okb.ttl != 'allNA', okb.ttl.shift := shift(okb.ttl, n = 1, type = 'lead'),
   by = .(id)]
### EQUI:
DT[equi.ttl != 'allNA', equi.ttl.shift := shift(equi.ttl, n = 1, type = 'lead'),
   by = .(id)]
### KBRS:
DT[kbrs.ttl != 'allNA', kbrs.ttl.shift := shift(kbrs.ttl, n = 1, type = 'lead'),
   by = .(id)]
### ALL BKI:
DT[allbki.ttl != 'nbki(allNA)::okb(allNA)::equi(allNA)::kbrs(allNA)',
   allbki.ttl.shift := shift(allbki.ttl, n = 1, type = 'lead'),
   by = .(id)]
# DT[id %in% DT[, .N, by = id][order(-N)][c(2), id], .(id, date.request, nbki.ttl, nbki.bin, nbki.ttl.shift)]


## STEP 3: Get ID&bki.BIN beetwen !='allNA': create tables with wanted index
### NBKI:
nbki.idx <- # inner join DT with
  DT[nbki.ttl == nbki.ttl.shift, # wanted number of group
     .(id, nbki.bin, nbki.sum, nbki.sum.active, nbki.count.active, nbki.payment,
       nbki.sum.overdue, nbki.count.delay, nbki.sum.debt, nbki.count.delay.full,
       nbki.ttl)
     ][                                                        # whith
       DT[!is.na(nbki.bin), .N, by = .(id, nbki.bin)][N > 1],  # [id & nbki.bin] that more then 1 row
       on = .(id, nbki.bin),
       nomatch=0
       ]
### OKB:
okb.idx <- # inner join DT with
  DT[okb.ttl == okb.ttl.shift, # wanted number of group
     .(id, okb.bin, okb.sum, okb.sum.active, okb.count.active, okb.payment,
       okb.sum.overdue, okb.count.delay, okb.sum.debt, okb.count.delay.full,
       okb.ttl)
     ][                                                        # whith
       DT[!is.na(okb.bin), .N, by = .(id, okb.bin)][N > 1],  # [id & okb.bin] that more then 1 row
       on = .(id, okb.bin),
       nomatch=0
       ]
### EQUI:
equi.idx <- # inner join DT with
  DT[equi.ttl == equi.ttl.shift, # wanted number of group
     .(id, equi.bin, equi.sum, equi.sum.active, equi.count.active, equi.payment,
       equi.sum.overdue, equi.count.delay, equi.sum.debt, equi.count.delay.full,
       equi.ttl)
     ][                                                        # whith
       DT[!is.na(equi.bin), .N, by = .(id, equi.bin)][N > 1],  # [id & equi.bin] that more then 1 row
       on = .(id, equi.bin),
       nomatch=0
       ]
### KBRS:
kbrs.idx <- # inner join DT with
  DT[kbrs.ttl == kbrs.ttl.shift, # wanted number of group
     .(id, kbrs.bin, kbrs.sum, kbrs.sum.active, kbrs.count.active, kbrs.payment,
       kbrs.sum.overdue, kbrs.count.delay, kbrs.sum.debt, kbrs.count.delay.full,
       kbrs.ttl)
     ][                                                        # whith
       DT[!is.na(kbrs.bin), .N, by = .(id, kbrs.bin)][N > 1],  # [id & kbrs.bin] that more then 1 row
       on = .(id, kbrs.bin),
       nomatch=0
       ]
### ALL BKI:
allbki.idx <- # inner join DT with
  DT[allbki.ttl == allbki.ttl.shift, # wanted number of group
     .(id, allbki.bin, allbki.sum, allbki.sum.active, allbki.count.active, allbki.payment,
       allbki.sum.overdue, allbki.count.delay, allbki.sum.debt, allbki.count.delay.full,
       allbki.ttl)
     ][                                                        # whith
       DT[!is.na(allbki.bin), .N, by = .(id, allbki.bin)][N > 1],  # [id & allbki.bin] that more then 1 row
       on = .(id, allbki.bin),
       nomatch=0
       ]
DT[nbki.ttl != 'allNA', .N]
DT[okb.ttl != 'allNA', .N]
DT[equi.ttl != 'allNA', .N]
DT[kbrs.ttl != 'allNA', .N]
DT[allbki.ttl != 'nbki(allNA)::okb(allNA)::equi(allNA)::kbrs(allNA)', .N]


## STEP 4: Get ID&bki.BIN beetwen !='allNA': and fill the gaps contained 'allNA'
### NBKI:
DT[nbki.idx, on = .(id, nbki.bin),
   ':=' (nbki.sum = i.nbki.sum,
         nbki.sum.active = i.nbki.sum.active,
         nbki.count.active = i.nbki.count.active,
         nbki.payment = i.nbki.payment,
         nbki.sum.overdue = i.nbki.sum.overdue,
         nbki.count.delay = i.nbki.count.delay,
         nbki.sum.debt = i.nbki.sum.debt,
         nbki.count.delay.full = i.nbki.count.delay.full,
         nbki.ttl = i.nbki.ttl)
   ]
### OKB:
DT[okb.idx, on = .(id, okb.bin),
   ':=' (okb.sum = i.okb.sum,
         okb.sum.active = i.okb.sum.active,
         okb.count.active = i.okb.count.active,
         okb.payment = i.okb.payment,
         okb.sum.overdue = i.okb.sum.overdue,
         okb.count.delay = i.okb.count.delay,
         okb.sum.debt = i.okb.sum.debt,
         okb.count.delay.full = i.okb.count.delay.full,
         okb.ttl = i.okb.ttl)
   ]
### EQUI:
DT[equi.idx, on = .(id, equi.bin),
   ':=' (equi.sum = i.equi.sum,
         equi.sum.active = i.equi.sum.active,
         equi.count.active = i.equi.count.active,
         equi.payment = i.equi.payment,
         equi.sum.overdue = i.equi.sum.overdue,
         equi.count.delay = i.equi.count.delay,
         equi.sum.debt = i.equi.sum.debt,
         equi.count.delay.full = i.equi.count.delay.full,
         equi.ttl = i.equi.ttl)
   ]
### KBRS:
DT[kbrs.idx, on = .(id, kbrs.bin),
   ':=' (kbrs.sum = i.kbrs.sum,
         kbrs.sum.active = i.kbrs.sum.active,
         kbrs.count.active = i.kbrs.count.active,
         kbrs.payment = i.kbrs.payment,
         kbrs.sum.overdue = i.kbrs.sum.overdue,
         kbrs.count.delay = i.kbrs.count.delay,
         kbrs.sum.debt = i.kbrs.sum.debt,
         kbrs.count.delay.full = i.kbrs.count.delay.full,
         kbrs.ttl = i.kbrs.ttl)
   ]
### ALL BKI:
DT[allbki.idx, on = .(id, allbki.bin),
   ':=' (allbki.sum = i.allbki.sum,
         allbki.sum.active = i.allbki.sum.active,
         allbki.count.active = i.allbki.count.active,
         allbki.payment = i.allbki.payment,
         allbki.sum.overdue = i.allbki.sum.overdue,
         allbki.count.delay = i.allbki.count.delay,
         allbki.sum.debt = i.allbki.sum.debt,
         allbki.count.delay.full = i.allbki.count.delay.full,
         allbki.ttl = i.allbki.ttl)
   ]
DT[nbki.ttl != 'allNA', .N]
DT[okb.ttl != 'allNA', .N]
DT[equi.ttl != 'allNA', .N]
DT[kbrs.ttl != 'allNA', .N]
DT[allbki.ttl != 'nbki(allNA)::okb(allNA)::equi(allNA)::kbrs(allNA)', .N]


## STEP 5: Delete unused fields
DT[, ':=' (nbki.bin = NULL,
           nbki.ttl.shift = NULL,
           okb.bin = NULL,
           okb.ttl.shift = NULL,
           equi.bin = NULL,
           equi.ttl.shift = NULL,
           kbrs.bin = NULL,
           kbrs.ttl.shift = NULL,
           allbki.bin = NULL,
           allbki.ttl.shift = NULL)]
rm(nbki.idx, okb.idx, equi.idx, kbrs.idx, allbki.idx)
# END: Fill 'allNA' isssues beetwen same issues --------------------------------
names(DT)
DT[1:30, .(id, date.request)]

# BEG: Delete empty request rows -----------------------------------------------
### delete rows with ALL NA's & without inquiry
DT[, .N, by = inquiry]
DT[inquiry == 0 & allbki.ttl == 'nbki(allNA)::okb(allNA)::equi(allNA)::kbrs(allNA)', .N]
DT <- DT[!(inquiry == 0 & allbki.ttl == 'nbki(allNA)::okb(allNA)::equi(allNA)::kbrs(allNA)'), ]
DT[, .N, by = inquiry]

# ### save new data set
# write.table(DT, file = 'DT_cache.csv',
#             sep = ';', na = 'NA', dec = '.',
#             row.names = FALSE, fileEncoding = 'UTF-8')
# END: Delete empty request rows -----------------------------------------------



# BEG: Calculate time differance -----------------------------------------------
### calculate request number by date.request
#DT <- DT[order(id, date.request)]
DT[, num.request := seq_len(.N), by = .(id)]
DT[id == DT[num.request == max(num.request), id],
   .(id, date.request, num.request)]

### calculate difference in days & hours between requests by id
#diff(c(as.POSIXct('2016-12-14 19:05:43'), as.POSIXct('2016-12-15 19:05:44')))
DT[, diffdays.prev := c(NA, diff(as.Date(date.request))),
   by = .(id)]
DT[, diffhour.prev := c(NA, `units<-`(diff(date.request), 'hours')),
   by = .(id)]
DT[, diffhour.prev.int := floor(diffhour.prev)]

DT[id == DT[num.request == max(num.request), id],
   .(id, date.request, num.request, diffdays.prev, diffhour.prev, diffhour.prev.int)]
# END: Calculate time differance -----------------------------------------------
DT[1:30, .(id, date.request)]


# BEG: Finding inquiry 100% from cache & BKI -----------------------------------
## STEP 1: All inquires for all issues mark like 'hz'
DT[, inq := 'hz']
DT[, .N, by = .(inq)]

## STEP 2: Find & mark inquiries 100% from BKI
DT[, inq := ifelse (allbki.ttl != 'nbki(allNA)::okb(allNA)::equi(allNA)::kbrs(allNA)'
                    & (is.na(diffhour.prev.int) | diffhour.prev.int >= 24),
                    'bki', 'hz')
   ]
DT[, .N, by = .(inq)]
DT[id %in% DT[, .N, by = id][order(-N)][c(3), id],
   .(id, date.request, diffhour.prev, inq, allbki.ttl)]

## STEP 3: Create temporary field [bin] - bins between requests from BKI
DT[inq == 'bki', bin := seq_len(.N), by = .(id)
   ][, bin := zoo::na.locf(bin, na.rm = FALSE), by = .(id)]
DT[id %in% DT[, .N, by = id][order(-N)][c(3), id],
   .(id, date.request, diffhour.prev, inq, bin, allbki.ttl)]

## STEP 4: Calculate cumulative difference in hours between requests
##         after last request from BKI by id
DT[, diffhour.last.bki.cum :=
     cumsum(ifelse(is.na(diffhour.prev) |  inq == 'bki', 0, diffhour.prev)),
   by = .(id, bin)]
DT[id %in% DT[, .N, by = id][order(-N)][c(3, 6), id],
   .(id, date.request, diffhour.prev, inq, bin, diffhour.last.bki.cum
     , allbki.ttl )]

## STEP 5: Find & mark inquiries 100% from cache
DT[, inq := ifelse(inq == 'hz' & !is.na(bin) & diffhour.last.bki.cum <24,
                   'cache', inq)]
DT[id %in% DT[, .N, by = id][order(-N)][c(3, 6), id],
   .(id, date.request, diffhour.prev, inq, bin, diffhour.last.bki.cum
     , allbki.ttl )]
DT[, .N, by = .(inq)]
names(DT)
#DT[, bin := NULL]
# END: Finding inquiry 100% from cache & BKI -----------------------------------



##### check
DT[id %in% DT[, .N, by = id][order(-N)][c(2), id],
   .(id, id.request, date.request, nbki.ttl,
     nbki.sum, nbki.sum.active, nbki.count.active, nbki.payment,
     nbki.sum.overdue, nbki.count.delay, nbki.sum.debt, nbki.count.delay.full)]
DT[id %in% DT[, .N, by = id][order(-N)][c(3), id],
   .(id, date.request, inq, allbki.ttl,
     allbki.sum, allbki.sum.active, allbki.count.active, allbki.payment,
     allbki.sum.overdue, allbki.count.delay, allbki.sum.debt, allbki.count.delay.full)]
DT[allbki.ttl !='nbki(allNA)::okb(allNA)::equi(allNA)::kbrs(allNA)', .N]
DT[, .N, by = inq]



# BEG: Calculate the changes in BKI info ---------------------------------------
## 1. Calculate day's count beetwen reply from bki
DT[inq == 'bki',
   diffdays.prev.bki := floor(c(NA, diff(as.Date(date.request)))),
   by = .(id)]

DT[id %in% DT[, .N, by = id][order(-N)][c(2), id],
   .(id, date.request, inq, diffdays.prev, diffdays.prev.bki)]


## 2. SHIFT the info from bki and MARK the changes
DT[inq == 'bki',
   c('nbki.sum.diff', 'nbki.sum.active.diff', 'nbki.count.active.diff',
     'nbki.payment.diff', 'nbki.sum.overdue.diff', 'nbki.count.delay.diff',
     'nbki.sum.debt.diff', 'nbki.count.delay.full.diff', 'nbki.ttl.diff',

     'okb.sum.diff', 'okb.sum.active.diff', 'okb.count.active.diff',
     'okb.payment.diff', 'okb.sum.overdue.diff', 'okb.count.delay.diff',
     'okb.sum.debt.diff', 'okb.count.delay.full.diff', 'okb.ttl.diff',

     'equi.sum.diff', 'equi.sum.active.diff', 'equi.count.active.diff',
     'equi.payment.diff', 'equi.sum.overdue.diff', 'equi.count.delay.diff',
     'equi.sum.debt.diff', 'equi.count.delay.full.diff', 'equi.ttl.diff',

     'kbrs.sum.diff', 'kbrs.sum.active.diff', 'kbrs.count.active.diff',
     'kbrs.payment.diff', 'kbrs.sum.overdue.diff', 'kbrs.count.delay.diff',
     'kbrs.sum.debt.diff', 'kbrs.count.delay.full.diff', 'kbrs.ttl.diff',

     'allbki.sum.diff', 'allbki.sum.active.diff', 'allbki.count.active.diff',
     'allbki.payment.diff', 'allbki.sum.overdue.diff', 'allbki.count.delay.diff',
     'allbki.sum.debt.diff', 'allbki.count.delay.full.diff', 'allbki.ttl.diff'
   )
   := lapply(.SD, function (x) {as.integer(!(ifelse(is.na(diffdays.prev.bki), NA,
                                                   ifelse(is.na(x), 0, x) ==
                                                     ifelse(is.na(shift(x)), 0, shift(x))
                                                   )
                                            )
                                          )}),
   by = .(id),
   .SDcols =
     c('nbki.sum', 'nbki.sum.active', 'nbki.count.active',
       'nbki.payment', 'nbki.sum.overdue', 'nbki.count.delay',
       'nbki.sum.debt', 'nbki.count.delay.full', 'nbki.ttl',

       'okb.sum', 'okb.sum.active', 'okb.count.active',
       'okb.payment', 'okb.sum.overdue', 'okb.count.delay',
       'okb.sum.debt', 'okb.count.delay.full', 'okb.ttl',

       'equi.sum', 'equi.sum.active', 'equi.count.active',
       'equi.payment', 'equi.sum.overdue', 'equi.count.delay',
       'equi.sum.debt', 'equi.count.delay.full', 'equi.ttl',

       'kbrs.sum', 'kbrs.sum.active', 'kbrs.count.active',
       'kbrs.payment', 'kbrs.sum.overdue', 'kbrs.count.delay',
       'kbrs.sum.debt', 'kbrs.count.delay.full', 'kbrs.ttl',

       'allbki.sum', 'allbki.sum.active', 'allbki.count.active',
       'allbki.payment', 'allbki.sum.overdue', 'allbki.count.delay',
       'allbki.sum.debt', 'allbki.count.delay.full', 'allbki.ttl'
     )
   ]
#### view
DT[id %in% DT[, .N, by = id][order(-N)][c(2), id],
   c('id', 'date.request', 'inq', 'diffdays.prev.bki',
     'nbki.sum','nbki.sum.diff',
     'nbki.sum.active', 'nbki.sum.active.diff',
     'nbki.count.active', 'nbki.count.active.diff',
     'nbki.payment', 'nbki.payment.diff',
     'nbki.sum.overdue', 'nbki.sum.overdue.diff',
     'nbki.count.delay', 'nbki.count.delay.diff',
     'nbki.sum.debt', 'nbki.sum.debt.diff',
     'nbki.count.delay.full', 'nbki.count.delay.full.diff',
     'nbki.ttl', 'nbki.ttl.diff'
   ), with = FALSE]


## 3. Calculate the TYPE of changes
DT[inq == 'bki',
   c('nbki.sum.diff.type', 'nbki.sum.active.diff.type', 'nbki.count.active.diff.type',
     'nbki.payment.diff.type', 'nbki.sum.overdue.diff.type', 'nbki.count.delay.diff.type',
     'nbki.sum.debt.diff.type', 'nbki.count.delay.full.diff.type',

     'okb.sum.diff.type', 'okb.sum.active.diff.type', 'okb.count.active.diff.type',
     'okb.payment.diff.type', 'okb.sum.overdue.diff.type', 'okb.count.delay.diff.type',
     'okb.sum.debt.diff.type', 'okb.count.delay.full.diff.type',

     'equi.sum.diff.type', 'equi.sum.active.diff.type', 'equi.count.active.diff.type',
     'equi.payment.diff.type', 'equi.sum.overdue.diff.type', 'equi.count.delay.diff.type',
     'equi.sum.debt.diff.type', 'equi.count.delay.full.diff.type',

     'kbrs.sum.diff.type', 'kbrs.sum.active.diff.type', 'kbrs.count.active.diff.type',
     'kbrs.payment.diff.type', 'kbrs.sum.overdue.diff.type', 'kbrs.count.delay.diff.type',
     'kbrs.sum.debt.diff.type', 'kbrs.count.delay.full.diff.type',

     'allbki.sum.diff.type', 'allbki.sum.active.diff.type', 'allbki.count.active.diff.type',
     'allbki.payment.diff.type', 'allbki.sum.overdue.diff.type', 'allbki.count.delay.diff.type',
     'allbki.sum.debt.diff.type', 'allbki.count.delay.full.diff.type'
   )
   := lapply(.SD, function (x) {as.character(
     ifelse(is.na(diffdays.prev.bki), NA,
            ifelse(ifelse(is.na(x), 0, x) > ifelse(is.na(shift(x)), 0, shift(x)),
                   'negative',
                   ifelse(ifelse(is.na(x), 0, x) < ifelse(is.na(shift(x)), 0, shift(x)),
                          'positive','neutral')
                   )
            )
     )}),
   by = .(id),
   .SDcols =
     c('nbki.sum', 'nbki.sum.active', 'nbki.count.active',
       'nbki.payment', 'nbki.sum.overdue', 'nbki.count.delay',
       'nbki.sum.debt', 'nbki.count.delay.full',

       'okb.sum', 'okb.sum.active', 'okb.count.active',
       'okb.payment', 'okb.sum.overdue', 'okb.count.delay',
       'okb.sum.debt', 'okb.count.delay.full',

       'equi.sum', 'equi.sum.active', 'equi.count.active',
       'equi.payment', 'equi.sum.overdue', 'equi.count.delay',
       'equi.sum.debt', 'equi.count.delay.full',

       'kbrs.sum', 'kbrs.sum.active', 'kbrs.count.active',
       'kbrs.payment', 'kbrs.sum.overdue', 'kbrs.count.delay',
       'kbrs.sum.debt', 'kbrs.count.delay.full',

       'allbki.sum', 'allbki.sum.active', 'allbki.count.active',
       'allbki.payment', 'allbki.sum.overdue', 'allbki.count.delay',
       'allbki.sum.debt', 'allbki.count.delay.full'
     )
   ]
#### view
DT[id %in% DT[, .N, by = id][order(-N)][c(3), id] & inq == 'bki',
   c('id', 'date.request', 'inq', 'diffdays.prev.bki',
     'nbki.sum', 'nbki.sum.diff', 'nbki.sum.diff.type',
     'nbki.sum.active', 'nbki.sum.active.diff', 'nbki.sum.active.diff.type',
     'nbki.count.active', 'nbki.count.active.diff', 'nbki.count.active.diff.type',
     'nbki.payment', 'nbki.payment.diff', 'nbki.payment.diff.type',
     'nbki.sum.overdue', 'nbki.sum.overdue.diff', 'nbki.sum.overdue.diff.type',
     'nbki.count.delay', 'nbki.count.delay.diff', 'nbki.count.delay.diff.type',
     'nbki.sum.debt', 'nbki.sum.debt.diff',  'nbki.sum.debt.diff.type',
     'nbki.count.delay.full', 'nbki.count.delay.full.diff', 'nbki.count.delay.full.diff.type',
     'nbki.ttl', 'nbki.ttl.diff'
   ), with = FALSE]
# END: Calculate the changes in BKI info ---------------------------------------

### SAVE NEW DATA SET
write.table(DT, file = 'DT_cache_02.csv',
             sep = ';', na = 'NA', dec = '.',
             row.names = FALSE, fileEncoding = 'UTF-8')

DT[inq == 'bki', .N, by = .(diffdays.prev.bki)][order(diffdays.prev.bki)]









##### BEG ATTANTION !!!! THIS BLOCK ONLY FOR MERGING
# BEG: Create one set ----------------------------------------------------------
DT.01 <- fread('~/RProjects/201706-UnderOptim/UO.CacheBKI/DT_cache_01.csv',
            encoding = 'UTF-8', na.strings = c('', 'NA', 'N/A')
)
(names(DT.01) <- gsub('_', '.', tolower(names(DT.01))))

DT.02 <- fread('~/RProjects/201706-UnderOptim/UO.CacheBKI/DT_cache_02.csv',
               encoding = 'UTF-8', na.strings = c('', 'NA', 'N/A')
)
(names(DT.02) <- gsub('_', '.', tolower(names(DT.02))))


### date
DT.01[, date.request:= as.POSIXct(date.request, format = '%d.%m.%Y %H:%M:%S')]
DT.01[, date.of := as.Date(gsub('01.01.1900', NA, gsub("[-/]", ".", date.of)),
                        '%d.%m.%Y')]
DT.02[, date.request:= as.POSIXct(date.request, format = '%d.%m.%Y %H:%M:%S')]
DT.02[, date.of := as.Date(gsub('01.01.1900', NA, gsub("[-/]", ".", date.of)),
                           '%d.%m.%Y')]

identical(
DT.01[, lapply(.SD, class)],
DT.02[, lapply(.SD, class)]
)

DT <- rbindlist(list(DT.01, DT.02))
### SAVE NEW DATA SET
write.table(DT, file = 'DT_cache_fin.csv',
            sep = ';', na = 'NA', dec = '.',
            row.names = FALSE, fileEncoding = 'UTF-8')

rm(DT.01, DT.02)
gc()
# END: Create one set --------------------------------------------------------
##### END ATTANTION !!!! THIS BLOCK ONLY FOR MERGING









# BEG: Plot the results --------------------------------------------------------
## 1. Create dataset about main params for plotting
i <- 'sum'
ii <- 'nbki'
BKI <- list()
for (i in c('sum', 'sum.active', 'count.active', 'payment', 'sum.overdue',
            'count.delay', 'sum.debt', 'count.delay.full')) {
  for (ii in c('nbki', 'okb', 'equi', 'kbrs', 'allbki')) {

  iii <- switch(ii, nbki = '1.NBKI', okb = '2.OKB', equi = '3.EQFAX', kbrs = '4.KBRS', allbki = 'ALL')
  print(paste0(iii, '.', i))
  ### calculate the data NEGATIVE
  assign(
    paste0(ii, '.', i, '.neg'),
    DT[inq == 'bki' & get(paste0(ii, '.', i, '.diff.type')) == 'negative',
       .(N.neg = .N), by = .(days = diffdays.prev.bki)
       ][DT[inq == 'bki',
            .(.N, diff = mean(get(paste0(ii, '.', i, '.diff')))*100),
            by = .(days = diffdays.prev.bki)],
         on = 'days'
         ][, diff.neg := ifelse(is.na(N.neg), 0, N.neg/N*100)]
  )

  ### calculate the data POSITIVE
  assign(
    paste0(ii, '.', i, '.pos'),
    DT[inq == 'bki' & get(paste0(ii, '.', i, '.diff.type')) == 'positive',
       .(N.pos = .N), by = .(days = diffdays.prev.bki)
       ][DT[inq == 'bki',
            .(.N, diff = mean(get(paste0(ii, '.', i, '.diff')))*100),
            by = .(days = diffdays.prev.bki)],
         on = 'days'
         ][, diff.pos := ifelse(is.na(N.pos), 0, N.pos/N*100)]
  )

  ### calculate the data NEUTRAL unchanged
  assign(
    paste0(ii, '.', i, '.neut'),
    DT[inq == 'bki' & get(paste0(ii, '.', i, '.diff.type')) == 'neutral',
       .(N.neut = .N), by = .(days = diffdays.prev.bki)
       ][DT[inq == 'bki',
            .(.N, diff = mean(get(paste0(ii, '.', i, '.diff')))*100),
            by = .(days = diffdays.prev.bki)],
         on = 'days'
         ][, diff.neut := ifelse(is.na(N.neut), 0, N.neut/N*100)]
  )


    ### transform and add into table
  BKI <- rbindlist(list(BKI,
    get(paste0(ii, '.', i, '.neg'))[, .(days, Number = N.neg, fraction.Of = diff.neg, Changes.Type = 'negative',
                           bki = iii, param = i)],
    get(paste0(ii, '.', i, '.pos'))[, .(days, Number = N.pos, fraction.Of = diff.pos, Changes.Type = 'positive',
                               bki = iii, param = i)],
    get(paste0(ii, '.', i, '.neut'))[, .(days, Number = N.neut, fraction.Of = diff.neut, Changes.Type = 'unchanged',
                               bki = iii, param = i)],
    get(paste0(ii, '.', i, '.neg'))[, .(days, Number = get(paste0(ii, '.', i, '.neg'))[, N.neg] +
                                                       get(paste0(ii, '.', i, '.pos'))[, N.pos],
                                        fraction.Of = diff, Changes.Type = 'any changes',
                                        bki = iii, param = i)]
    ))
  rm(list = paste0(ii, '.', i, '.neg'))
  rm(list = paste0(ii, '.', i, '.pos'))
  rm(list = paste0(ii, '.', i, '.neut'))

  }
}; rm(i, ii, iii)


## 2. Create dataset about TTL for plotting
allbki.ttl.neg <- # NEGATIVE
  DT[inq == 'bki' & (
    allbki.sum.diff.type == 'negative' |
      allbki.sum.active.diff.type == 'negative' |
      allbki.count.active.diff.type == 'negative' |
      allbki.payment.diff.type == 'negative' |
      allbki.sum.overdue.diff.type == 'negative' |
      allbki.count.delay.diff.type == 'negative' |
      allbki.sum.debt.diff.type == 'negative' |
      allbki.count.delay.full.diff.type == 'negative'),
    .(N.neg = .N), by = .(days = diffdays.prev.bki)
    ][DT[inq == 'bki',
         .(.N, diff = mean(allbki.ttl.diff)*100),
         by = .(days = diffdays.prev.bki)],
      on = 'days'
      ][, diff.neg := ifelse(is.na(N.neg), 0, N.neg/N*100)]

allbki.ttl.pos <- # POSITIVE
  DT[inq == 'bki' & !(
    allbki.sum.diff.type == 'negative' |
      allbki.sum.active.diff.type == 'negative' |
      allbki.count.active.diff.type == 'negative' |
      allbki.payment.diff.type == 'negative' |
      allbki.sum.overdue.diff.type == 'negative' |
      allbki.count.delay.diff.type == 'negative' |
      allbki.sum.debt.diff.type == 'negative' |
      allbki.count.delay.full.diff.type == 'negative'
    ) & (
    allbki.sum.diff.type == 'positive' |
      allbki.sum.active.diff.type == 'positive' |
      allbki.count.active.diff.type == 'positive' |
      allbki.payment.diff.type == 'positive' |
      allbki.sum.overdue.diff.type == 'positive' |
      allbki.count.delay.diff.type == 'positive' |
      allbki.sum.debt.diff.type == 'positive' |
      allbki.count.delay.full.diff.type == 'positive')
      ,
    .(N.pos = .N), by = .(days = diffdays.prev.bki)
    ][DT[inq == 'bki',
         .(.N, diff = mean(allbki.ttl.diff)*100),
         by = .(days = diffdays.prev.bki)],
      on = 'days'
      ][, diff.pos := ifelse(is.na(N.pos), 0, N.pos/N*100)]


allbki.ttl.any <- # ANY CHANGES
  DT[inq == 'bki',
  .(N.any = sum(allbki.ttl.diff), diff = mean(allbki.ttl.diff)*100),
  by = .(days = diffdays.prev.bki)
  ]

### and add TTL to table
BKI <-
  rbindlist(list(BKI,
                 allbki.ttl.neg[, .(days, Numder = N.neg, fraction.Of = diff.neg, Changes.Type = 'negative',
                                bki = 'ALL', param = 'ALL')],
                 allbki.ttl.pos[, .(days, Numder = N.pos, fraction.Of = diff.pos, Changes.Type = 'positive',
                                bki = 'ALL', param = 'ALL')],
                 allbki.ttl.any[, .(days, Number = N.any, fraction.Of = diff, Changes.Type = 'any changes',
                                bki = 'ALL', param = 'ALL')]
                 ))
BKI <- BKI[!is.na(days), ]
rm(allbki.ttl.neg, allbki.ttl.pos, allbki.ttl.any)

## 3. Create table about share of the days
share <-
  DT[inq == 'bki' & diffdays.prev.bki < 31,
     .(.N),
     by = .(days = diffdays.prev.bki)
     ][, Share := N/sum(N)*100][order(days),]


### SAVE TABLE FOR REPORT
write.table(BKI, file = 'BKI.csv',
            sep = ';', na = 'NA', dec = '.',
            row.names = FALSE, fileEncoding = 'UTF-8')
write.table(share, file = 'share.csv',
            sep = ';', na = 'NA', dec = '.',
            row.names = FALSE, fileEncoding = 'UTF-8')


## 4. Plot share of the days
ggplot(share,
       aes(x = days, y = Share)) +
  geom_point(aes(size = N), color = 'red') + geom_line(color = 'red4') +
  labs(title = 'Доля запросов в общем кол-ве запросов в интервале [1-30] дней') +
  scale_x_continuous(name = 'дней после последнего запроса в БКИ, #',
                     #trans = 'sqrt',
                     breaks =  c(seq(1, 30, 1), 40, 50, 60, 75, 91, 100)) +
  scale_y_continuous(name = 'Доля в общем кол-ве, %',
                     #trans = 'sqrt',
                     breaks =  c(seq(0, 20, 1), seq(25, 100, 5))) +
  theme(axis.text.x = element_text(size = 7),
        axis.title.x = element_text(size = 9, color = 'darkred', face = 'bold'),
        axis.text.y = element_text(size = 7),
        axis.title.y = element_text(size = 9, color = 'darkred', face = 'bold'),
        legend.position = c(0.9, 0.85),
        legend.box = 'horizontal',
        title = element_text(size = 9)
  )


d <- 91 # limit of days for plot
## 4. Plot TOTAL CHANGES
ggplot(BKI[bki == 'ALL' & param == 'ALL' &
             Changes.Type %in% c('negative', 'any changes'),
           .(days, Number, fraction.Of, Changes.Type, bki, param)
           ][days < d, ],
       aes(x = days, y = fraction.Of, color = Changes.Type)) +
  geom_point(aes(size = Number),  alpha = 0.5) +
  annotate(geom = 'text',
           x = BKI[bki == 'ALL' & param == 'ALL' & Changes.Type == 'negative' & days < 8, days],
           y = BKI[bki == 'ALL' & param == 'ALL' & Changes.Type == 'negative' & days < 8, fraction.Of - 1],
           label = BKI[bki == 'ALL' & param == 'ALL' & Changes.Type == 'negative' & days < 8,
                       round(fraction.Of, 1)],
           size = 2.5) +
  geom_smooth(data = BKI[bki == 'ALL' & param == 'ALL' & Changes.Type == 'negative',
                         .(days, Number, fraction.Of, Changes.Type, bki, param)][days < d, ],
              method = 'loess', se = FALSE, color = c('red4'), size = .5) +
  geom_smooth(data = BKI[bki == 'ALL' & param == 'ALL' & Changes.Type == 'any changes',
                         .(days, Number, fraction.Of, Changes.Type, bki, param)][days < d, ],
              method = 'loess', se = FALSE, color = c('green4'), size = .5) +
  labs(title = 'Доля изменившихся КИ от числа дней с последнего запроса',
       subtitle = '*изменения были хотя бы в одном из критичных параметров КИ') +
  scale_color_manual(values = c('lightgreen', 'red')) +
  scale_x_continuous(name = 'дней после последнего запроса в БКИ, #',
                     trans = 'sqrt',
                     breaks =  c(1, 2, 3, 5, 7, 10, 14, 21, 30, 40, 50, 60, 75, 91, 100)) +
  scale_y_continuous(name = 'изменилось КИ, %',
                     trans = 'sqrt',
                     breaks =  c(0, 1, 2, 3 ,seq(5, 100, 5))) +
  theme(axis.text.x = element_text(size = 7),
        axis.title.x = element_text(size = 9, color = 'darkred', face = 'bold'),
        axis.text.y = element_text(size = 7),
        axis.title.y = element_text(size = 9, color = 'darkred', face = 'bold'),
        legend.position = c(0.8, 0.15),
        legend.box = 'horizontal',
        title = element_text(size = 9)
  )


## 5. Plot OTHER main params
i <- 'sum'
for (i in c('sum', 'sum.active', 'count.active', 'payment', 'sum.overdue',
            'count.delay', 'sum.debt', 'count.delay.full')) {
    ii <- switch(i, sum = 'Сумма всех кредитов',
                 sum.active = 'Сумма активных кредитов',
                 count.active = 'Кол-во активных кредитов',
                 payment = 'Сумма ежемесячных платежей',
                 sum.overdue = 'Сумма текущей просрочки',
                 count.delay = 'Кол-во просрочек за последние 2 года',
                 sum.debt = 'Сумма текущих долгов',
                 count.delay.full = 'Кол-во просрочек за весь период')

print(
ggplot(BKI[param == i & Changes.Type %in% c('negative', 'any changes'),
           .(days, Number, fraction.Of, Changes.Type, bki, param)][days < d, ],
       aes(x = days, y = fraction.Of, color = Changes.Type)) +
  geom_point(aes(size = Number),  alpha = 0.5) +
  geom_smooth(data = BKI[param == i & Changes.Type == 'negative',
                         .(days, Number, fraction.Of, Changes.Type, bki, param)][days < d, ],
              method = 'loess', se = FALSE, color = c('red4'), size = .5) +
  geom_smooth(data = BKI[param == i & Changes.Type == 'any changes',
                         .(days, Number, fraction.Of, Changes.Type, bki, param)][days < d, ],
              method = 'loess', se = FALSE, color = c('green4'), size = .5) +
  facet_wrap(facets = ~ bki, ncol = 2) +
  labs(title = paste('Параметр:',ii),
       subtitle = 'Доля изменившихся КИ от числа дней с последнего запроса') +
  scale_color_manual(values = c('lightgreen', 'red')) +
  scale_x_continuous(name = 'дней после последнего запроса в БКИ, #',
                     trans = 'sqrt',
                     breaks =  c(1, 2, 3, 5, 7, 10, 14, 21, 30, 40, 50, 60, 75, 91, 100)) +
  scale_y_continuous(name = 'изменилось КИ, %',
                     trans = 'sqrt',
                     breaks =  c(0, 1, 2, 3 ,seq(5, 100, 5))) +
  theme(axis.text.x = element_text(size = 7),
        axis.title.x = element_text(size = 9, color = 'darkred', face = 'bold'),
        axis.text.y = element_text(size = 7),
        axis.title.y = element_text(size = 9, color = 'darkred', face = 'bold'),
        legend.position = c(0.8, 0.15),
        legend.box = 'horizontal',
        title = element_text(size = 9)
        )
)
}; rm(i, ii)
# END: Plot the results --------------------------------------------------------
########################## DONE LINE ###########################################
save.image()
time.end <- Sys.time()
time.start
time.end
time.end - time.start
gc()

names(DT)

DT[, .N, by = .(tip, decision)]
DT[, .N]
DT[, .N, by = .(inq)][order(-N)]
DT[, .N, by = .(inq, decision)]
DT[, max(date.request)]
DT[, min(date.request)]

DT[id %in% DT[, .N, by = id][order(-N)][c(2), id],
   .(id, id.request, date.request, inq, allbki.ttl)]


#-------------------------------------------------------------------------------
DT[diffdays.prev.bki == 1 &
     (
       (allbki.sum.diff == 1 & allbki.sum.active.diff == 0)
       ),
   .(id, date.request, id.request,
     allbki.sum, allbki.sum.diff, allbki.sum.diff.type,
     allbki.sum.active, allbki.sum.active.diff, allbki.sum.active.diff.type)]


DT[inq == 'bki' &
  id %in% DT[diffdays.prev.bki == 1 &
                (allbki.sum.diff.type == 'negative' &
                   allbki.sum.active.diff == 0), id
              ][1:5],
   .(id, id.request, date.request,
     allbki.sum, allbki.sum.diff, allbki.sum.diff.type,
     allbki.sum.active, allbki.sum.active.diff, allbki.sum.active.diff.type)
   ][order(id, date.request)]


DT[inq == 'bki' &
     id %in% DT[diffdays.prev.bki == 1 &
                  (allbki.sum.diff.type == 'positive' &
                     allbki.sum.active.diff == 0), id
                ][1:5],
   .(id, id.request, date.request,
     allbki.sum, allbki.sum.diff, allbki.sum.diff.type,
     allbki.sum.active, allbki.sum.active.diff, allbki.sum.active.diff.type)
   ][order(id, date.request)]


DT[, .N, by = .(diffdays.prev.bki)][order(diffdays.prev.bki)]




#-------------------------------------------------------------------------------
DT.date <- fread('~/RProjects/201706-UnderOptim/DS_CacheBKI_20171225_notNA_(20171201).csv',
            encoding = 'UTF-8', na.strings = c('', 'NA', 'N/A')
            , select = c('ID_REQUEST', 'DATE_REQUEST')
)
DT <- merge(DT, DT.date, by.x = 'id.request', by.y = 'ID_REQUEST', all.x = TRUE, sort = FALSE)
setnames(DT, 'DATE_REQUEST', 'date.request')
names(DT)
DT[1:30, .(id, date.request)]
#-------------------------------------------------------------------------------
