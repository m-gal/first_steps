# This script does EDA

save.image()
# First deals
rm(list=ls())
gc()
opar <- par(no.readonly = TRUE)
options(scipen=9999)

# Load libraries
library(data.table)
library(Hmisc)
library(ggplot2)

# BEG: Load & view data --------------------------------------------------------
## 1. load data
if (grepl('C:/', getwd()) == 1) {
  DT <- fread('~/RProjects/201706-UnderOptim/DS_UnderOptim_20171009_(01092016).csv',
              encoding = 'UTF-8', na.strings = c('', 'NA', 'N/A'))
  DT.f <- fread('~/RProjects/201706-UnderOptim/DS_FieldDescript.csv')
} else {
  DT <- fread('T:/RProjects/201706-UnderOptim/DS_UnderOptim_20171009_(01092016).csv',
              encoding = 'UTF-8', na.strings = c('', 'NA', 'N/A'))
  DT.f <- fread('T:/RProjects/201706-UnderOptim/DS_FieldDescript.csv')
}

DT.f[, feature := gsub('_', '.', tolower(feature))]
(names(DT) <- gsub('_', '.', tolower(names(DT))))

## 2. view min-max request date
DT[, lapply(.SD, function(x){max(
  as.Date(gsub('01.01.1900', NA, gsub("[-/]", ".", x)), '%d.%m.%Y'))}),
  .SDcols = 'date.request']
DT[, lapply(.SD, function(x){min(
  as.Date(gsub('01.01.1900', NA, gsub("[-/]", ".", x)), '%d.%m.%Y'))}),
  .SDcols = 'date.request']

## 3. preview data
### view all types & count of classes
(tb.clss <- data.table(table(DT[, sapply(.SD, class)]))) # save types of classes
#str(DT, list.len = ncol(DT)) # view structure


## 4. describe all features
# for (i in 1:nrow(tb.clss)) {
#   print(rep(paste('#', tb.clss[i, V1], '#'), 4))
#   print(describe(DT[, .SD,
#                        .SDcols = DT[, sapply(.SD, class)] == tb.clss[i, V1]]))
# }; rm(i)
# END: Load & view data --------------------------------------------------------



# BEG: Change values in new-added columns ------------------------------------------
DT[, .N, keyby = bki.last.active.credit.type]
DT[, .N, by = bki.last.active.credit.ourbank]
DT[, .N, by = wave]
DT[, .N, by = top.up]
DT[, .N, keyby = dbr.name]

## type of last active credit
DT[bki.last.active.credit.type %in% c('Авто', 'Автокредит'),
   bki.last.active.credit.type := 'Авто']
DT[bki.last.active.credit.type %in% c('Другое', 'Не определен', 'Неизвестен'),
   bki.last.active.credit.type := '.Другой']
DT[bki.last.active.credit.type %in% c('Микрокредит'),
   bki.last.active.credit.type := 'Микро']
DT[bki.last.active.credit.type %in% c('Потребительский'),
   bki.last.active.credit.type := 'Потреб.']

## bank of last active credit
DT[is.na(bki.last.active.credit.ourbank),
   bki.last.active.credit.ourbank := 'false']
# END: Change values in new-added columns ------------------------------------------



# BEG: Delete empty rows & column & only value ---------------------------------
## 1. coerce '###...#' to NA
### get cols names that contains '###...#'
(x.grid <- colnames(DT)[sapply(DT, function(x){
  any(grepl('^#+$', x))
      })])

### view counts about all '###...#'
lapply(# for all elements of list (xx) apply the summary func
  lapply(DT, function(x) {
    # find ###...# and return to list (x)
    grep('^#+$', x, value = TRUE)
    # get index where ### exist (y)
  })[x.grid]
  , function(xx) {
    summary(as.factor(xx), maxsum = 10)
  })
summary(DT[, lapply(.SD, as.factor), .SDcols = (x.grid)])

### replace '###...#' to NA
DT[, (x.grid) := lapply(.SD, function(x) {
  gsub ('^#+$', NA, x)
}), .SDcols = (x.grid)]
summary(DT[, lapply(.SD, as.factor), .SDcols = (x.grid)])


## 2.1 find id.request with all NA's
DT[rowSums(is.na(DT[, !c('id.request', 'dim.issued', 'dim.preapproved',
                         'dim.segment', 'decision'),
                    with = FALSE])) == (ncol(DT) - 5),
   id.request]


## 2.2 find cols with all NA's
(x.NA <- colnames(DT)[sapply(DT, function(x) {
  all(is.na(x))
})])
describe(DT[, sapply(.SD, as.factor), .SDcols = x.NA])


## 3. find cols with all only 1 unique value
(x.only <- colnames(DT)[sapply(DT, function(x) {
  length(unique(x)) == 1 & sum(is.na(x)) == 0
})])
describe(DT[, sapply(.SD, as.factor), .SDcols = x.only])
DT[, c(x.NA, x.only) := NULL] # delete
dim(DT)

### save classes & count
(tb.clss <- merge(tb.clss, data.table(table(DT[, sapply(.SD, class)])),
                  by = 'V1', all = TRUE, suffixes = c(".all", ".del")))
# END: Delete empty rows & column & only value ---------------------------------



# BEG: Assing fields classes ---------------------------------------------------
## 1. find & format cols with date values
### view fields with 'date' in name
str(DT[, grep('date', names(DT), value = TRUE), with = FALSE])
summary(DT[, sapply(.SD, as.factor),
              .SDcols =  grep('date', names(DT), value = TRUE)])
length(grep('date', names(DT), value = TRUE))
grep('date', names(DT), value = TRUE)

### get field names that really have date value
(x.date <- colnames(DT)[sapply(DT, function(x) {
  !all(is.na(grep("^\\d{2,4}[-./]\\d{2}[-./]\\d{2,4}", x)))
})])

x.date[x.date %in% grep('date', names(DT), value = TRUE)]
x.date[!(x.date %in% grep('date', names(DT), value = TRUE))]
grep('date', names(DT), value = TRUE)[!(grep('date', names(DT), value = TRUE) %in% x.date)]

DT[, sapply(.SD, class), .SDcols = (x.date)] # view class & summary
summary(DT[, sapply(.SD, as.factor), .SDcols = (x.date)])
describe(DT[, sapply(.SD, as.factor), .SDcols = (x.date)])

### replace [-/] & coerce to "Date" class
DT[, (x.date) := lapply(.SD, function(x) {
  as.Date(gsub('01.01.1900', NA, gsub("[-/]", ".", x)), '%d.%m.%Y')
}), .SDcols = (x.date)]
DT[, sapply(.SD, class), .SDcols = (x.date)] # view class & summary
summary(DT[, sapply(.SD, as.factor), .SDcols = (x.date)])
describe(DT[, .SD, .SDcols = (x.date)])

### save current classes & count
(tb.clss <- merge(tb.clss, data.table(table(DT[, sapply(.SD, class)])),
                  by = 'V1', all = TRUE))
names(tb.clss)[4] <- 'N.date'
tb.clss


## 2. find & format cols with binomial (0\1)
### get col names
(x.bin <- colnames(DT)[sapply(DT, function(x) {
  all(grepl('^0{1}[.,]?0*$|^1{1}[.,]?0*$', na.omit(x)))
})])
table(DT[, sapply(.SD, class), .SDcols = (x.bin)])
summary(DT[, sapply(.SD, as.factor), .SDcols = (x.bin)])

### coerce to "integer" class
DT[, (x.bin) := lapply(.SD, function(x) {
  as.integer(x)
}), .SDcols = (x.bin)]
DT[, sapply(.SD, class), .SDcols = (x.bin)] # view class & summary
table(DT[, sapply(.SD, class), .SDcols = (x.bin)])
summary(DT[, sapply(.SD, as.factor), .SDcols = (x.bin)])

### save current classes & count
(tb.clss <- merge(tb.clss, data.table(table(DT[, sapply(.SD, class)])),
                  by = 'V1', all = TRUE))
names(tb.clss)[5] <- 'N.bin'
tb.clss


## 3. find & format cols with numeric
### get col names
(x.num <- names(DT[, -(x.bin), with = FALSE])[DT[, sapply(.SD, function(x) {
  all(grepl('^-?[0-9]*[.,]?\\d+$', na.omit(x)))
}), .SDcols = -(x.bin)]])
table(DT[, sapply(.SD, class), .SDcols = (x.num)])
summary(DT[, sapply(.SD, function(x) {
  as.numeric(sub(',', '.', x))
}), .SDcols = (x.num)])
#describe(DT[, sapply(.SD, as.numeric), .SDcols = (x.num[1:100])])
#describe(DT[, sapply(.SD, as.numeric), .SDcols = (x.num[100:length(x.num)])])

### replace ',' to '.' in x.num and coerce to "numeric" class
DT[, (x.num) := lapply(.SD, function(x) {
  as.numeric(sub(',', '.', x))
}), .SDcols = (x.num)]
table(DT[, sapply(.SD, class), .SDcols = (x.num)])
#describe(DT[, .SD, .SDcols = x.num[1:50]])
#describe(DT[, .SD, .SDcols = x.num[51:100]])
#describe(DT[, .SD, .SDcols = x.num[101:length(x.num)]])

### save current classes & count
(tb.clss <- merge(tb.clss, data.table(table(DT[, sapply(.SD, class)])),
                  by = 'V1', all = TRUE))
names(tb.clss)[6] <- 'N.num'
tb.clss


## 4. find character cols
describe(DT[, .SD, .SDcols = DT[, sapply(.SD, class)] == 'character'])
(x.char <- names(DT[, .SD,
                       .SDcols = DT[, sapply(.SD, class)] == 'character']))
tb.clss
# END: Assing fields classes ---------------------------------------------------



# BEG: Add, clean & delete some features ----------------------------------------------
## 1. Clean some fetures
### Age
DT[, .N, by = (age < 18)]
# hist(DT[age < 18, age])
DT[age < 18, age := NA]

### Segment
DT [, .N, by = .(dim.segment)]
DT[dim.segment %in% c('Сотрудники группы БИН',
                      'Сотрудники МДМ Банка',
                      'Сотрудники Банка'),
   dim.segment := 'Сотрудники']
DT[dim.segment %in% c('Зарплатные клиенты',
                      'Зарплатные клиенты БИН'),
   dim.segment := 'ЗП клиенты']
DT[dim.segment == 'Корпоративный канал', dim.segment := 'Корп. канал']
DT[dim.segment == 'Сотрудники бюджетных организаций', dim.segment := 'Бюджетники']
DT [, .N, by = .(dim.segment)]


## 2. Add new periodical features
DT.f[feature %in% x.date, .(feature, descript)]

### create names for 'periodical' var
(x.days <- sub('date.|day.', 'days.', x.date)[-(1:2)])
(x.days <- x.days[!(x.days %in% c('bki.last.creditopendate',
                                 'bki.last.active.creditopendate'))])

### create new features
DT[, ':=' (days.last.del.31.days = date.request - date.last.del.31.days,
              days.last.del.less.31.days = date.request - date.last.del.less.31.days,
              days.last.del.31.93.d.2 = date.request - date.last.del.31.93.d.2,
              days.last.del.more.93.d.2 = date.request - date.last.del.more.93.d.2,
              spouse.days.last.del.less.31 = date.request - spouse.date.last.del.less.31,
              spouse.days.last.del.31.93 = date.request - spouse.date.last.del.31.93,
              spouse.days.last.del.more.93 = date.request - spouse.date.last.del.more.93,
              bscor.days.since.any = date.request - bscor.day.since.any,
              packets.days.last.salary = date.request - packets.date.last.salary,
              days.of.last.offset.of.debt = date.request - date.of.last.offset.of.debt,
              days.last.del.less.6.days = date.request - date.last.del.less.6.days,
              days.last.del.6.30.days = date.request - date.last.del.6.30.days,
              packets.days.last.salary.new = date.request - packets.date.last.salary.new
)]
### view classes & summary of news
DT[, lapply(.SD, class), .SDcols = (x.days)]
### coerce difftime to "numeric" class
DT[, (x.days) := lapply(.SD, function(x) {
  as.numeric(x, units = 'days')
}), .SDcols = (x.days)]
DT[, lapply(.SD, class), .SDcols = (x.days)]
summary(DT[, .SD, .SDcols = (x.days)])
### save current classes & count
(tb.clss <- merge(tb.clss, data.table(table(DT[, sapply(.SD, class)])),
                  by = 'V1', all = TRUE))
names(tb.clss)[7] <- 'N.+days'
tb.clss


## 3. Delete certanly unimportant features
describe(DT[, .SD, .SDcols = DT[, sapply(.SD, class)] %in% c('character')])

### create names for deleting
(x.del <- c(x.date, DT.f[group == 'id' & feature != 'id.request', feature],
            'fio', 'was.ovd30.4mob', 'was.ovd60.6mob', 'was.ovd90.12mob',
            #'requested.loan.amount',
            'cross.sale'))
DT[, c(x.del) := NULL]

### save current classes & count
(tb.clss <- merge(tb.clss, data.table(table(DT[, sapply(.SD, class)])),
                  by = 'V1', all = TRUE))
names(tb.clss)[8] <- 'N.-del'
tb.clss


### Collect info to list
x.list <- list(x.grid, x.NA, x.only, x.date, x.bin, x.num, x.char, x.days, x.del)
names(x.list) <- c('x.grid', 'x.NA', 'x.only', 'x.date', 'x.bin', 'x.num',
                   'x.char', 'x.days', 'x.del')
rm(x.grid, x.NA, x.only, x.date, x.bin, x.num, x.char, x.days, x.del)

### save datasets and info
#### convert list to df
x.list <- data.frame( # to data.frame
  do.call(cbind, # bind the columns
          lapply(x.list, # applay to all elements of list
                 function(x) {
                   x <- c(x, # add into X
                          rep(NA, # the NA's at count = maxLength - lengthX
                              max(sapply(x.list, length)) - length(x)
                          ))
          }
          )))
fwrite(x.list, 'xlist.csv', sep = ';')
fwrite(tb.clss, 'tb_clss.csv', sep = ';')

if (grepl('C:/', getwd()) == 1) {
  fwrite(DT, file = '~/RProjects/201706-UnderOptim/UO_DT_cleaned.csv',
       sep = ';', dec = '.')
} else {
  fwrite(DT, file = 'T:/RProjects/201706-UnderOptim/UO_DT_cleaned.csv',
       sep = ';', dec = '.')
}

if (grepl('C:/', getwd()) == 1) {
  write.table(DT, '~/RProjects/201706-UnderOptim/UO_DF_cleaned.csv',
            sep = ';', na = 'NA', dec = '.', row.names = FALSE,
            fileEncoding = 'UTF-8')
} else {
  write.table(DT, 'T:/RProjects/201706-UnderOptim/UO_DF_cleaned.csv',
            sep = ';', na = 'NA', dec = '.', row.names = FALSE,
            fileEncoding = 'UTF-8')
}

#### check correctly
if (grepl('C:/', getwd()) == 1) {
  dt <- fread('~/RProjects/201706-UnderOptim/UO_DT_cleaned.csv',
            encoding = 'UTF-8')
  df <- fread('~/RProjects/201706-UnderOptim/UO_DF_cleaned.csv',
            encoding = 'UTF-8')
} else {
  dt <- fread('T:/RProjects/201706-UnderOptim/UO_DT_cleaned.csv',
              encoding = 'UTF-8')
  df <- fread('T:/RProjects/201706-UnderOptim/UO_DF_cleaned.csv',
              encoding = 'UTF-8')
}
dt[, .N, by = (dim.segment)]
df[, .N, by = (dim.segment)]
rm(dt, df)

# END: Add delete some date features -------------------------------------------
save.image()





# BEG: Get ISSUED dataset ------------------------------------------------------
## 1. Delete redundant features and lines
DT[, .N, keyby=.(dim.issued, dim.segment)]
DT.iss <- copy(DT[dim.issued == 1 & !(dim.segment == 'Сотрудники')])
DT.iss <- DT.iss[!(decision == 'Отказ')]
DT.iss[, c('id.request', 'dim.issued',
       'decision', 'otkaz.detail', 'otkaz.who') := NULL]

## 2. Find cols with all NA's
(x.NA <- colnames(DT.iss)[sapply(DT.iss, function(x) {all(is.na(x))})])
Hmisc::describe(DT.iss[, sapply(.SD, as.factor), .SDcols = x.NA])

## 3. Find cols with all only 1 unique value
(x.only <- colnames(DT.iss)[sapply(DT.iss, function(x) {
  length(unique(x)) == 1 & sum(is.na(x)) == 0
})])
Hmisc::describe(DT.iss[, sapply(.SD, as.factor), .SDcols = x.only])
DT.iss[, c(x.NA, x.only) := NULL] # delete
dim(DT.iss)

## 4. Check missing (NA) values in fields
x.char <- colnames(DT.iss)[sapply(DT.iss,is.character)]
cat('% NA in All =', sum(is.na(DT.iss))/(dim(DT.iss)[1]*dim(DT.iss)[2])*100, '\n',
    'All values=', dim(DT.iss)[1]*dim(DT.iss)[2])
cat('% NA in char =', sum(is.na(DT.iss[, .SD, .SDcols = (x.char)]))/
      (dim(DT.iss[, .SD, .SDcols = (x.char)])[1]*
         dim(DT.iss[, .SD, .SDcols = (x.char)])[2])*100, '\n',
    ' NA in char =', sum(is.na(DT.iss[, .SD, .SDcols = (x.char)])))
cat('% NA in NOchar =', sum(is.na(DT.iss[, .SD, .SDcols = -(x.char)]))/
      (dim(DT.iss[, .SD, .SDcols = -(x.char)])[1]*
         dim(DT.iss[, .SD, .SDcols = -(x.char)])[2])*100, '\n',
    ' NA in NOchar =', sum(is.na(DT.iss[, .SD, .SDcols = -(x.char)])))

## 5. Set all missing value in factors as "Missing"
for (j in seq_along(DT.iss)) {
  set(DT.iss, i = which(is.na(DT.iss[[j]]) & is.character(DT.iss[[j]])),
      j = j,
      value = 'Missing'
  )
}; rm(j)
sum(DT.iss[, lapply(.SD, function(x){sum(x=='Missing')}), .SDcols = (x.char)])

--DT.iss[, sapply(.SD, function(x){sum(x == 'Missing')/length(x)}), .SDcols = (x.char)]
--summary(as.factor(DT.iss[, birth.region]))


## 6. Coerce chars to factors
table(DT.iss[, sapply(.SD, class)])
#DT.iss[, (x.char) := lapply(.SD, as.factor), .SDcols = (x.char)]
table(DT.iss[, sapply(.SD, class)])
DT.iss[, mean(was.ovd90.12m)]*100

write.table(DT.iss, file = 'DT_issued.csv', sep = ';', na = 'NA', dec = '.',
            row.names = FALSE, fileEncoding = 'UTF-8')


rm(x.char, x.NA, x.only)
save.image()
# END: Get ISSUED dataset ------------------------------------------------------



# BEG: Get ISSUED & REFUSED dataset ------------------------------------------------------
## 1. Delete redundant features and lines
DT[, .N, keyby=.(dim.issued, dim.segment)]
DT.all <- copy(DT[!(dim.segment == 'Сотрудники')])
DT.all[, .N,  by = .(decision, dim.issued)]
DT.all[dim.issued == 1 & decision == 'Отказ', .(id.request)]
DT.all <- DT.all[!(decision == 'Отказ' & dim.issued == 1)]
DT.all[, c('id.request', 'decision') := NULL]
dim(DT.all)

## 2. Find cols with all NA's
(x.NA <- colnames(DT.all)[sapply(DT.all, function(x) {all(is.na(x))})])
Hmisc::describe(DT.all[, sapply(.SD, as.factor), .SDcols = x.NA])

## 3. Find cols with all only 1 unique value
(x.only <- colnames(DT.all)[sapply(DT.all, function(x) {
  length(unique(x)) == 1 & sum(is.na(x)) == 0
})])
Hmisc::describe(DT.all[, sapply(.SD, as.factor), .SDcols = x.only])
DT.all[, c(x.NA, x.only) := NULL] # delete
dim(DT.all)

## 4. Check missing (NA) values in fields
x.char <- colnames(DT.all)[sapply(DT.all,is.character)]
cat('% NA in All =', sum(is.na(DT.all))/(dim(DT.all)[1]*dim(DT.all)[2])*100, '\n',
    'All values=', dim(DT.all)[1]*dim(DT.all)[2])
cat('% NA in char =', sum(is.na(DT.all[, .SD, .SDcols = (x.char)]))/
      (dim(DT.all[, .SD, .SDcols = (x.char)])[1]*
         dim(DT.all[, .SD, .SDcols = (x.char)])[2])*100, '\n',
    ' NA in char =', sum(is.na(DT.all[, .SD, .SDcols = (x.char)])))
cat('% NA in NOchar =', sum(is.na(DT.all[, .SD, .SDcols = -(x.char)]))/
      (dim(DT.all[, .SD, .SDcols = -(x.char)])[1]*
         dim(DT.all[, .SD, .SDcols = -(x.char)])[2])*100, '\n',
    ' NA in NOchar =', sum(is.na(DT.all[, .SD, .SDcols = -(x.char)])))

## 5. Set all missing value in factors as "Missing"
for (j in seq_along(DT.all)) {
  set(DT.all, i = which(is.na(DT.all[[j]]) & is.character(DT.all[[j]])),
      j = j,
      value = 'Missing'
  )
}; rm(j)
sum(DT.all[, lapply(.SD, function(x){sum(x=='Missing')}), .SDcols = (x.char)])

--DT.all[, sapply(.SD, function(x){sum(x == 'Missing')/length(x)}), .SDcols = (x.char)]
--summary(as.factor(DT.all[, birth.region]))


## 6. Coerce chars to factors
table(DT.all[, sapply(.SD, class)])
#DT.all[, (x.char) := lapply(.SD, as.factor), .SDcols = (x.char)]
table(DT.all[, sapply(.SD, class)])
DT.all[was.ovd90.12m == 1, .N]/DT.all[, .N]*100

write.table(DT.all, file = 'DT_all.csv', sep = ';', na = 'NA', dec = '.',
            row.names = FALSE, fileEncoding = 'UTF-8')

rm(x.char, x.NA, x.only)
save.image()
# END: Get ISSUED & REFUSED dataset ------------------------------------------------------


####################  DONE BEFORE THIS  ########################################


# BEG: Some explore features ---------------------------------------------------
DT[, .N, by = dim.preapproved]
DT[, .N, by = dim.segment]
DT[, .N, by = dim.issued]
DT[, .N, by = (was.ovd90.12m)]
DT[, .N, keyby = matrix.id]
summary(as.factor(DT[, disposable.income]))
DT[, .N, keyby =.(dim.issued, dim.segment)]


## some plots
hist(DT[, matrix.id])
ggplot(data = DT[dim.issued == 1], aes(as.factor(was.ovd90.12m), matrix.id)) +
  facet_grid (dim.segment ~ dim.preapproved) +
  geom_boxplot()
ggplot(data = DT[dim.issued == 1], aes(as.factor(was.ovd90.12m), credit.sum)) +
  facet_grid (dim.segment ~ dim.preapproved) +
  geom_boxplot()
bpplot(DT[, credit.sum])
# END: Some explore features ---------------------------------------------------








