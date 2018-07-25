install.packages("caret")
install.packages("randomForest")
install.packages("ggfortify")

library(caret)
library(ggfortify)
library(rpart)
library(rpart.plot)
library(RSQLite)
library(DBI)
library(randomForest)
library(e1071)


classify_knn = function(pca.data, obs_data, point, k){
  neighs = get_nn(pca.data, point, k)
  neigh_data = obs_data[neighs,]
  return(names(which.max(table(neigh_data$fld))))
}

get_nn = function(d, point, k){
  point = d[point, ]
  dists = sort(sqrt(rowSums((t(t(d) - point))^2)))
  close_points = names(dists)[2:k]
  return(close_points)
}

base_dir<- "C:/Users/Jeff/Documents/research/Sadler_3rdPaper/manuscript/"
data_dir<- paste(base_dir, "Data/", sep="")
fig_dir <- paste(base_dir, "Figures/general/", sep="")
db_filename <- "floodData.sqlite"
print(paste(data_dir, db_filename))
con = dbConnect(RSQLite::SQLite(), dbname=paste(data_dir, db_filename, sep=""))

dbListTables(con)

df = dbReadTable(con, 'for_model')
df = subset(df, rain_daily_sum>0.1) # only look at days that have over 0.1 in of rainfall
df = df[!is.na(df$gw_daily_avg),] # remove null gw values
short_names = c('rd', 'rhr', 'rhmxtime', 'r15', 'r15mxtime', 'r3d', 'gw', 'td', 'tr15mx', 'trhrmx', 'wdd', 'wvd',  'wvh', 'nfld', 'fld', 'evnme', 'evdte')
colnames(df) = short_names

in_col_names = c('rd', 'rhr', 'r15', 'r3d', 'gw', 'td', 'wvd', 'tr15mx', 'trhrmx')
out_col_name = 'fld'
data = df[, c(in_col_names, out_col_name)]
data$fld = df$fld>0

# pca
pca_data = data.frame(scale(data))
pca = prcomp(pca_data)
pca$x = -pca$x
pca$rotation=-pca$rotation
p = ggplot(pca$x[,c(1,2)], aes(x=PC1, y=PC2, colour=data[, out_col_name], label=rownames(pca$x)))
p + geom_point() +geom_text()
point = "19"
neighs = get_nn(pca$x, "1697", 11)
data[neighs,]

data$fld = factor(data$fld)
prt = createDataPartition(data$fld, p=0.7)
train_ind = prt$Resample1

# knn
for (row_name in row.names(pca$x)){
  data[row_name, 'pred_knn'] = classify_knn(pca$x, data, row_name, 5) 
}
kfit = knn(pca$x[train_ind, ], pca$x[-train_ind, ], data[train_ind, 'fld'], k=5)
table(data[-train_ind, 'fld'], kfit)

# svm
svm_fit = svm(x=pca$x[train_ind,], y=data[train_ind, 'fld'])
svm_pred = predict(svm_fit, pca$x[-train_ind,])
table(data[-train_ind, out_col_name], svm_pred)

# decision tree
fmla = as.formula(paste(out_col_name, "~", paste(in_col_names, collapse="+")))

fit = rpart(fmla, method='class', data=data[train_ind, ], minsplit=2)
printcp(fit)

tiff(paste(fig_dir, "Plot2.tif"), width=9, height=6, units='in', res = 300)
rpart.plot(fit, under=TRUE, cex=0.9, extra=1, varlen = 6)
dev.off()
rpart.plot(fit, under=TRUE, cex=0.9, extra=1, varlen = 6)

pfit<- prune(fit, cp=fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])
rpart.plot(pfit, under=TRUE, cex=0.9, extra=1, varlen = 6)

pred = predict(pfit, data[train_ind, in_col_names], type = 'class')
true_fld = data[train_ind, 'fld']
print('DT train')
table(data[train_ind, out_col_name], pred)

pred = predict(pfit, data[-train_ind, in_col_names], type = 'class')
true_fld = data[train_ind, 'fld']
print('DT test')
table(data[-train_ind, out_col_name], pred)

# creating predictive model
forest = randomForest(fmla, data = data[train_ind, ], importance = TRUE, type="classification", nodesize=2)

# check on training data
pred = predict(forest, data[train_ind, in_col_names])
true_fld = data[train_ind, 'fld']
print('RF train')
print(sum(pred == true_fld)/length(true_fld))
table(data[train_ind, out_col_name], pred)

# check on testing data
pred = predict(forest, data[-train_ind, in_col_names])
true_fld = data[-train_ind, 'fld']
print('RF test')
table(data[-train_ind, out_col_name], pred)
forest$importance

