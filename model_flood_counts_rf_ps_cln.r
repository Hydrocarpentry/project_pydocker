library(caret)
library(randomForest)

run_model = function(model_type, trn_data, trn_in_data, trn_out_data, tst_in_data, tst_out_data, fmla){
  if (model_type == 'poisson'){
    print('normalizing')
  	train_col_stds = apply(trn_in_data, 2, sd)
  	train_col_means = colMeans(trn_in_data)
  
  	train_normalized = t((t(trn_in_data)-train_col_means)/train_col_stds)
  	test_normalized = t((t(tst_in_data)-train_col_means)/train_col_stds)
  
  	pca = prcomp(train_normalized)
  
  	trn_preprocessed = predict(pca, train_normalized)
  	tst_preprocessed = predict(pca, test_normalized)
  
  	fmla = as.formula(paste(out_col_name, "~", paste(colnames(trn_preprocessed), collapse="+")))
  
  	train_data = cbind(as.data.frame(trn_preprocessed), num_flooded = model_data[prt$Resample1, out_col_name])
  	trn_in_data = trn_preprocessed
  	tst_in_data = tst_preprocessed
  	output = glm(fmla, data=train_data, family = poisson)
  }
  else if (model_type == 'rf'){
	output = randomForest(fmla, data=trn_data, importance = TRUE, ntree=100, mtry=16)
	impo = as.data.frame(output$importance)
	impo = impo[,1]
  }

  pred_trn = predict(output, newdata = as.data.frame(trn_in_data), type='response')
  pred_tst = predict(output, newdata = as.data.frame(tst_in_data), type='response')
  
  if (model_type == 'rf'){
       return(list(pred_trn, pred_tst, impo))
  }
  else {
       return(list(pred_trn, pred_tst))
  }
  
}

remove_cols= function(l, cols){
    return(l[! l %in% cols])
}

#clear_db_tables = function(models, con, suffix){
 # db_table_list = dbListTables(con)
  #for (model in models){
   # table_suffixes = c('train', 'test')
    #for (t in table_suffixes){
     # table = paste(model, '_', suffix, '_', t, sep="")
      #if (table %in% db_table_list){
       # dbGetQuery(con, paste("DROP TABLE", table))
     # }
   # }
  #}
#}


df = read.csv('for_model_avgs.csv', )

colnames(df)

set.seed(5)

df = df[df[,'rd']>0.01,]

cols_to_remove = c('event_name', 'event_date', 'num_flooded')
in_col_names = remove_cols(colnames(df), cols_to_remove)
out_col_name = 'num_flooded'

model_data = df[, append(in_col_names, out_col_name)]
model_data = na.omit(model_data)

import_df = data.frame(matrix(nrow=length(in_col_names)))
all_pred_tst = c()
all_pred_trn = c()
all_tst = c()
all_trn = c()
fomla = as.formula(paste(out_col_name, "~", paste(in_col_names, collapse="+")))
model_types = c('rf', 'poisson')
suffix = 'out'
#clear_db_tables(model_types, con, suffix)

for (i in 1:101){
  prt = createDataPartition(model_data[, out_col_name], p=0.7)
  train_data = model_data[prt$Resample1,]
  train_in_data = data.frame(train_data[, in_col_names])
  colnames(train_in_data) = in_col_names
  train_out_data = train_data[, out_col_name]
  test_in_data = data.frame(model_data[-prt$Resample1, in_col_names])
  colnames(test_in_data) = in_col_names
  test_out_data = model_data[-prt$Resample1, out_col_name]
  
  for (model in model_types){
	  print(paste("run: ", i, sep = ''))
    
    model_results = run_model(model, train_data, train_in_data, train_out_data, test_in_data, test_out_data, fomla)
	  pred_train = model_results[1]
	  pred_test = model_results[2]

	  all_trn_df = data.frame(train_out_data, unlist(pred_train))
	  colnames(all_trn_df) = c('all_trn', 'all_pred_trn')
	  all_tst_df = data.frame(test_out_data, unlist(pred_test))
	  colnames(all_tst_df) = c('all_tst', 'all_pred_tst')
	  write.table(all_trn_df, paste(model, '_', suffix, '_train.csv', sep=""), append=TRUE,  sep=",", col.names = F)
	  write.table(all_tst_df, paste(model, '_', suffix, '_test.csv', sep=""), append=TRUE,  sep=",", col.names = F)

	  if (model == 'rf'){
      impo = model_results[3]
	    import_df = cbind(import_df, impo)
	  }
	}
}

colnames(import_df) = 1:ncol(import_df)
row.names(import_df) = in_col_names
write.csv(import_df, paste('rf_impo_', suffix, sep=""), append=TRUE)
