##Name the file as a Dockerfile
##when you build the dockerfile you should use other name
FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python-pandas
###by default it is python 2.7###
RUN apt-get install -y r-base
RUN echo "r <- getOption('repos'); r['CRAN'] <- 'http://cran.us.r-project.org'; options(repos = r);" > ~/.Rprofile
RUN Rscript -e "install.packages(‘caret’)”
RUN Rscript -e "install.packages(‘randomForest’)”


ADD STORM_data_flooded_streets_2010-2016.csv hampt_rd_data.sqlite prepare_flood_events_table.py make_dly_obs_table_standalone.py by_event_for_model.py model_flood_counts_rf_ps_cln.r plot_count_model_results.py test.sh /

####those scripts with data and test.sh contains bash file with input data files#

RUN chmod +x ./test.sh
CMD ["./test.sh"]

##usage docker build -t python_dockerfile .
### !!!!!!ATTENTION dot is important!!!!!! 
##docker run python-dockerfile
