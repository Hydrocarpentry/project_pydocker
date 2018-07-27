FROM ubuntu:18.04
RUN apt-get update && apt-get install -y python-pandas 
#RUN apt-get install -y r-base
RUN apt-get install -y wget
RUN mkdir /data 
WORKDIR /data
RUN wget -O /data/hampt_rd_data.sqlite https://osf.io/mr7jx/?action=download  
RUN wget -O /data/STORM_data_flooded_streets_2010-2016.csv https://raw.githubusercontent.com/Hydrocarpentry/reproduced_data/master/STORM_data_flooded_streets_2010-2016.csv
#COPY hampt_rd_data.sqlite /data/
#COPY hampt_rd_data.sqlite STORM_data_flooded_streets_2010-2016.csv hampt_rd_data.sqlite /data/
#COPY hampt_rd_data.sqlite /data
#RUN echo "r <- getOption('repos'); r['CRAN'] <- 'http://cran.us.r-project.org'; options(repos = r);" > ~/.Rprofile
#RUN Rscript -e "install.packages('caret')"
#RUN Rscript -e "install.packages('randomForest')"
COPY prepare_flood_events_table.py make_dly_obs_table_standalone.py by_event_for_model.py model_flood_counts_rf_ps_cln.r plot_count_model_results.py test.sh /
RUN chmod +x /test.sh
CMD ["/test.sh"]
