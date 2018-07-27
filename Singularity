Bootstrap: docker
From: ubuntu

%post 
    apt-get update
    apt-get install -y python-pandas
    apt-get install -y r-base
    apt-get install wget
    wget -O /hampt_rd_data.sqlite https://osf.io/mr7jx/?action=download  
    wget -O https://github.com/Hydrocarpentry/reproduced_data/blob/master/STORM_data_flooded_streets_2010-2016.csv    
    # Install required R packages
    R --slave -e 'install.packages("caret")'
    R --slave -e 'install.packages("randomForest")'

%files
    prepare_flood_events_table.py
    make_dly_obs_table_standalone.py
    by_event_for_model.py
    model_flood_counts_rf_ps_cln.r
    plot_count_model_results.py

%runscript
    python /prepare_flood_events_table.py 
    python /make_dly_obs_table_standalone.py
    python /by_event_for_model.py
    Rscript /model_flood_counts_rf_ps_cln.r
    python /plot_count_model_results.py out
