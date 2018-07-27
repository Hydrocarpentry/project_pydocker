#!/bin/bash
#wget -O /data/STORM_data_flooded_streets_2010-2016.csv https://raw.githubusercontent.com/Hydrocarpentry/reproduced_data/master/STORM_data_flooded_streets_2010-2016.csv 
python /prepare_flood_events_table.py STORM_data_flooded_streets_2010-2016.csv flood_events.csv
#wget -O /data/hampt_rd_data.sqlite https://osf.io/mr7jx/?action=download 
python /make_dly_obs_table_standalone.py hampt_rd_data.sqlite
python /by_event_for_model.py flood_events.csv for_model_avgs.csv
#Rscript /model_flood_counts_rf_ps_cln.r  --input for_model_avgs.csv
#python /plot_count_model_results.py out 
