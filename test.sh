#!/bin/bash
python /prepare_flood_events_table.py 
python /make_dly_obs_table_standalone.py
python /by_event_for_model.py
Rscript /install.r
Rscript /model_flood_counts_rf_ps_cln.r
python /plot_count_model_results.py out
