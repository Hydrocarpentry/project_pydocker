# project_pydocker
## Reproducing the Hydroshare research paper using Docker

Source: Sadler, J. (2018). Data-driven street flood severity modeling in Norfolk, Virginia USA 2010-2016, HydroShare, http://www.hydroshare.org/resource/9db60cf6c8394a0fa24777c8b9363a9b

-- Scripts and data files are found in the following links:
Hydroshare website and Github repository.
-- Following script files are needed for building the Docker image:
```prepare_flood_events_table.py; make_dly_obs_table_standalone.py; by_event_for_model.py; model_flood_counts_rf_ps_cln.r; plot_count_model_results.py; test.sh```
-- The following input data files are needed to be in a separate directory 'data' for building the Docker image:
```STORM_data_flooded_streets_2010-2016.csv; hampt_rd_data.sqlite```
  - Download the data "Raw street flood report data from Norfolk,VA 2010-2016" from the [GitHub repository](https://github.com/Hydrocarpentry/reproduced_data/blob/master/STORM_data_flooded_streets_2010-2016.csv). 
  - Download the data "Hamton Roads Enfironmental Time Series Data" from Hydroshare repository or from [OSF source](https://osf.io/mr7jx/?action=download)
-- The Docker image can be built using the above input files.
