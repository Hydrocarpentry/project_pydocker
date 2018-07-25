
# coding: utf-8

# In[1]:

#get_ipython().magic(u'matplotlib inline')
import os
import sys
import pandas as pd
import numpy as np
import sqlite3
db_filepath = "hampt_rd_data.sqlite"
import matplotlib
matplotlib.use('Agg')

# In[2]:

def get_db_table_as_df(name, sql="""SELECT * FROM {};""", date_col=None, dbfilename=db_filepath):
    con = sqlite3.connect(dbfilename)
    sql = sql.format(name)
    if name == 'datavalues':
        date_col = 'Datetime'
    df = pd.read_sql(sql, con, parse_dates=date_col)
    if name == 'datavalues':
        df = make_date_index(df, 'Datetime')
    return df


# In[3]:

def get_id(typ, data):
    """
    gets either the siteid or variableid from the db
    :param typ: String. Either "Site" or "Variable"
    :param data: Dict. the site or variable data
    :return: int. id of site or variable
    """
    data_df = pd.DataFrame(data, index=[0])
    code_name = '{}Code'.format(typ)
    table_name = '{}s'.format(typ.lower())
    id_name = '{}ID'.format(typ)
    code = data[code_name]
    check_by = [code_name]
    table = get_db_table_as_df(table_name)
    id_row = table[table[code_name] == code]
    id_num = id_row[id_name].values[0]
    return id_num


# In[4]:

def make_date_index(df, field, fmt=None):
    df.loc[:, field] = pd.to_datetime(df.loc[:, field], format=fmt)
    df.set_index(field, drop=True, inplace=True)
    return df


# In[5]:

def get_table_for_variable_code(variable_code, site_id=None, start_date=None, end_date=None):
    var_id = get_id('Variable', {'VariableCode': variable_code})
    table_name = 'datavalues'
    sql = """SELECT * FROM {} WHERE VariableID={};""".format(table_name, var_id)
    if start_date or end_date:
        if not start_date:
            start_date = '1900-01-01'
        elif not end_date:
            end_date = '2100-01-01'
        sql = """SELECT * FROM {} WHERE VariableID={} AND Datetime BETWEEN '{}' AND '{}';""".format(
            table_name,
            var_id,
            start_date,
            end_date
        )

    df = get_db_table_as_df(table_name, sql=sql)
    df = df.sort_index()
    if site_id:
        df = df[df['SiteID'] == site_id]
    return df


# In[6]:

def round_down_near_24(datetimes): # round down the times near midnight so the tide levels stay on the correct day
    close_time_idx = datetimes.indexer_between_time('23:29', '23:59')
    adjusted_times = datetimes[close_time_idx] - pd.Timedelta(minutes=15)
    dt = pd.Series(datetimes)
    dt[close_time_idx] = adjusted_times
    dt = pd.DatetimeIndex(dt)
    return dt


# In[7]:

def cln_n_rnd_times(df):
    for i in range(df.shape[1]):
        datetimes = df.iloc[:, i]
        times = pd.DatetimeIndex(datetimes)
        rnd_dn = round_down_near_24(times)
        df.iloc[:, i] = rnd_dn
    return df


# In[8]:

def pivot_dv_df(df):
    return df.pivot(columns='SiteID', values='Value')


# In[9]:

def rename_cols(df, var_abbrev):
    if var_abbrev != "":
        new_df = df.copy()
        cols = df.columns.tolist()
        new_cols = ['{}-{}'.format(var_abbrev, c) for c in cols]
        new_df.columns = new_cols
        return new_df
    else:
        return df


# In[10]:

def filter_max_rain_time_dfs(rain_daily_df, time_df):
    timemx_filt = pd.DataFrame(np.where(rain_daily_df>0, time_df, np.datetime64('NaT')))
    timemx_filt.columns = time_df.columns
    timemx_filt.index = time_df.index
    return timemx_filt


# In[11]:

def tide_when_rain_max(rn_mx_time_df):
    td_df = get_table_for_variable_code('six_min_tide')
    try:
        td_df = pivot_dv_df(td_df)
    except:
        td_df = remove_duplicates(td_df)
        td_df = pivot_dv_df(td_df)
    td_df = td_df.resample('15T').mean()
    rn_mx_time_rnd = cln_n_rnd_times(rn_mx_time_df)
    l = []
    for c in rn_mx_time_rnd.columns:
        times = rn_mx_time_rnd.loc[:, c]
        tides = td_df.loc[times].resample('D').max()
        rain_var = c.split('_')[0]
        rain_site = c.split('-')[-1]
        new_cols = ['{}-{}_td-{}'.format(rain_var, rain_site, col) for col in tides.columns]
        tides.columns = new_cols
        l.append(tides)
    new_df = pd.concat(l, axis=1)
    new_df.sort_index(inplace=True)
    return new_df


# In[12]:

def remove_duplicates(df):
    siteids = df['SiteID'].unique()
    df.reset_index(inplace=True)
    print df.shape
    non_duplicated = list()
    for site in siteids:
        df_site = df[df['SiteID'] == site]
        df_site_vals = df_site['Datetime']
        df_no_dups = ~df_site_vals.duplicated()
        df_no_dups_idx = df_site[df_no_dups].index
        non_duplicated.extend(df_no_dups_idx.tolist())
    df = df.loc[non_duplicated]
    df.set_index('Datetime', drop=True, inplace=True)
    print df.shape
    return df


# In[13]:

def daily_pivot_table(var_code, agg_function, abbreviation):    
    df = get_table_for_variable_code(var_code)
    try:
        dfp = pivot_dv_df(df)
    except ValueError:
        df = remove_duplicates(df)
        dfp = pivot_dv_df(df)
    dfd = dfp.resample('D')
    aggrd = dfd.agg(agg_function)
    rnmed = rename_cols(aggrd, abbreviation)
    return rnmed


# #  Rainfall

# In[14]:

# get rainfall data at 15 min interval
rain_df = get_table_for_variable_code('rainfall')


# ## Daily Rainfall

# In[15]:

rain_daily15 = daily_pivot_table('rainfall', np.sum, '')
rain_daily = daily_pivot_table('daily_rainfall', np.sum, '')
rain_daily_comb_no_name = pd.concat([rain_daily, rain_daily15], axis=1)
rain_daily_comb_named = rename_cols(rain_daily_comb_no_name, 'rd')
rain_daily_comb_named.head()


# In[16]:

rain_daily_comb_named.tail()


# ## Hourly Rainfall

# In[17]:

rain15 = pivot_dv_df(rain_df)
rain_hourly_totals = rain15.rolling(window='H').sum()
rhr_mx = rain_hourly_totals.resample('D').max()
rhr_mx = rename_cols(rhr_mx, 'rhrmx')
rhr_mx.head()


# In[18]:

rhr_timemx = rain_hourly_totals.groupby(pd.TimeGrouper('D')).idxmax()
rhr_timemx = rename_cols(rhr_timemx, 'rhr_mxtime')
rhr_timemx = filter_max_rain_time_dfs(rain_daily15, rhr_timemx)
rhr_timemx.head()


# ## 15-min max rainfall

# In[19]:

r15_mx = rain15.resample('D').max()
r15_mx = rename_cols(r15_mx, 'r15mx')
r15_mx.head()


# In[20]:

r15_timemx = rain15.groupby(pd.TimeGrouper('D')).idxmax()
r15_timemx = rename_cols(r15_timemx, 'r15_mxtime')
r15_timemx = filter_max_rain_time_dfs(rain_daily15, r15_timemx)
r15_timemx.head()


# ### Rain prev 3 days

# In[21]:

rain_prev_3_days = rain_daily_comb_no_name.shift(1).rolling(window=3).sum()
rain_prev_3_days = rename_cols(rain_prev_3_days, 'r3d')
rain_prev_3_days.head()


# In[22]:

rain_daily_comb_named['rd-14'][rain_daily_comb_named['rd-14']<0]


# In[23]:

rain15.loc['2014-06-24']


# In[24]:

rain_prev_3_days.plot.box()


# #  Groundwater

# In[25]:

gw_df = daily_pivot_table('shallow_well_depth', np.mean, 'gw_av')
gw_df.head()


# #  Tide

# ## Average daily tide

# In[26]:

tide_df = daily_pivot_table('six_min_tide', np.mean, 'td_av')
tide_df.head()


# ##  Tide when rain is at max

# In[27]:

td_r15mx = tide_when_rain_max(r15_timemx)
td_r15mx.head()


# In[28]:

td_rhrmx = tide_when_rain_max(rhr_timemx)
td_rhrmx.head()


# ## HI/LOs

# In[29]:

hilos = []
for v in ['high_tide', 'high_high_tide', 'low_tide', 'low_low_tide']:
    hilos.append(daily_pivot_table(v, np.mean, "".join(w[0] for w in v.split('_'))))


# In[30]:

hilo_df = pd.concat(hilos, axis=1)
hilo_df.head()


# #  Wind

# In[31]:

wind_dfs = []
for v in ['WDF2', 'WSF2', 'AWDR', 'AWND', 'WGF6', 'WSF6', 'WDF6', 'WS2min', 'WD2min']:
    if v == 'WSF6':
        abbr = 'AWND'
    elif v == 'WDF6':
        abbr = 'AWDR'
    elif v == 'WS2min':
        abbr = 'AWND'
    elif v == 'WD2min':
        abbr = 'AWDR'
    else:
        abbr = v
    wind_dfs.append(daily_pivot_table(v, np.mean, abbr))
all_wind = pd.concat(wind_dfs, axis=1)
all_wind.head()


# In[32]:

feature_df = pd.concat([all_wind, hilo_df, td_r15mx, td_rhrmx, tide_df, gw_df, r15_mx, rhr_mx, rain_daily_comb_named, rain_prev_3_days], axis=1)
feature_df = feature_df.loc['2010-09-15':'2016-10-15']
feature_df.head()


# 
# ### Save Daily Observations to DB

# In[34]:

# con = sqlite3.connect(db_filename)
# feature_df.to_sql(con=con, name="nor_daily_observations", if_exists="replace")
feature_df.to_csv('nor_daily_observations_standalone.csv')


# In[ ]:



