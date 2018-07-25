
# coding: utf-8

# ## Intro
# This notebook aggregates the environmental data by event whereas before we were looking at the data by date. 

# ### Calculate number of locations that flooded

# In[1]:

#get_ipython().magic(u'matplotlib inline')
import os
import sys
#module_path = os.path.abspath(os.path.join('..'))
#if module_path not in sys.path:
#    sys.path.append(module_path)
#from focus_intersection import subset_floods, flood_df, subset_locations
#from main_db_script import data_dir, db_filename
#from hr_db_scripts.main_db_script import get_table_for_variable_code, get_db_table_as_df
import pandas as pd
import numpy as np
import re
import sqlite3
import math
#con = sqlite3.connect(db_filename)
pd.options.mode.chained_assignment = None  # default='warn'


# In this case we are just focusing on the subset of points that is in the downtown area thus the "subset_floods."

# In[2]:

flood_events = pd.read_csv('flood_events.csv')
flood_events['event_date'] = pd.to_datetime(flood_events['event_date'])
flood_events['event_name'] = flood_events['event_name'].str.strip()
flood_events['dates'] = pd.to_datetime(flood_events['dates'])


# In[3]:

grouped = flood_events.groupby(['event_date', 'event_name'])


# Get the number of dates the event spanned, the number of unique locations that were flooded during the event and the total number of locations flooded on all event dates. 

# In[4]:

event_total_flooded = grouped.size()
event_dates = grouped['dates'].unique()
num_event_dates = grouped['dates'].nunique()
num_locations = grouped['location'].nunique()


# In[5]:

event_df = pd.concat([event_dates, event_total_flooded, num_event_dates, num_locations], axis=1)
event_df.columns = ['dates', 'num_flooded', 'num_dates', 'num_locations']


# In[6]:

event_df.tail()


# ### Where num_flooded does not equal num_locations _investigation_
# Let's checkout one of the events where the num_flooded is greater than the num_locations. I would expect this to mean that one location was flooded on multiple days of the same event. But for '2014-07-24' the event is only on one day so that isn't what I expected.

# In[7]:

idx = pd.IndexSlice
event_df.sort_index(inplace=True)
event_df.loc[idx['2014-07-24', :], :]


# In[8]:

fl_724 = flood_events[flood_events['dates'] == '2014-07-24']
fl_724[fl_724['location'].duplicated(keep=False)]


# So _here's_ what is happening. The location name is the same in two rows but there are two different event types: "flooded street" and "flooded underpass."
# Now that I think about it, that may explain all the differences between the num_location and num_flooded columns. Let's try another one, this time one that spans more than one day: Irene.

# In[9]:

event_df.sort_index(inplace=True)
event_df.loc[idx[:, 'Irene-2011-08-27'], :]


# In[10]:

irene = flood_events[flood_events['event_name'].str.contains('Irene')].sort_values('location')
irene[irene['location'].duplicated(keep=False)]


# Looks like that's it. Which is not what I was hoping to show. I was thinking that that tell me something about the variety of locations that were flooded over the days but that's not the case.

# Let's try this one more time with Hurricane Joaquin

# In[11]:

jqn = flood_events[flood_events['event_name'].str.contains('Joaquin')]


# In[12]:

jqn[jqn['location'].duplicated(keep=False)]


# So that is interesting. Even though for hurricanes Matthew and Joaquin, the seven and six days respectively, none
# of the flooded locations were reported twice for one event. Very interesting. So to me, this means we really should be looking at these things by 'event' and not by '\_date'. It also means that the num_locations col doesn't add any information. So imma delete that.

# In[13]:

del event_df['num_locations']


# ### Looking into date in "event" column versus dates in "\_date" column
# Sometimes the date listed in the "event" column is quite different than the date(s) listed in the "\_date" column. A good example of this is the event "unnamed (2/25/2016)" where the dates in the "\_date" column are 2016-05-05, 2016-05-06, and 2016-05-31"

# In[14]:

flood_events[flood_events['event_date']=='2/25/2016']


# So to look at this more closely, I will calculate the difference in days between the "event" column date and the dates in the "\_date" column.

# When I tried to calculate the time between the 'event_date' and the 'dates' to see how far off these were I found that two events had the same 'event_date'. So I think it's appropriate to drop the 'unnamed' one based on the fact that the dates in the "\_date" column are further from the "event_date".

# In[15]:

event_df.sort_index(inplace=True)
event_df.loc['2016-07-30']


# In[16]:

i = event_df.loc['2016-07-30', 'unnamed-2016-07-30'].name
event_df.drop(i, inplace=True)
i = event_df.loc['2014-09-13', "NAPSG-2014-09-13"].name
event_df.drop(i, inplace=True)


# In[17]:

event_df.reset_index(inplace=True)
event_df.set_index('event_date', inplace=True)
event_df


# In[18]:

days_away = []
max_days = []
for d in event_df.index:
    try:
        ar = event_df.loc[d, 'dates'] - np.datetime64(d)
        ar = ar.astype('timedelta64[D]')
        days = ar / np.timedelta64(1, 'D')
        days_away.append(days)
        max_days.append(days.max())
    except ValueError:
        print(d)
event_df['days_away_from_event'] = days_away
event_df['max_days_away'] = max_days
print(event_df.shape)
event_df.head()


# I don't trust the events that have higher days away so I will disregard any event with a "max_days_away" greater than 10. Five events fall under this category.

# In[19]:

# event_df = event_df[event_df['max_days_away']<10]
print(event_df.shape)
event_df


# In[20]:

#feature_df = get_db_table_as_df('nor_daily_observations', dbfilename=db_filename)
feature_df = pd.read_csv('nor_daily_observations_standalone.csv')
feature_df['Datetime'] = pd.to_datetime(feature_df['Datetime'])
feature_df.set_index('Datetime', inplace = True)
feature_df.head()


# ### Combine env. data with event data

# In[21]:

def add_event_data(evnt_data, evnt_df, col_name, func, idx):
    res = func(evnt_data[col_name])
    evnt_df.loc[idx, col_name] = res
    return evnt_df


# Now for each event we get an aggregate of the different variables for the given dates

# In[22]:

event_df = pd.concat([event_df, pd.DataFrame(columns=feature_df.columns)])

for ind in event_df.index:
    # get the dates of the event and include the date in the "event" column
    ds = event_df.loc[ind, 'dates']
    ind = np.datetime64(ind)
    ds = np.append(ds, ind) if not ind in ds else ds
    
    event_data = feature_df.loc[ds]
 
    # combining data on event scale
    # get max over the event for these features
    max_cols = ['rhrmx', 'r15mx', 'wind_vel_daily_avg', 'wind_vel_hourly_max_avg', 'ht', 'hht', 'lt', 'llt']
    
    # get mean over the event for these features
    mean_cols = ['W', 'td', 'gw', 'AWDR', 'AWND']
    
    # get sum over the event for these features
    sum_cols = ['rd']
    
    for feat in feature_df.columns:
        if any(feat.startswith(col) for col in max_cols):
            event_df = add_event_data(event_data, event_df, feat, np.max, ind)
        elif any(feat.startswith(col) for col in mean_cols):
            event_df = add_event_data(event_data, event_df, feat, np.mean, ind)
        elif any(feat.startswith(col) for col in sum_cols):
            event_df = add_event_data(event_data, event_df, feat, np.sum, ind)
        elif feat.startswith('r3d'):
            event_df.loc[ind, feat] = event_data.loc[ind, feat]
        elif re.search(re.compile(r'r\w{2}-\d+_td-\d+'), feat):
            feat_spl = feat.split('-')
            var = '{}mx-{}'.format(feat_spl[0], feat_spl[1].split('_')[0])
            max_ind = event_data[var].idxmax()
            if isinstance(max_ind, float):
                if math.isnan(max_ind):
                    event_df.loc[ind, feat] = np.nan
            else:
                val = event_data.loc[max_ind, feat]
                event_df.loc[ind, feat] = event_data.loc[max_ind, feat]
        
event_df.head()


# In[23]:

event_df.shape


# In[24]:

event_df.head()


# In[25]:

cols = event_df.columns.tolist()
lft_cols = ['event_name', 'dates', 'num_flooded', 'days_away_from_event', 'max_days_away', 'num_dates']
lft_cols.reverse()
for c in lft_cols:
    cols.insert(0, cols.pop(cols.index(c)))
event_df = event_df.loc[:, cols]
event_df_for_storage = event_df.reset_index()
event_df_for_storage['dates'] = event_df_for_storage['dates'].apply(str)
event_df_for_storage['days_away_from_event'] = event_df_for_storage['days_away_from_event'].apply(str)
event_df_for_storage.rename(columns={'index':'event_date'}, inplace=True)
event_df_for_storage.head()


# In[26]:

event_df_for_storage.to_csv('{}event_data.csv'.format(''), index=False)
#event_df_for_storage.to_sql(name='event_data', con=con, if_exists='replace', index=False)


# ### Combining with the non-flooding event data
# First we have to combine all the dates in the "dates" column of the event_df into one array so we can filter those out of the overall dataset.

# In[27]:

flooded_dates = [np.datetime64(i) for i in event_df.index]
flooded_dates = np.array(flooded_dates)
fl_event_dates = np.concatenate(event_df['dates'].tolist())
all_fl_dates = np.concatenate([fl_event_dates, flooded_dates])


# In[28]:

non_flooded_records = feature_df[feature_df.index.isin(all_fl_dates) != True]
non_flooded_records['num_flooded'] = 0
non_flooded_records['flooded'] = False
non_flooded_records['event_name'] = np.nan
non_flooded_records['event_date'] = non_flooded_records.index
non_flooded_records.reset_index(drop=True, inplace=True)
non_flooded_records.head()


# Combine with flooded events

# In[29]:

event_df.reset_index(inplace=True)
flooded_records = event_df
flooded_records['event_date'] = event_df['index']
flooded_records['flooded'] = True
flooded_records.head()


# In[30]:

reformat = pd.concat([flooded_records, non_flooded_records], join='inner')
reformat.reset_index(inplace=True, drop=True)
reformat.head()


# In[31]:

#reformat.to_sql(name="for_model", con=con, index=False, if_exists='replace')


# ## Make average table

# In[32]:

cols = pd.Series(feature_df.columns)
cols_splt = cols.str.split('-', expand=True)
# do this to make sure the tide when the hourly and 15-min max rains have unique col names
for a in cols_splt.iterrows():
    if a[1].str.contains('\d_td').sum() == 1:
        cols_splt.loc[a[0], 0] += "_td"
col_vars = cols_splt[0].unique()
col_vars


# In[33]:

avdf = pd.DataFrame()
for v in col_vars:
    if v not in ['r15_td', 'rhr_td']:
        avdf[v] = reformat[[a for a in feature_df.columns if a.startswith(v)]].mean(axis=1)
    else:
        avdf[v] = reformat[cols[cols.str.contains(r'{}-\d+_td-\d+'.format(v.split('_')[0]))]].mean(axis=1)


# In[34]:

avdf = pd.concat([reformat[['event_date', 'event_name', 'num_flooded']], avdf], axis=1)


# In[35]:

avdf.head()


# In[36]:

avdf['ht'] = np.where(avdf['ht'].isnull(), avdf['hht'], avdf['ht'])
avdf['hht'] = np.where(avdf['hht'].isnull(), avdf['ht'], avdf['hht'])
avdf['lt'] = np.where(avdf['lt'].isnull(), avdf['llt'], avdf['lt'])
avdf['llt'] = np.where(avdf['llt'].isnull(), avdf['lt'], avdf['llt'])
avdf['WGF6'] = np.where(avdf['WGF6'].isnull(), avdf['AWND'], avdf['WGF6'])


# In[37]:

#avdf.to_sql(name='for_model_avgs', con=con, index=False, if_exists='replace')
avdf.to_csv('for_model_avgs.csv')


# In[38]:

avdf

