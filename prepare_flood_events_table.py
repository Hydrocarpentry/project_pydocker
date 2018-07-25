
# coding: utf-8

# # Prepare Flood Events data for DB

# In[1]:

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
# from db_scripts.main_db_script import data_dir, db_filename
import pandas as pd
import sqlite3


# ### Read in the data

# In[2]:

cds = pd.read_csv('STORM_data_flooded_streets_2010-2016.csv')


# ### Index by location name and subset to just columns we want

# In[3]:

cds = cds[['location', 'event', 'eventType', 'dt']]


# In[4]:

cds


# In[5]:

event_dates = cds.event.str.extract(r'(\d*/\d*/\d*)', expand=False)


# In[6]:

event_names = cds.event.str.replace(r'(\(\d*/\d*/\d*)\)', '')


# In[7]:

cds['event_name'] = event_names


# In[8]:

cds['event_date'] = pd.to_datetime(event_dates)
event_date_str = cds['event_date'].dt.strftime('%Y-%m-%d').str.replace('/', '-')


# In[9]:

cds['dt'] = pd.to_datetime(cds['dt'])


# In[10]:

cds['dates'] = cds['dt'].dt.strftime('%Y-%m-%d')


# In[11]:

cds['event_name'] = event_names.str.strip()+ '-' + event_date_str


# In[12]:

del cds['event']


# In[13]:

cds


# In[14]:

#con = sqlite3.connect(db_filename)
#cds.to_sql(con=con, name="flood_events", if_exists="replace")
cds.to_csv('flood_events.csv')


# In[15]:

cds.set_index('event_name')
