# In[1]:


import pandas as pd
import numpy as np
import seaborn as sn
import pickle


# In[2]:


sub_df = pd.read_pickle('subscribers')
ad_df = pd.read_csv('advertising_spend_data.csv',index_col='date')


# In[3]:


sub_df['account_creation_month'] = sub_df.account_creation_date.apply(lambda x: x.strftime('%Y-%m'))


# In[4]:


channels = ['facebook','email','search','brand sem intent google','affiliate','email_blast','pinterest','referral']


# In[5]:


ad_df.index = ['2019-06','2019-07','2019-08','2019-09','2019-10','2019-11','2019-12','2020-01','2020-02','2020-03']


# In[6]:


ad_df


# ## atribution_survey

# In[13]:


survey_channel= set(channels).intersection(set(sub_df.attribution_survey.unique()))


# ### cac of each month

# In[14]:


def get_cac(month,channel):
    target_df = sub_df.loc[(sub_df.account_creation_month==month) & (sub_df.attribution_survey==channel)]
    target_number = len(target_df)
    target_cost = ad_df.loc[month,channel]
    cac = target_cost/target_number
    return cac


# In[15]:


survey_dict = {}
for channel in survey_channel:
    survey_dict.setdefault(channel,{})
    for month in ad_df.index:
        survey_dict[channel][month]=get_cac(month,channel)


# In[16]:


cac_df = pd.DataFrame(survey_dict)


# In[17]:


cac_df_2 =cac_df[1:]


# In[18]:


cac_df_2.to_csv('survey_cac.csv')


# In[19]:


cac_df_2.index


# ### marginal cac for each month

# In[7]:


last_month_dict = {'2019-07':'2019-06', '2019-08':'2019-07', '2019-09':'2019-08', '2019-10':'2019-09', 
                   '2019-11':'2019-10', '2019-12':'2019-11','2020-01':'2019-12', '2020-02':'2020-01', '2020-03':'2020-02'}


# In[8]:


def get_margin_cac(month,channel):
    target_df = sub_df.loc[(sub_df.account_creation_month==month) & (sub_df.attribution_survey==channel)]
    last_df = sub_df.loc[(sub_df.account_creation_month==last_month_dict[month]) & (sub_df.attribution_survey==channel)]
    target_number = len(target_df)
    last_number = len(last_df)
    target_cost = ad_df.loc[month,channel]
    last_cost = ad_df.loc[last_month_dict[month],channel]
    margin_cac = (target_cost-last_cost)/(target_number-last_number)
    return margin_cac


# In[9]:


survey_margin_dict = {}
for channel in survey_channel:
    survey_margin_dict.setdefault(channel,{})
    for month in cac_df_2.index:
        survey_margin_dict[channel][month]=get_margin_cac(month,channel)


# In[76]:


pd.DataFrame(survey_margin_dict).to_csv('survey_margin.csv')


# ## Attribution Tech

# In[20]:


tech_channel= set(channels).intersection(set(sub_df.attribution_technical.unique()))


# ### cac of each month

# In[21]:


def get_cac(month,channel):
    target_df = sub_df.loc[(sub_df.account_creation_month==month) & (sub_df.attribution_technical==channel)]
    target_number = len(target_df)
    target_cost = ad_df.loc[month,channel]
    cac = target_cost/target_number
    return cac


# In[40]:


target_df = sub_df.loc[(sub_df.account_creation_month=='2019-10') & (sub_df.attribution_technical=='facebook')]
target_number = len(target_df)


# In[41]:


target_number


# In[36]:


print(ad_df.loc['2019-10','facebook'])
ad_df.loc['2019-09','facebook']


# In[39]:


(51300-49000)/(9489-6126)


# In[26]:


target_df = sub_df.loc[(sub_df.account_creation_month=='2019-08') & (sub_df.attribution_technical=='facebook')]
len(target_df)


# In[22]:


survey_dict = {}
for channel in tech_channel:
    survey_dict.setdefault(channel,{})
    for month in cac_df_2.index:
        survey_dict[channel][month]=get_cac(month,channel)


# In[81]:


pd.DataFrame(survey_dict).to_csv('tech_cac.csv')


# ### margin cac of each month

# In[43]:


def get_margin_cac(month,channel):
    target_df = sub_df.loc[(sub_df.account_creation_month==month) & (sub_df.attribution_technical==channel)]
    last_df = sub_df.loc[(sub_df.account_creation_month==last_month_dict[month]) & (sub_df.attribution_technical==channel)]
    target_number = len(target_df)
    last_number = len(last_df)
    target_cost = ad_df.loc[month,channel]
    last_cost = ad_df.loc[last_month_dict[month],channel]
    margin_cac = (target_cost-last_cost)/(target_number-last_number)
    return margin_cac


# In[44]:


survey_margin_dict = {}
for channel in tech_channel:
    survey_margin_dict.setdefault(channel,{})
    for month in cac_df_2.index:
        survey_margin_dict[channel][month]=get_margin_cac(month,channel)


# In[46]:


pd.DataFrame(survey_margin_dict).to_csv('tech_margin.csv')

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sn
import pickle
from scipy import stats


# In[2]:


reps_df = pd.read_pickle('customer_service_reps')


# In[4]:


trial_df = reps_df.loc[reps_df.payment_period==0]


# In[5]:


trial_df['renew'] = [0 if str(record)=='NaT' else 1 for record in trial_df.next_payment]


# In[7]:


trial_df_2 = trial_df[['subid','num_trial_days','billing_channel','revenue_net_1month','renew']]


# In[8]:


trial_df_2['creation_month']=pd.DatetimeIndex(trial_df.account_creation_date).month


# In[9]:


subid_df = trial_df_2.groupby('subid').last()


# In[10]:


subid_df.renew.value_counts()


# In[11]:


subid_df.billing_channel.value_counts()


# ## OTT 0 vs 14

# In[12]:


OTT_df = subid_df.loc[subid_df.billing_channel=='OTT']


# In[13]:


pd.crosstab(OTT_df.creation_month,OTT_df.num_trial_days)


# In[14]:


OTT_df = OTT_df.loc[subid_df.creation_month!=6]


# In[15]:


#extract 0 days and 14 days
OTT_0 = OTT_df.loc[subid_df.num_trial_days==0]
OTT_14 = OTT_df.loc[subid_df.num_trial_days==14]


# In[16]:


# population p
p_14 = np.mean(OTT_14.renew)


# In[18]:


from scipy.stats import norm
from random import sample
import math


# In[19]:


#H0: miu1=miu2
#H1: miu1!=miu2
#alpha =0.05
alpha = 0.05
z_alpha =norm.ppf(1-alpha/2)


# In[20]:


p_0 = np.mean(OTT_0.renew)


# In[21]:


z_score = (p_0-p_14)/np.sqrt(p_0*(1-p_0)/len(OTT_0))


# In[22]:


z_score


# In[46]:


np.absolute(z_score)>np.absolute(z_alpha)


# ## itunes 7 vs 14

# In[23]:


itunes_df = subid_df.loc[subid_df.billing_channel=='itunes']


# In[24]:


pd.crosstab(itunes_df.creation_month,itunes_df.num_trial_days)


# In[25]:


itunes_df = itunes_df.loc[itunes_df.creation_month.isin([10,11,12])]


# In[26]:


itunes_7 = itunes_df.loc[itunes_df.num_trial_days==7]
itunes_14 = itunes_df.loc[itunes_df.num_trial_days==14]


# In[27]:


#homo-variance check·
stats.levene(itunes_7.renew,itunes_14.renew)


# In[28]:


# two-tail test
#H0: p1=p2
#H1: p1!=p2
p_14 = np.mean(itunes_14.renew)
p_7 = np.mean(itunes_7.renew)


# In[29]:


p = np.mean(itunes_df.renew)
q = 1-p


# In[30]:


#z_alpha
alpha = 0.05
z_alpha = norm.ppf(1-alpha/2)


# In[31]:


z_score = (p_7-p_14)/np.sqrt(p*q*(1/len(itunes_14)+1/len(itunes_7)))


# In[32]:


z_score


# In[83]:


## itunes 14 vs OTT 14
df_14 = subid_df.loc[subid_df.num_trial_days==14]


# In[86]:


df_ott = df_14.loc[df_14.billing_channel=='OTT']
df_itunes = df_14.loc[df_14.billing_channel=='itunes']


# In[87]:


## treat OTT 14 as population
p_ott = np.mean(df_ott.renew)
p_itunes = np.mean(df_itunes.renew)


# In[92]:


z_score =  (p_itunes-p_ott)/np.sqrt(p_itunes*(1-p_itunes)/len(df_itunes))


# In[93]:


z_score

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sn
import pickle
import scipy as sp


# In[2]:


sub_df = pd.read_pickle('subscribers')
reps_df = pd.read_pickle('customer_service_reps')
eng_df = pd.read_pickle('engagement')
ad_df = pd.read_csv('advertising_spend_data.csv')


# In[3]:


##get intersection set of sub_df and eng_df
sub_df_id = set(sub_df.subid)
eng_df_id = set(eng_df.subid)
cust_id = sub_df_id.intersection(eng_df_id)


# In[4]:


## deal with the dirty data
### age
new_age_0 = [(2019-record) if record>1000 and record<2000 else record for record in sub_df.age]


# In[5]:


sub_df['new_age']=new_age_0


# In[6]:


sn.distplot(sub_df.loc[sub_df.new_age<100].new_age.dropna())


# In[7]:


new_age_1 = [10 if record==0 else record for record in sub_df.new_age]


# In[8]:


sub_df['new_age']=new_age_1


# In[9]:


average_age = np.mean(sub_df.loc[sub_df.new_age<100].new_age)


# In[10]:


new_age_2 = [average_age if record>100 else record for record in sub_df.new_age]
sub_df['new_age']=new_age_2


# In[11]:


### gender
np.sum(sub_df.male_TF.isna())


# In[12]:


sub_df_2 = sub_df.loc[~sub_df.male_TF.isna()]


# In[13]:


cust_df = sub_df_2.loc[sub_df_2.subid.isin(cust_id)]


# In[14]:


###  prefered genre
np.sum(cust_df.preferred_genre.isna()) # imputate with another category


# In[15]:


cust_df.preferred_genre.value_counts()


# In[16]:


genre = ['no_pre' if type(record)==float else record for record in cust_df.preferred_genre]


# In[17]:


cust_df['new_genre']=genre


# In[18]:


cust_df.new_genre.value_counts()


# In[80]:


### intended use
np.sum(cust_df.intended_use.isna()) # imputate with others


# In[19]:


cust_df.intended_use.value_counts()


# In[20]:


new_intend =['other' if type(record)==float else record for record in cust_df.intended_use]


# In[21]:


cust_df['new_intend']=new_intend


# In[22]:


cust_df.new_intend.value_counts()


# In[23]:


### number weekly service utilized & number ideal streaming services
np.sum(cust_df.num_weekly_services_utilized.isna())


# In[24]:


cust_df['new_num_weekly_services'] = cust_df.num_weekly_services_utilized.fillna(0)


# In[25]:


### weekly consumption hour
np.sum(cust_df.weekly_consumption_hour.isna())


# In[26]:


from random import sample


# In[27]:


data = cust_df.weekly_consumption_hour.dropna()
sample(list(data),1)


# In[28]:


def resample_hour():
    #data = cust_df.weekly_consumption_hour.dropna()
    number = sample(list(data),1)
    number = number[0]
    return number


# In[33]:


new_hour = [resample_hour() if np.isnan(record)==True else record for record in cust_df.weekly_consumption_hour]


# In[34]:


new_consumption_hour = [25.851492  if record<0 else record for record in new_hour]


# In[35]:


cust_df['new_hour']=new_consumption_hour


# In[36]:


cust_df.columns


# In[37]:


cust_df.to_csv('cust_df.csv')


# In[38]:


cust_df_2 = cust_df[['subid','male_TF','new_age','new_genre','new_intend','new_num_weekly_services','new_hour']]


# In[39]:


## deal with eng_df
eng_df.columns


# In[40]:


eng_df_2 = eng_df[['subid', 'app_opens', 'cust_service_mssgs',
       'num_videos_completed', 'num_videos_more_than_30_seconds',
       'num_videos_rated', 'num_series_started']]


# In[41]:


for column in eng_df_2.columns:
    print(np.sum(eng_df_2[column].isna()))


# In[42]:


eng_df_3 = eng_df_2.fillna(0)


# In[44]:


eng_cust = eng_df_3.groupby('subid').sum()


# In[45]:


eng_cust.columns = ['app_open_total','mssgs_total','videos_completed_total','morethan30_total','rated_total','series_total']


# In[46]:


eng_cust_2 = eng_df_3.groupby('subid').mean()


# In[47]:


eng_cust_3 = pd.merge(eng_cust,eng_cust_2,on='subid')


# In[48]:


duration = eng_df_3.subid.value_counts()


# In[50]:


duration_dict = pd.DataFrame(duration).to_dict('index')


# In[51]:


eng_cust_3['duration'] = [duration_dict[record]['subid'] for record in eng_cust_3.index]


# In[52]:


cust_df_3 = pd.get_dummies(cust_df_2)


# In[53]:


cust_df_3['new_age']=cust_df_3.new_age.fillna(average_age)


# In[55]:


cust_df_4 = cust_df_3.drop(['male_TF_True','new_genre_no_pre','new_intend_other'],axis=1)


# In[56]:


cust_df_5 = cust_df_4.set_index('subid')


# In[59]:


cluster_df = pd.merge(cust_df_5,eng_cust_3,on='subid')


# In[60]:


cluster_df.shape


# In[61]:


cluster_df.to_csv('cluster.csv')


# In[62]:


# Standardize
from sklearn.preprocessing import StandardScaler,MinMaxScaler
X_scale =StandardScaler().fit_transform(cluster_df)


# In[63]:


### Cluster
from sklearn.cluster import KMeans


# In[64]:


#define Kmeans Cluster plot
def cluster_plot(data):
    ratios = []
    for i in range(1,10):
        cluster = KMeans(n_clusters=i,init='k-means++')
        cluster.fit(data)
        ratios.append(cluster.inertia_)
        
    sn.lineplot(range(1,10),ratios)


# In[65]:


cluster_plot(X_scale)


# In[67]:


first_label = KMeans(n_clusters=2,init='k-means++').fit(X_scale).predict(X_scale)


# In[68]:


cluster_df['first_label']=first_label


# In[69]:


# firstlabel 0: low frequency customer
# firstlabel 1: high frequency customer
## low frequency customer
low_df = cluster_df.loc[cluster_df.first_label==0]


# In[70]:


low_df_2 = low_df.drop(['first_label'],axis=1)


# In[71]:


X_low = MinMaxScaler().fit_transform(low_df_2)


# In[288]:


cluster_plot(X_low)


# In[72]:


low_df_2['2nd_label']=KMeans(n_clusters=4,init='k-means++').fit(X_low).predict(X_low)


# In[292]:


low_df_2.groupby('2nd_label').mean().T


# In[73]:


## high frequency customer
high_df = cluster_df.loc[cluster_df.first_label==1]
high_df_2 = high_df.drop(['first_label'],axis=1)
X_high = MinMaxScaler().fit_transform(high_df_2)


# In[286]:


cluster_plot(X_high)


# In[74]:


high_df['2nd_label'] = KMeans(n_clusters=4,init='k-means++').fit(X_high).predict(X_high)


# In[88]:


low_high_dict = {0:1,1:2,2:0,3:3}


# In[89]:


high_df['2nd_label']=[low_high_dict[record] for record in high_df['2nd_label']]


# In[90]:


high_df.groupby('2nd_label').mean().T


# In[91]:


clusters = pd.concat([low_df,high_df]) 


# In[92]:


reps_cust_df = reps_df.groupby('subid').last()


# In[94]:


reps_cust_dic = reps_cust_df.to_dict('index')


# In[95]:


clusters['period']=[reps_cust_dic[record]['payment_period'] for record in clusters.index]
clusters['revenue']=[reps_cust_dic[record]['revenue_net_1month'] for record in clusters.index]


# In[96]:


clusters.groupby('2nd_label').mean()



