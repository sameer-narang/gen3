import _pickle as cPickle
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn

from datetime import timedelta
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

#NA_FILL_VAL = 1e-9
NA_FILL_VAL = None

DATE_FMT = "%m/%d/%Y"
suffix = "_ratio"
suffix2 = "_ratio_annual"

def summarize (y_act, y_pred, info):
        print (info + '. MSE: ' + str (np.mean ((y_act-y_pred)**2)))
        print (info + '. MAD: ' + str (np.mean (np.fabs(y_act-y_pred))))

def eval (mdl2):
        if False and os.path.isfile ('rf.mdl'):
            with open ('rf.mdl', 'rb') as f:
                mdl2 = cPickle.load (f)
        else:
            mdl2.fit (X_train, y_train)
            with open ('rf.mdl', 'wb') as f:
                cPickle.dump (mdl2, f)

        train_preds2 = mdl2.predict (X_train)
        val_preds2 = mdl2.predict (X_val)
        test_preds2 = mdl2.predict (X_test)

        summarize (y_train, train_preds2, 'Training')
        summarize (y_val, val_preds2, 'Validation')
        summarize (y_test, test_preds2, 'Test')
        return mdl2

def benchmark (mdl, features, df):
        bm_df = pd.read_csv ('benchmark_inputs.csv')
        bm_df ['Date'] = bm_df ['Date'].apply (datetime.datetime.strptime, args=(DATE_FMT,))
        bm_df ['TargetPeriodDate'] = bm_df ['TargetPeriodDate'].apply (datetime.datetime.strptime, args=(DATE_FMT,))
        bm_df = bm_df [(bm_df ['Date'] <= bm_df ['TargetPeriodDate'])]
        bm_df = pd.merge (bm_df, df, on='Date')
        y_preds = mdl.predict (bm_df [features])
        y_acts = bm_df ['gdp_label']
        ny_fed = bm_df ['ny_fed']

        summarize (y_preds, y_acts, 'Benchmarking - Our model vs Actual')
        summarize (ny_fed, y_acts, 'Benchmarking - NY Fed vs Actual')

        x = range (bm_df.shape [0])
        #x = bm_df ['Date']
        fig = plt.figure ()
        ax1 = fig.add_subplot (111, xlabel='Projection Dates', ylabel='Annualized QoQ GDP growth', title='Model comparison with NY Fed Reserve Nowcasts')
        ax1.scatter (x, y_acts, s=40, marker='s', label='Official')
        ax1.scatter (x, y_preds, s=40, marker='o', label='Our Model')
        ax1.scatter (x, ny_fed, s=40, marker='v', label='NY Fed')
        for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] +
                ax1.get_xticklabels() + ax1.get_yticklabels()):
             item.set_fontsize(20)
        plt.legend(loc='lower right');
        ##plt.show()
        #benchmark_df ['Value'] = np.asfarray (benchmark_df ['Value'])
        


df = pd.read_csv ('data2_11.csv')
df ['Date'] = df ['Date'].apply (datetime.datetime.strptime, args=(DATE_FMT,))
df ['prev_qtr'] = df ['prev_qtr'].apply (datetime.datetime.strptime, args=(DATE_FMT,))
df ['next_qtr'] = df ['next_qtr'].apply (datetime.datetime.strptime, args=(DATE_FMT,))
df ['prev_qtr_start'] = df ['prev_qtr_start'].apply (datetime.datetime.strptime, args=(DATE_FMT,))
#df = df [df ['Seq'] > 10000].reset_index ()

'''
y_train = df [(df ['Seq'] >= 10958) & (df ['Seq'] < 22646)] ['gdp_label']
y_val = df [(df ['Seq'] >= 22646) & (df ['Seq'] < 24106) ] ['gdp_label']
y_test = df [(df ['Seq'] >= 24106) & (df ['Seq'] < 24838) ] ['gdp_label']
'''
y_train = df [(df ['Seq'] >= 10958) & (df ['Seq'] < 22645)] ['gdp_label']
y_val = df [(df ['Seq'] >= 22645) & (df ['Seq'] < 24106) ] ['gdp_label']
y_test = df [(df ['Seq'] >= 24106) & (df ['Seq'] < 24838) ] ['gdp_label']


ts = ['GTII10 Govt','DXY Curncy','VIX Index','SPX Index','DOW US Equity','NDX Index', \
        'AUDUSD Curncy', 'USDCAD Curncy' ,'USDJPY Curncy', 'GBPUSD Curncy', 'USDMXN Curncy','USDCNY Curncy', \
        'USGG10YR Index', 'FDTR Index', 'MPMIUSMA Index', 'NAPMPMI Index', 'NAPMPRIC Index', \
        'CNSTTMOM Index', 'NAPMNEWO Index', 'NAPMEMPL Index', 'SAARTOTL Index', 'SAARDTOT Index', \
        'MBAVCHNG Index', 'ADP CHNG Index', 'MPMIUSSA Index', 'MPMIUSCA Index', 'NAPMNMI Index', \
        'CHALYOY% Index', 'INJCJC Index', 'INJCSP Index', 'USTBTOT Index', 'COMFCOMF Index', \
        'TMNOCHNG Index', 'TMNOXTM% Index', 'DGNOCHNG Index', 'DGNOXTCH Index', 'CGNOXAI% Index', \
        'CGSHXAI% Index', 'NFP TCH Index', 'NFP PCH Index', 'USMMMNCH Index', 'USURTOT Index', \
        'AHE MOM% Index', 'AHE YOY% Index', 'AWH TOTL Index', 'PRUSTOT Index', 'USUDMAER Index', \
        'MWINCHNG Index', 'MWSLCHNG Index', 'CICRTOT Index', 'SBOITOTL Index', 'JOLTTOTL Index', \
        'FDIDFDMO Index', 'FDIDSGMO Index', 'FDIDSGUM Index', 'FDIUFDYO Index', 'FDIUSGYO Index', \
        'FDIUSGUY Index', 'CPI CHNG Index', 'CPUPXCHG Index', 'CPI YOY Index', 'CPI XYOY Index', \
        'CPUPAXFE Index', 'CPURNSA Index', 'REALYRAE Index', 'REALYRAW Index', 'RSTAMOM Index', \
        'RSTAXMOM Index', 'RSTAXAG% Index', 'RSTAXAGM Index', 'CONSSENT Index', 'CONSCURR Index', \
        'CONSEXP Index', 'CONSPXMD Index', 'CONSP5MD Index', 'MTIBCHNG Index', 'EMPRGBCI Index', \
        'IMP1CHNG Index', 'IMP1XPM% Index', 'IMP1YOY% Index', 'EXP1CMOM Index', 'EXP1CYOY Index', \
        'IP  CHNG Index', 'CPTICHNG Index', 'IPMGCHNG Index', 'USHBMIDX Index', 'FRNTTNET Index', \
        'NHSPSTOT Index', 'NHCHSTCH Index', 'NHSPATOT Index', 'NHCHATCH Index', 'OUTFGAF Index', \
        'COMFBTWR Index', 'LEI CHNG Index', 'ETSLTOTL Index', 'ETSLMOM Index', 'FDDSSD Index', \
        'CFNAI Index', 'RCHSINDX Index', 'HPIMMOM% Index', 'NHSLTOT Index', 'NHSLCHNG Index', \
        'RSRSTMOM Index', 'USPHTMOM Index', 'USPHTYOY Index', 'KCLSSACI Index', 'GDPCTOT% Index', \
        'GDP PIQQ Index', 'GDPCPCEC Index', 'PITLCHNG Index', 'PCE CRCH Index', 'PCE CHNC Index', \
        'PCE DEFM Index', 'PCE DEFY Index', 'PCE CMOM Index', 'PCE CYOY Index', 'DFEDGBA Index', \
        'ECI SA% Index', 'SPCS20SM Index', 'SPCS20Y% Index', 'CHPMINDX Index', 'CONCCONF Index']
'''

ts = ['SPX Index', 'DXY Curncy', 'DOW US Equity', 'AUDUSD Curncy', 'USDCAD Curncy', \
             'USDJPY Curncy', 'GBPUSD Curncy', 'USDMXN Curncy', 'USDCNY Curncy', 'USGG10YR Index', \
             'FDTR Index', 'NAPMPMI Index', 'NAPMPRIC Index', 'NAPMNEWO Index', 'NAPMEMPL Index', \
             'SAARTOTL Index', 'INJCJC Index', 'INJCSP Index', 'TMNOCHNG Index', 'DGNOXTCH Index', \
             'CGNOXAI% Index', 'NFP TCH Index', 'USMMMNCH Index', 'NFP PCH Index', 'NFP TCH Index', \
             'PRUSTOT Index', 'CICRTOT Index', 'SBOITOTL Index', 'CPI CHNG Index', 'CPUPXCHG Index', \
             'CPI YOY Index', 'CPI XYOY Index', 'CPURNSA Index', 'CONSSENT Index', 'CONSCURR Index', \
             'CONSEXP Index', 'CONSPXMD Index', 'MTIBCHNG Index', 'IMP1YOY% Index', 'IP  CHNG Index', \
             'CPTICHNG Index', 'IPMGCHNG Index', 'FRNTTNET Index', 'NHSPSTOT Index', 'NHCHSTCH Index', \
             'NHSPATOT Index', 'NHCHATCH Index', 'OUTFGAF Index', 'LEI CHNG Index', 'CFNAI Index', \
             'NHSLTOT Index', 'GDPCTOT% Index', 'GDP PIQQ Index', 'GDPCPCEC Index', 'PITLCHNG Index', \
             'PCE DEFM Index', 'PCE DEFY Index', 'PCE CMOM Index', 'PCE CYOY Index', 'CHPMINDX Index', \
             'CONCCONF Index'
     ]



ts = [
                'GTII10 Govt'
                ]

'''
features = [
                        
                        'EHGDUS Index',
                        'days_to_go',
                        '1_nf', '4_nf', '5_nf'
                        ]

f_cols = {}

for s in ts:
        #features.append (s)
        df [s + suffix] = np.nan
        #features.append (s + suffix)
        f_cols [s + suffix] = []
        df [s + suffix2] = np.nan
        #features.append (s + suffix2)
        f_cols [s + suffix2] = []

for i in range (1,123):
    #features.append (str (i) + '_xlratio')
    #features.append (str (i) + '_xlratio2')
    None

for s in ts:
        continue
        f_cols ['Date'] = []
        if os.path.isfile (s + suffix + '.csv'):
                tmp_df = pd.read_csv (s + suffix + '.csv')
                df [s + suffix] = tmp_df [s + suffix]
        else:
                #import pdb; pdb.set_trace ()
                for i, row in df.iterrows():
                        running_avg_ratio = np.nan
                        if "%" in s:
                                running_avg_ratio = np.nanmean (df [(df ['Date'] > row ['prev_qtr']) & (df ['Date'] <= row ['next_qtr'])] [s]) - \
                                                np.nanmean (df [(df ['Date'] >= row ['prev_qtr_start']) & (df ['Date'] <= row ['Date'])] [s])
                        else:
                                running_avg_ratio = np.nanmean (df [(df ['Date'] > row ['prev_qtr']) & (df ['Date'] <= row ['Date'])] [s]) * 1000/ \
                                                np.nanmean (df [(df ['Date'] >= row ['prev_qtr_start']) & (df ['Date'] <= row ['prev_qtr'])] [s])
                        if np.isfinite (running_avg_ratio):
                                #df.set_value(i, s + suffix, running_avg_ratio)
                                f_cols [s + suffix].append (running_avg_ratio)
                        else:
                                f_cols [s + suffix].append (np.nan)
                        f_cols ['Date'].append (row ['Date'])
                #pd.DataFrame (df [[s + suffix], 'Date']).to_csv (s + suffix + '.csv', index=False)
                pd.DataFrame.from_dict ({'Date': f_cols ['Date'], s + suffix: f_cols [s + suffix]}).to_csv (s + suffix + '.csv', index=False)
                df [s + suffix] = f_cols [s + suffix]

        f_cols ['Date'] = []
        if os.path.isfile (s + suffix2 + '.csv'):
                tmp_df = pd.read_csv (s + suffix2 + '.csv')
                df [s + suffix2] = tmp_df [s + suffix2]
        else:
                #import pdb; pdb.set_trace ()
                for i, row in df.iterrows():
                        running_avg_ratio = np.nan
                        y1 = row ['Date'] + relativedelta (years=-1)
                        y2 = row ['Date'] + relativedelta (years=-2)
                        if "%" in s:
                                running_avg_ratio = np.nanmean (df [(df ['Date'] >= y1) & (df ['Date'] < row ['Date'])] [s]) - \
                                                np.nanmean (df [(df ['Date'] >= y2) & (df ['Date'] < y1)] [s])
                        else:
                                running_avg_ratio = np.nanmean (df [(df ['Date'] >= y1) & (df ['Date'] < row ['Date'])] [s]) * 1000/ \
                                                np.nanmean (df [(df ['Date'] >= y2) & (df ['Date'] < y1)] [s])
                        if np.isfinite (running_avg_ratio):
                                #df.set_value(i, s + suffix, running_avg_ratio)
                                f_cols [s + suffix2].append (running_avg_ratio)
                        else:
                                f_cols [s + suffix2].append (np.nan)
                        f_cols ['Date'].append (row ['Date'])
                #pd.DataFrame (df [[s + suffix], 'Date']).to_csv (s + suffix + '.csv', index=False)
                pd.DataFrame.from_dict ({'Date': f_cols ['Date'], s + suffix2: f_cols [s + suffix2]}).to_csv (s + suffix2 + '.csv', index=False)
                df [s + suffix2] = f_cols [s + suffix2]

#s = 'GTII10 Govt_ratio'
#df = df.drop (['index'], axis=1)
#df [s] = f_cols [s]
#f_df = pd.DataFrame.from_dict (f_cols)
#f_df.to_csv ('data_new.csv', index=False)

#df = pd.concat ([df, f_df], axis=1)
#df = df.merge (f_df, left_on='Date', right_on='Date', how='inner')
#df.to_csv ('data_all.csv', index=False)

#import pdb; pdb.set_trace ()
if NA_FILL_VAL:
        df = df.fillna (NA_FILL_VAL)
else:
        df = df.fillna (df.mean ())

#df = df.drop (['Date', 'next_qtr', 'prev_qtr'], axis=1)

X_train = df [(df ['Seq'] >= 10958) & (df ['Seq'] < 22645)] [features]
X_val = df [(df ['Seq'] >= 22645) & (df ['Seq'] < 24106) ] [features]
X_test = df [(df ['Seq'] >= 24106) & (df ['Seq'] < 24838) ] [features]

#mdl = RandomForestRegressor (n_estimators=500, random_state=11,max_leaf_nodes=50, min_samples_leaf=500, max_depth=6)
mdl = RandomForestRegressor (n_estimators=500)
#mdl = RandomForestRegressor (n_estimators=500, random_state=11, max_leaf_nodes=50, min_samples_leaf=500, max_depth=6)
#mdl = GradientBoostingRegressor (n_estimators=500, random_state=11, max_leaf_nodes=50, min_samples_leaf=500, max_depth=6)
#mdl = AdaBoostRegressor (random_state=10)
mdl = eval (mdl)
benchmark (mdl, features, df)
'''
X_train = X_train.drop (['Seq', 'gdp_label'], axis=1)
X_val = X_val.drop (['Seq', 'gdp_label'], axis=1)
X_test = X_test.drop (['Seq', 'gdp_label'], axis=1)
'''

'''
mdl = RandomForestRegressor (n_estimators=500)
mdl.fit (X_train, y_train)

train_preds = mdl.predict (X_train)
val_preds = mdl.predict (X_val)
test_preds = mdl.predict (X_test)

summarize (y_train, train_preds, 'Training')
summarize (y_val, val_preds, 'Vallidation')
summarize (y_test, test_preds, 'Test')
'''
#print ("---------- Regularized below ----------")
#import pdb; pdb.set_trace ()
'''
max_leaf_nodes_range = np.arange (2, 100, 30)
param_grid = {'max_leaf_nodes': max_leaf_nodes_range}
mdl2 = GridSearchCV (RandomForestRegressor (n_estimators=500, verbose=0, warm_start=False), param_grid=param_grid)
'''

'''
            max_features=0.3, max_depth=30, min_samples_split=7, \
            min_samples_leaf=4)
'''

#import pdb; pdb.set_trace ()
