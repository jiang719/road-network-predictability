import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import model_selection
import statsmodels.api as sm
import scipy
from statsmodels.stats.outliers_influence import variance_inflation_factor
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl


def confusion_mat(lst):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for k in range(len(lst)):
        target = lst[k]['target']
        predict = lst[k]['predict']
        if target == 1 and predict == 1:
            TP += 1
        elif target ==1 and predict == 0:
            FN += 1
        elif target == 0 and predict == 1:
            FP += 1
        elif target == 0 and predict == 0:
            TN += 1
    assert TP+TN+FP+FN == len(lst)
    return TP, TN, FP, FN

def indexs(lst, ele):
    for i in range(len(lst)):
        if lst[i] == ele:
           break
    return i
'''
filepath = './results'
filenames = os.listdir(filepath)

data = {}
for filename in filenames:
    city = filename[15:-12]
    file = filepath + '/' + filename
    with open(file, 'r') as f:
         results = json.load(f)
    f.close()
    data[city] = results


#f = open('./results/Relational-GCN-result.json', 'r')
#data = json.load(f)

cities = data.keys()

F1 = {}

for city in cities:
    F1[city] = {}
    samples = data[city].keys()
    for sample in samples:
        lst = data[city][sample]
        TP, TN, FP, FN = confusion_mat(lst)
        if (TP + FP) < 1e-17:
           precision = 0.0
        else:
           precision = TP*1.0/(TP + FP)
        if (TP + FN) < 1e-17:
            recall = 0.0
        else:
            recall = TP*1.0/(TP + FN)
        #print(precision, recall)
        if (precision+recall) < 1e-17:
            F1[city][sample] = 0.0
        else:
            F1[city][sample] = 2.0*precision*recall/(precision + recall)
#print(F1)
f1 = {}
for city in cities:
    f1[city] = {'sample_1': 0, 'sample_2': 0}
    samples = F1[city].keys()
    num = len(samples)/2.0
    sample_1 = 0.0
    sample_2 = 0.0
    for sample in samples:
        #print(sample)
        if sample[-1] == '1':
            sample_1 += F1[city][sample]/num
        elif sample[-1] == '2':
            sample_2 += F1[city][sample]/num
    f1[city]['sample_1'] = sample_1
    f1[city]['sample_2'] = sample_2
'''
#print(f1)
census = pd.read_excel('data.xlsx')
census_heat = copy.deepcopy(census)

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
cities_cs = list(census.loc[:, 'City'])
cities = cities_cs
#print(cities)
'''
for i in range(1,8):
    print('Lambdas_'+ '%1d' % i)
    for city in cities:
        ind = cities_cs.index(city)
        print(census.loc[ind, 'Population-2020']/census.loc[ind, 'Population-'+ '%4d' % years[i-1]])

print('ratio0')
for city in cities:
    ind = cities.index(city)
    print(census.loc[ind, 'road_number']/census.loc[ind, 'intersection_number'])
print('ratio1')
for city in cities:
    ind = cities.index(city)
    print(census.loc[ind, 'road_length']/census.loc[ind, 'intersection_number'])
'''



mpl.rc('font', **{'family' : 'sans-serif', 'sans-serif' : ['Myriad Pro']})
mpl.rcParams['pdf.fonttype'] = 42

census = census.drop('Population-1950', axis = 1)
census = census.drop('Population-1960', axis = 1)
census = census.drop('Population-1970', axis = 1)
census = census.drop('Population-1980', axis = 1)
census = census.drop('Population-1990', axis = 1)
census = census.drop('Population-2000', axis = 1)
census = census.drop('Population-2010', axis = 1)
census = census.drop('Budget', axis = 1)
census_model = copy.deepcopy(census)
census_model = census_model.drop('Lambdas_2', axis = 1)
census_model = census_model.drop('Lambdas_1', axis = 1)
census_model = census_model.drop('Lambdas_4', axis = 1)
census_model = census_model.drop('Lambdas_5', axis = 1)
census_model = census_model.drop('Lambdas_6', axis = 1)
census_model = census_model.drop('Lambdas_7', axis = 1)
census_model = census_model.drop('road_length', axis = 1)
census_model = census_model.drop('intersection_number', axis = 1)
census_model = census_model.drop('road_number', axis = 1)
census_model = census_model.drop('land_area_net', axis = 1)
census_model = census_model.drop('average_road_length', axis = 1)
census_model = census_model.drop('airport_annual_passengers', axis = 1)
census_model = census_model.drop('Population_2020', axis = 1)
census_model = census_model.drop('Area', axis = 1)
census_model = census_model.drop('road_length_density', axis = 1)
census_model = census_model.drop('road_number_density', axis = 1)
census_model = census_model.drop('intersection_density', axis = 1)
census_model = census_model.drop('ratio0', axis = 1)
census_model = census_model.drop('ratio_road', axis = 1)
#census_model = census_model.drop('GDP', axis = 1)
census_model = census_model.drop('Betweenness', axis = 1)
#census_model = census_model.drop('average_betweennese', axis = 1)
train = census_model
train = train.drop([26])

#model = sm.formula.ols('F1 ~ Population + Area + GDP + Lambdas_1 + Lambdas_2 + Lambdas_3 + Lambdas_4 + Lambdas_5 + Lambdas_6 + Lambdas_7', data = train).fit()
model = sm.formula.ols('F1 ~ Lambdas_3 + GDP + ratio1 + average_betweennese', data = train).fit()


ybar = train.F1.mean()
#print('F1 mean: ', ybar)

p = model.df_model
n = train.shape[0]
RSS = np.sum((model.fittedvalues - ybar)**2)
ESS = np.sum(model.resid**2)
F = (RSS/p)/(ESS/(n-p-1))
#print('F value(by hand): ', F)
#print('F value(by model):', model.fvalue)

F_Theory = scipy.stats.f.ppf(q=0.95, dfn = p, dfd = n-p-1)
print('F value of theory: ', F_Theory)
print(model.summary())

plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']
plt.rcParams['axes.unicode_minus'] = False
sns.distplot(a = census.F1, bins = 10, fit = scipy.stats.norm, norm_hist = True, 
    hist_kws={'color':'steelblue', 'edgecolor':'black'},
    kde_kws={'color':'black', 'linestyle':'--', 'label':'pdf'},
    fit_kws={'color':'red', 'linestyle':':', 'label':'Gaussian distribution'})
plt.legend()
plt.savefig('hist.jpg')
#plt.show()
plt.close()

X = sm.add_constant(train.loc[:, ['GDP', 'Lambdas_3', 'ratio1', 'average_betweennese']])

vif = pd.DataFrame()
vif['feature'] = X.columns
vif['VIF Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print('variance_inflation_factor: ')
print(vif, '\n')

print('Pearson correlation: ')
print(census_model.drop('F1', axis = 1).corrwith(census_model.F1))


ND = ["Low node_den","Medium node_den","High node_den"]
LD = ["Large link_den","Medium link_den","Small link_den"]
#census = census_heat
nds = list(census.loc[:,'intersection_density'])
nds.sort()
nd1 = nds[10]
nd2 = nds[20]
#print(gdps)
lstcity = list(census.loc[:, 'City'])

CITIES = {'LG_LL':[], 'LG_ML':[], 'LG_SL':[], 'MG_LL':[], 'MG_ML':[], 'MG_SL':[], 'HG_LL':[], 'HG_ML':[], 'HG_SL':[]}
lds = list(census.loc[:,'road_length_density'])
lds.sort()
ld1 = lds[10]
ld2 = lds[20]
gL = 0
gM = 0
gH = 0
lS = 0
lM = 0
lL = 0
for city in range(30):
    gdp = census.loc[city, 'intersection_density']
    lbd = census.loc[city, 'road_length_density']
    if gdp < nd1:
        fg = 'L'
        gL += 1
    elif gdp < nd2:
        fg = 'M'
        gM += 1
    else:
        fg = 'H'
        gH += 1
    if lbd < ld1:
        fl = 'S'
        lS += 1
    elif lbd < ld2:
        fl = 'M'
        lM += 1
    else:
        fl = 'L'
        lL += 1
    CITIES[fg + 'G_' + fl + 'L'].append(city)
print(CITIES)
f1_values = []
f1_nums = []
for label in CITIES.keys():
    f1 = 0.0
    for city in CITIES[label]:
        f1 += census.loc[city, 'F1']
    if len(CITIES[label]) > 0:
       f1_values.append(round(f1/len(CITIES[label]),2))
    else:
       f1_values.append(round(f1,2))
    f1_nums.append(len(CITIES[label]))
f1_value = np.array([f1_values[0:3], f1_values[3:6], f1_values[6:9]])
f1_value = np.transpose(f1_value)
f1_num = np.array([f1_nums[0:3], f1_nums[3:6], f1_nums[6:9]])
f1_num = np.transpose(f1_num)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5),dpi=1200)
im = ax.imshow(f1_value,cmap='viridis')
#fig.colorbar(im, ax=ax)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cax.tick_params(labelsize=10)

fig.colorbar(im, cax=cax)

ax.set_xticks(np.arange(len(ND)))
ax.set_yticks(np.arange(len(LD)))
ax.set_xticklabels(ND)
ax.set_yticklabels(LD)

plt.setp(ax.get_xticklabels(), rotation=0, rotation_mode="anchor",fontsize =10)
plt.setp(ax.get_yticklabels(), rotation=90,ha= 'center',rotation_mode="anchor",fontsize =10)

for i in range(len(ND)):
    for j in range(len(LD)):
        if f1_value[i,j] > 0:
            text = ax.text(j, i, f1_value[i, j],
                       ha="center", va="center", color="w",fontsize =20)
        else:
            text = ax.text(j, i, 'None',
                       ha="center", va="center", color="w",fontsize =20)
#ax.set_title("Average F1 scores")
fig.tight_layout()
#plt.title(lbd_num, horizontalalignment = 'left')
plt.savefig('density' + '.svg',bbox_inches = 'tight')
plt.savefig('density' + '_heatmap.jpg',bbox_inches = 'tight') #change the name here
#pdf = PdfPages(lbd_num + '.pdf')
plt.savefig('density' + '.pdf',bbox_inches = 'tight')   #change the name here
#pdf.savefig()
#plt.show()
plt.close()
#pdf.close()

