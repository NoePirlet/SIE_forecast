# -*- coding: utf-8 -*-
"""
Created on Thu May 13 11:13:23 2021

@author: noepi
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import scipy.stats
import plotly.graph_objects as go
# Open file------------------------------------------------------------------------------------------------
f = open("extent_data.txt", "r")
contents = f.readlines()


# Description of the data file------------------------------------------------------------------------------
# FracYear YYYY MM DD SIE[km^2]
# Area: Northern Hemisphere
# Quantity: Sea Ice Extent
# Product: EUMETSAT OSI SAF Sea Ice Index v2p1 (demonstration)
# References: Product User Manual for OSI-420, Lavergne et al., v1.0, November 2020
# Source: EUMETSAT OSI SAF v2 data, with R&D input from ESA CCI (Lavergne et al., 2019)
# Creation date: 2021-03-30 10:02:15.039862

# Count lines----------------------------------------------------------------------------------------------
K = 0
for x in contents:
    if not x.startswith(' '):
        K = K + 1
# Extract data---------------------------------------------------------------------------------------------
FracYear = np.zeros(K)
Year = np.zeros(K)
Month = np.zeros(K)
Day = np.zeros(K)
SIE = np.zeros(K)  # [km^2]


K = 0
for x in contents:
    if not x.startswith(' '):
        strArray = x.split(' ')
        FracYear[K] = float(strArray[0])
        Year[K] = float(strArray[1])
        Month[K] = float(strArray[2])
        Day[K] = float(strArray[3])
        SIE[K] = float(strArray[4])
        K = K + 1

# The Gaussian---------------------------------------------------------------------------------------------
def get_gaussian(N_yr):

    N_m = N_yr*12  # Nombre de mois avec le nombre d'année sélectionné

    SIE_uncertainty = 900000  # [km^2] Integrated Ice-Edge Error


    j = 0
    k = 0
    l = 0
    m = 0
    w = 0
    c = 0
    v = 0
    b = 0
    g = 0

    h = 0
    q = 0
    s = 0
    SIE_aug = np.zeros(N_yr)
    SIE_jui = np.zeros(N_yr)
    SIE_jjl = np.zeros(N_yr)

    SIE_sept = np.zeros(N_yr)
    SIE_may = np.zeros(N_yr)
    SIE_oct = np.zeros(N_yr)
    SIE_nov = np.zeros(N_yr)
    SIE_dec = np.zeros(N_yr)
    SIE_jan = np.zeros(N_yr)
    SIE_fev = np.zeros(N_yr)
    SIE_mar = np.zeros(N_yr)
    SIE_avr = np.zeros(N_yr)

    if N_yr >= 8:
        SIE_may = np.zeros(N_yr-1)
    if N_yr >= 8:
        SIE_avr = np.zeros(N_yr-1)
    if N_yr >= 8:
        SIE_jui = np.zeros(N_yr-1)
    if N_yr >= 9:
        SIE_dec = np.zeros(N_yr-1)

    for i in np.arange(N_m):
        moy = (np.arange(0, N_m)) % 12+1  # mois 1-12 pour chaque année
        if moy[i] == 9 and SIE[i] > 0:
            SIE_sept[j] = SIE[i]
            j = j+1
        if moy[i] == 10 and SIE[i] > 0:
            SIE_oct[l] = SIE[i]
            l = l+1
        if moy[i] == 11 and SIE[i] > 0:
            SIE_nov[w] = SIE[i]
            w = w+1
        if moy[i] == 12 and SIE[i] > 0:
            SIE_dec[c] = SIE[i]
            c = c+1
        if moy[i] == 1 and SIE[i] > 0:
            SIE_jan[v] = SIE[i]
            v = v+1
        if moy[i] == 2 and SIE[i] > 0:
            SIE_fev[b] = SIE[i]
            b = b+1
        if moy[i] == 3 and SIE[i] > 0:
            SIE_mar[g] = SIE[i]
            g = g+1
        if moy[i] == 4 and SIE[i] > 0:
            SIE_avr[k] = SIE[i]
            k = k+1
        if moy[i] == 5 and SIE[i] > 0:
            SIE_may[m] = SIE[i]
            m = m+1
        if moy[i] == 6 and SIE[i] > 0:
            SIE_jui[h] = SIE[i]
            h = h+1
        if moy[i] == 7 and SIE[i] > 0:
            SIE_jjl[q] = SIE[i]
            q = q+1
        if moy[i] == 8 and SIE[i] > 0:
            SIE_aug[s] = SIE[i]
            s = s+1

    if N_yr == 7:
        #pas de données pour avril et mai  1986 qui interviennent dans le calcul de sept 1986  (no data for june)
        SIE_mean = (SIE[N_m+2] + SIE[N_m+1] + SIE[N_m+0] + SIE[N_m-1] + SIE[N_m-2] + SIE[N_m-3])/6
    elif N_yr == 9:
        #pas de données pour dec 1987 qui intervient dans le calcul de sept 1988
        SIE_mean = (SIE[N_m+4] + SIE[N_m+3] + SIE[N_m+2] + SIE[N_m+1] + SIE[N_m+0] + SIE[N_m-2] + SIE[N_m-3])/7
    else:
        SIE_mean = (SIE[N_m+4] + SIE[N_m+3] + SIE[N_m+2] + SIE[N_m+1] + SIE[N_m+0] + SIE[N_m-1] + SIE[N_m-2] + SIE[N_m-3])/8  

    SIE_mean_mean = (np.mean(SIE_may) + np.mean(SIE_avr) + np.mean(SIE_mar) + np.mean(SIE_fev) + np.mean(SIE_jan) + np.mean(SIE_dec) +
                     np.mean(SIE_nov) + np.mean(SIE_oct))/8   
    mu = np.mean(SIE_sept) + (SIE_mean - SIE_mean_mean)  # mean
    sigma = ((1/(N_yr) * (np.var(SIE_sept) + np.var(SIE_may) + np.var(SIE_oct) + np.var(SIE_nov) + np.var(SIE_dec) + np.var(SIE_jan) + np.var(SIE_fev) +
             np.var(SIE_mar) + np.var(SIE_avr))+SIE_uncertainty))**(1/2)   #variance

    Gaussian = np.random.normal(mu, sigma, 10000)
    return(Gaussian, mu, sigma, SIE_sept, SIE_may)


# Setup----------------------------------------------------------------------------------------------------
year = np.arange(1981, 2021)  # 2020
n = 42 #nombre d'année qu'on veut prendre en compte (41 pour 2019, 42 pour 2020, 43 pour 2021)

# Another def to Sept_Sea_Ice_extent-----------------------------------------------------------------------
SSIE = np.zeros(n)  # de 1979 à 2020  ######plus de 1981
for i in range(0, n):
    SSIE[i] = SIE[12*i + 9-1]

# Mean vector---------------------------------------------------------------------------------------------------
mu_vec = np.zeros(n-2)
for i in np.arange(2, n):  # 1981 a 2020
    mu_vec[i-2] = get_gaussian(i)[1]

for i in range(1, n-2):   
        if mu_vec[i] < 0:
            mu_vec[i] = (mu_vec[i-1]+mu_vec[i+1])/2
            print("Pas de valeur d'étendue de glace pour l'année :" + str(1981+i))

# Trend----------------------------------------------------------------------------------------------------

slope1, intercept1, r1, p1, se1 = scipy.stats.linregress(year[:i], SSIE[:i])
trend1 = intercept1 + slope1 * year
slope2, intercept2, r2, p2, se2 = scipy.stats.linregress(year[:i], mu_vec[:i])
trend2 = intercept2 + slope2 * year

# Correction Bias of the trend-----------------------------------------------------------------------------
trend_diff = trend1-trend2
new_mu_vec = mu_vec + trend_diff

# Standart deviation---------------------------------------------------------------------------------------
std_dev_p = np.zeros(n-2)
std_dev_n = np.zeros(n-2)
j = 0
for i in np.arange(2, n):  # 1981 a 2020
    std_dev_p[i-2] = new_mu_vec[i-2] + 2*get_gaussian(i)[2]
    std_dev_n[i-2] = new_mu_vec[i-2] - 2*get_gaussian(i)[2]

            
        

# Probability----------------------------------------------------------------------------------------------

Prob= np.zeros(n-2)
for i in np.arange(2, n):
    mu= get_gaussian(i)[1]
    sigma= get_gaussian(i)[2]
    Prob[i-2] = norm.cdf((SSIE[i-1] - new_mu_vec[i-2])/sigma)


SSIE_shift = np.zeros(n-2)
for i in np.arange(n-2):
    SSIE_shift[i]= SSIE[i+2]



# Verification of retrospective forecast-------------------------------------------------------------------
o = np.zeros(n-2)
for i in range(0, n-2):
    if SSIE[i+2] < SSIE[i-1+2] and SSIE[i+2] > 0:
        o[i]= 1

Prob_climato = sum(o)/np.size(o)
BS_ref = Prob_climato*(1-Prob_climato)


som= 0
for i in range(0, n-2):
    som += (Prob[i] - o[i])**2

BS= som/(n)
BSS = 1-(BS/BS_ref)

print(BS)
print(BS_ref)
print(BSS)


# Graphics-------------------------------------------------------------------------------------------------

fig, ax= plt.subplots()
ax.fill_between(year, (std_dev_p), (std_dev_n), color='b', alpha=.1)
plt.plot(year, SSIE_shift, label ='Observation')
#plt.plot(x, trend1)
#plt.plot(x, trend2)
plt.plot(year, new_mu_vec, label = 'Forecast')
plt.legend()
plt.show()


fig, ax = plt.subplots()
col = []
for i in np.arange(40):
    if o[i] == 0:
        col.append("darkred")
    else:
        col.append("limegreen")
plt.bar(np.arange(1981, 2021), Prob, color =col, label = 'Probability of our model')
#plt.plot(np.arange(1981, 2021), o, '--', linewidth=1, color='red', label = 'Event occur or not' )
#plt.plot(np.arange(1981, 2021), Prob, color ='white', label = 'Probability of our model')
colors = {'Event':'limegreen', 'Non-event':'darkred'}        
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
plt.xlabel('Time[year]')
plt.ylabel('Probabitlity [%]')
fig.patch.set_facecolor('#205873')
ax.set_facecolor('#205873')
#plt.legend()
plt.show()


fig = go.Figure(go.Indicator(
     mode = "gauge+number",
     value = BSS,
     domain = {'x': [0, 0.5], 'y': [0, 0.5]},
     title = {'text': "Performance of the forecast (BBS)", 'font': {'size': 40}},
     delta = {'reference': 400, 'increasing': {'color': "teal"}},
     number={"font":{"size":20}},
     gauge = {
         'axis': {'range': [None, 1], 'tickwidth': 4, 'tickcolor': "white", 'ticklen': 10, 'tickfont_size' :40},
         'bar': {'color': "limegreen" , 'thickness' : 0.5},
         'bgcolor': "white",
         'borderwidth': 2,
         'bordercolor': "black",
         'steps': [
            {'range': [0, 250], 'color': 'green'},
            {'range': [250, 400], 'color': 'royalblue'}],
        }))
 
fig.update_layout(paper_bgcolor = "#205873", font = {'color': "white", 'family': "Arial"})
 
fig.show()


