
# coding: utf-8

# In[1]:

#%load "/user_home/w_abray/HomeBatteryOptimization1.py"


# In[1]:

"""
Created on Wed Dec 30 13:59:39 2015
https://www.chrisstucchio.com/blog/2014/work_hours_with_python.html
@author: abray
"""
#Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import datetime as dt


#Load 1st 50 rows of data. Size to house size.
priceData = pd.DataFrame.from_csv("/user_home/w_abray/ComedPricesJan16.csv", parse_dates=True, index_col=0, header=1)
loadData = pd.DataFrame.from_csv("/user_home/w_abray/ComedHouseLoadJan16.csv", parse_dates=True, index_col=0, header=1)
#Reorder columns
cols = loadData.columns.tolist()
cols.insert(0,cols.pop(cols.index("24:00:00")))
loadData = loadData.reindex(columns = cols)
#Rename 1st column
loadData.columns.values[0] = "0:00"
#Shift column down a day
loadData["0:00"] = loadData["0:00"].shift(1)
#Fill in missing data with next hour's data
loadData["0:00"][0] = loadData["1:00"][0]

#Move hours into a series
stacked = loadData.stack()
ind = [str(x[0])[:-9] + " " + x[1] for x in stacked.index]
pdt = pd.to_datetime(ind)
frame = pd.DataFrame(stacked)
lData = frame.set_index(pd.DatetimeIndex(ind))
lData["Hourly Load (kWh)"] = lData[0]
lData = lData.drop(0, axis=1)

#Merge price and load
data = lData.merge(priceData, left_index=True, right_index=True)


# In[2]:

#1 week's worth of data 7*24
data = data.iloc[:168,]
n = len(data)
#Check that it's coming through
print data.columns.values

from pylab import figure, show, legend, ylabel
 
# create the general figure
fig1 = figure()
 
# and the first axes using subplot populated with data 
ax1 = fig1.add_subplot(111)
line1 = ax1.plot(data.index, data["Hourly Load (kWh)"])
ylabel(data.columns.values[0])
 
# now, the second axes that shares the x-axis with the ax1
ax2 = fig1.add_subplot(111, sharex=ax1, frameon=False)
line2 = ax2.plot(data.index, data["Hourly Price ($/kWh)"],color="red")
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ylabel(data.columns.values[1])


# In[3]:

np.corrcoef(data.iloc[:,0],data.iloc[:,1])[1,0]


# In[4]:

#Regular Electricity Cost
#print data[:1]
hourlyCost = data["Hourly Load (kWh)"]*data["Hourly Price ($/kWh)"]
weekCost = sum(hourlyCost)
print weekCost


# In[5]:

#1 - Optimize battery only, no load

#Parameters
BatterySize = 7 #7 kWh
BegCharge = 1
maxPower = 2

#Energy moved in hour
#(+) : Selling electricity, get $
#(-) : Buying electricity, lose $
power = np.ones([n,1])
def obj(power):
    price = np.array(data["Hourly Price ($/kWh)"])#Price of Power
    cashFlow = np.transpose(price * np.transpose(power)) #Cash flow in each hour
    cumSumCash = np.cumsum(cashFlow) #Cumulative sum of cash flow in each hour
    
    #Return the last cumulative as a negative value since we're minimizing
    return -1*cumSumCash[len(cumSumCash)-1]
    
#Maximum amount of power that can be moved
powerRange = np.tile([-maxPower,maxPower],[n,1])

#optimize
cons = ({'type':'ineq','fun': lambda x:  np.cumsum(x)},#Battery Size minimum (0)
        {'type':'ineq','fun': lambda x: BatterySize - np.cumsum(x)}, #Maximum battery size
        )

res = minimize(obj, power, bounds = powerRange, constraints=cons)
print res


# In[6]:

#Final optimal money to make
fin = -1*res.fun
powerMoved = res.x
#State of the battery by cum suming power flow
batteryState = cumsum(powerMoved)


# In[7]:

#print powerMoved
print fin


# In[8]:

#Graph Results
fig2 = figure()
 
# and the first axes using subplot populated with data 
ax1 = fig2.add_subplot(111)
line1 = ax1.plot(data.index, data["Hourly Price ($/kWh)"])
ylabel(data.columns.values[1], color = "blue")
 
# now, the second axes that shares the x-axis with the ax1
ax2 = fig2.add_subplot(111, sharex=ax1, frameon=False)
line2 = ax2.plot(data.index, batteryState,color="red")
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ylabel("Battery State (kWh)",color="red")


# In[9]:

#2 - Optimize battery only, load, discharge to 50%

#Parameters
BatterySize = 7 #7 kWh
BatteryMin = np.repeat(BatterySize*.5,n)
BegCharge = 1
maxPower = 2


# In[10]:

#2 - Optimize battery only, load, discharge to 50%

#Parameters
BatterySize = 7 #7 kWh
BatteryMin = np.repeat(BatterySize*.5,n)
BegCharge = 1
maxPower = 2

#Energy moved in hour
#(+) : Charging battery, buying electricity, lose $
#(-) : Draining battery, selling electricity, Gain $
power = np.ones([n,1])
battery = np.ones([n,0])
def obj(power):
    price = np.array(data["Hourly Price ($/kWh)"])#Price of Power
    cashFlow = -np.transpose(price * np.transpose(power + np.array(data["Hourly Load (kWh)"]))) #Cash flow in each hour
    
    cumSumCash = np.cumsum(cashFlow) #Cumulative sum of cash flow in each hour
    
    #Return the last cumulative as a negative value since we're minimizing
    return -1*cumSumCash[len(cumSumCash)-1]
    
#Maximum amount of power that can be moved
powerRange = np.tile([-maxPower,maxPower],[n,1])

#optimize
cons = ({'type':'ineq','fun': lambda x: np.cumsum(x)-data["Hourly Load (kWh)"]},#Battery Size minimum (0)
        {'type':'ineq','fun': lambda x: np.cumsum(x)-np.array(BatteryMin)},#Battery Size minimum charge (50%)
        {'type':'ineq','fun': lambda x: BatterySize - np.cumsum(x)}, #Maximum battery size
        )

res = minimize(obj, power, bounds = powerRange, constraints=cons)



# In[11]:

#print res
#print power
#print power + 
#plt.plot( np.array(data["Hourly Load (kWh)"]))


# In[12]:

#Final optimal money to make
fin = -1*res.fun
powerMoved = res.x
#State of the battery by cum suming power flow
batteryState = cumsum(powerMoved)
print powerMoved[:10]
print fin


# In[13]:

#Graph Results
fig2 = figure()
 
# and the first axes using subplot populated with data 
ax1 = fig2.add_subplot(111)
line1 = ax1.plot(data.index, data["Hourly Price ($/kWh)"])
ylabel(data.columns.values[1], color = "blue")
 
# now, the second axes that shares the x-axis with the ax1
ax2 = fig2.add_subplot(111, sharex=ax1, frameon=False)
line2 = ax2.plot(data.index, batteryState,color="red")
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ylabel("Battery State (kWh)",color="red")


# In[14]:

#Battery with Solar Cell
#3
#Not Optimized
#Parameters
BatterySize = 7 #7 kWh
BatteryMin = np.repeat(BatterySize*.5,n)
BegCharge = 1
maxPower = 2

#Energy moved in hour
#(+) : Charging battery, buying electricity, lose $
#(-) : Draining battery, selling electricity, Gain $
power = np.ones([n,1])
battery = np.ones([n,0])

hourlyLoad = np.array(data["Hourly Load (kWh)"])
price = np.array(data["Hourly Price ($/kWh)"])#Price of Power

#battery = -hourlyLoad+solarCont
costInHour = price*hourlyLoad




# In[15]:

from datetime import datetime, time
from __future__ import division

#Don't pay for electricity when the sun's out
data ["Sun"] = 1*((data.index.hour > 17)| (data.index.hour < 17-4.5) )
#print data
data["Hourly Load Solar"] = np.ones([n,1])
batCharge = 0
prevsun=1
for index, row in data.iterrows():
    sun = row[2]
    if (prevsun == 0) & (sun == 1):
        print "Recharge!"
        batCharge = BatterySize/2
        print batCharge
        
    
    minum = min(data["Hourly Load (kWh)"][index], batCharge)
    data["Hourly Load Solar"][index] = data["Hourly Load (kWh)"][index] - minum
    batCharge = batCharge - minum
    prevsun = sun
    



# In[16]:

data["Hourly Cost"] = data["Hourly Price ($/kWh)"]*data["Sun"]*data["Hourly Load Solar"]
#print data
totalCost = sum(data["Hourly Cost"])
print totalCost


# In[ ]:



