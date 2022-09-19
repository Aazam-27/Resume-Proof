import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

bikes = pd.read_csv("C:/Users/Asus/Desktop/c++ programs/hour.csv")

bikes_prep = bikes.copy()
bikes_prep = bikes_prep.drop(["index","date","casual","registered"],axis=1) 

#plt.subplot(2,2,1)
#plt.title("Temperature Vs Demand")
#plt.scatter(bikes_prep["temp"],bikes_prep["demand"],s=0.002,c="r")

#plt.subplot(2,2,2)
#plt.title("aTemp Vs Demand")
#plt.scatter(bikes_prep["atemp"],bikes_prep["demand"],s=0.002,c="g")

#plt.subplot(2,2,3)
#plt.title("Humidity Vs Demand")
#plt.scatter(bikes_prep["humidity"],bikes_prep["demand"],s=0.002,c="b")

plt.subplot(2,2,4)
plt.title("Windspeed Vs Demand")
plt.scatter(bikes_prep["windspeed"],bikes_prep["demand"],s=0.2,c="r")

plt.tight_layout()

plt.subplot(4,2,1)
plt.title("Average Demand per Season")
colours=["r","g","b","y"]

cat_list=bikes_prep["season"].unique()
cat_average=bikes_prep.groupby("season").mean()["demand"]
plt.bar(cat_list,cat_average,color=colours)


plt.subplot(4,2,2)
plt.title("Average Demand per Month")

cat_list=bikes_prep["month"].unique()
cat_average=bikes_prep.groupby("month").mean()["demand"]
plt.bar(cat_list,cat_average,color=colours)


plt.subplot(4,2,3)
plt.title("Average Demand per Holiday")

cat_list=bikes_prep["holiday"].unique()
cat_average=bikes_prep.groupby("holiday").mean()["demand"]
plt.bar(cat_list,cat_average,color=colours)


plt.subplot(4,2,4)
plt.title("Average Demand per Weekday")

cat_list=bikes_prep["weekday"].unique()
cat_average=bikes_prep.groupby("weekday").mean()["demand"]
plt.bar(cat_list,cat_average,color=colours)


plt.subplot(4,2,5)
plt.title("Average Demand per Year")

cat_list=bikes_prep["year"].unique()
cat_average=bikes_prep.groupby("year").mean()["demand"]
plt.bar(cat_list,cat_average,color=colours)


plt.subplot(4,2,6)
plt.title("Average Demand per Hour")

cat_list=bikes_prep["hour"].unique()
cat_average=bikes_prep.groupby("hour").mean()["demand"]
plt.bar(cat_list,cat_average,color=colours)


plt.subplot(4,2,7)
plt.title("Average Demand per Working Day")

cat_list=bikes_prep["workingday"].unique()
cat_average=bikes_prep.groupby("workingday").mean()["demand"]
plt.bar(cat_list,cat_average,color=colours)


plt.subplot(4,2,8)
plt.title("Average Demand per Weather")

cat_list=bikes_prep["weather"].unique()
cat_average=bikes_prep.groupby("weather").mean()["demand"]
plt.bar(cat_list,cat_average,color=colours)

bikes_prep["demand"].quantile([0.05,0.1,0.15,0.9,0.95,0.99])


correlation = bikes_prep[["temp","atemp","humidity","windspeed","demand"]].corr()

bikes_prep=bikes_prep.drop(["weekday","year","workingday","atemp","windspeed"],axis=1)


df1= pd.to_numeric(bikes_prep["demand"],downcast="float")
plt.acorr(df1,maxlags=12)


df1 = bikes_prep["demand"]
df2=np.log(df1)

plt.figure()
plt.hist(df1,rwidth=0.9,bins=20,)


plt.figure()
plt.hist(df2,rwidth=0.9,bins=20)


bikes_prep["demand"]=np.log(bikes_prep["demand"])

t_1=bikes_prep["demand"].shift(+1).to_frame()
t_1.columns=["t-1"]

t_2=bikes_prep["demand"].shift(+2).to_frame()
t_2.columns=["t-2"]

t_3=bikes_prep["demand"].shift(+3).to_frame()
t_3.columns=["t-3"]

bikes_prep_lag = pd.concat([bikes_prep,t_1,t_2,t_3],axis=1)
bikes_prep_lag=bikes_prep_lag.dropna()


bikes_prep_lag["season"]=bikes_prep_lag["season"].astype("category")
bikes_prep_lag["month"]=bikes_prep_lag["month"].astype("category")
bikes_prep_lag["hour"]=bikes_prep_lag["hour"].astype("category")
bikes_prep_lag["holiday"]=bikes_prep_lag["holiday"].astype("category")
bikes_prep_lag["weather"]=bikes_prep_lag["weather"].astype("category")

bikes_prep_lag=pd.get_dummies(bikes_prep_lag, drop_first=True)


Y=bikes_prep_lag[["demand"]]
X=bikes_prep_lag.drop(["demand"],axis=1)
tr_size=int(0.7*len(X))

X_Train=X.values[0:tr_size]
X_Test=X.values[tr_size:len(X)]


Y_Train=Y.values[0:tr_size]
Y_Test=Y.values[tr_size:len(X)]



from sklearn.linear_model import LinearRegression
std_reg = LinearRegression()
std_reg.fit(X_Train,Y_Train)

r2_train=std_reg.score(X_Train,Y_Train)
r2_test=std_reg.score(X_Test,Y_Test)


Y_predict = std_reg.predict(X_Test)

from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(Y_Test,Y_predict))























