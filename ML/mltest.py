# //-----------------------------------
# from sklearn.datasets import fetch_california_housing
# import pandas as pd
# ch=fetch_california_housing()
# print(ch)
#
# x=pd.DataFrame(ch.data,columns=ch.feature_names)
# y=pd.DataFrame(ch.target, columns=["AveInc"])
# data=pd.concat([x,y],axis=1)
# data.head()
# -----------------------------------//

# //-----------------------------------
# import numpy as np
# import pandas as pd
# from sklearn import datasets
# iris = datasets.load_iris()
# iris
#
# x=pd.DataFrame(iris.data, columns=iris.feature_names)
# y=pd.DataFrame(iris['target_names'][iris['target']], columns=['species'])
# print(x)
# iris_df=pd.concat([x,y], axis=1)
# iris_df.head()
# -----------------------------------//

# //-----------------------------------
# import statsmodels.api as sm
# iris_data = sm.datasets.get_rdataset('iris',package='datasets')
# iris_data.title
#
# print(iris_data.__doc__)
# -----------------------------------//

# //-----------------------------------
# import statsmodels.api as sm
# titanic_data=sm.datasets.get_rdataset('Titanic',package='datasets')
# titanic_data
#
# titanic_data.data.head()
# -----------------------------------//

# //-----------------------------------
# import statsmodels.api as sm
# airpassengers_data=sm.datasets.get_rdataset('AirPassengers')
# airpassengers=airpassengers_data.data
# airpassengers.head()
#
# def yearfraction2datetime(yearfraction, startyear=0):
#     import  datetime
#     import dateutil
#     year=int(yearfraction)+startyear
#     month=int(round(12*(yearfraction-year)))
#     delta=dateutil.relativedelta.relativedelta(months=month)
#     date=datetime.datetime(year,1,1)+delta
#     return date
#
# airpassengers['datetime']=airpassengers.time.map(yearfraction2datetime)
# airpassengers.head()
# -----------------------------------//

# //-----------------------------------
# import seaborn as sns
# iris=sns.load_dataset('iris')
# x=iris.iloc[:,:-1]
# x.head()
#
# from sklearn.preprocessing import scale
# x_scaled=scale(x)
# x_scaled[:5,:]
# x_scaled.mean(axis=0)
# for scaled_mean in x_scaled.mean(axis=0):
#     print('{:10.9f}'.format(scaled_mean))
#
# from sklearn.preprocessing import robust_scale
# iris_robust_scaled=robust_scale(x)
# iris_robust_scaled[:5,:]
#
# from sklearn.preprocessing import minmax_scale
# iris_minmax_scaled=minmax_scale(x)
# iris_minmax_scaled[:5,:]
#
# from sklearn.preprocessing import maxabs_scale
# iris_maxabs_scaled=maxabs_scale(x)
# iris_maxabs_scaled[:5,:]
# -----------------------------------//

# //-----------------------------------
# import seaborn as sns
# iris=sns.load_dataset('iris')
# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# iris_scaled=sc.fit_transform(iris.iloc[:,:-1])
# iris_scaled[:5,:]
#
# iris_origin=sc.inverse_transform(iris_scaled)
# iris_origin[:5,:]
#
# # 추가 실습 내용
# from sklearn.preprocessing import MinMaxScaler
# min_max_scaler=MinMaxScaler()
# iris_scaled=min_max_scaler.fit_transform(iris.iloc[:,:-1])
# iris_scaled[:5,:]
#
# # 추가 실습 내용
# iris_origin=min_max_scaler.inverse_transform(iris_scaled)
# iris_origin[:5,:]
#
# # 추가 실습 내용
# from sklearn.preprocessing import MaxAbsScaler
# max_abs_scaler=MaxAbsScaler()
# iris_scaled=max_abs_scaler.fit_transform(iris.iloc[:,:-1])
# iris_scaled[:5,:]
#
# # 추가 실습 내용
# iris_origin=max_abs_scaler.inverse_transform(iris_scaled)
# iris_origin[:5,:]
# -----------------------------------//

# //-----------------------------------
# from sklearn.preprocessing import normalize
# x=[[1.,-1.,2.],[2.,0.,0.],[0.,1.,-1.]]
# x_normalized_l1=normalize(x, norm='l1')
# x_normalized_l1
#
# x_normalized_l2=normalize(x, norm='l2')
# x_normalized_l2
# -----------------------------------//

# //-----------------------------------
# import seaborn as sns
# iris=sns.load_dataset('iris')
# y=iris.species
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# le.fit(y)
# species=le.transform(y)
# species
#
# from sklearn.preprocessing import OneHotEncoder
# enc=OneHotEncoder()
# enc.fit(species.reshape(-1,1))
#
# iris_onehot=enc.transform(species.reshape(-1,1))
# iris_onehot
#
# iris_onehot.toarray()
# -----------------------------------//

# //-----------------------------------
# import seaborn as sns
# iris=sns.load_dataset('iris')
# x=iris.iloc[:,:-1]
# import random
# for col in range(4):
#     x.iloc[[random.sample(range(len(iris)), 10)],col]=float('nan')
# x.head()
#
# x.mean(axis=0)
#
# from sklearn.preprocessing import Imputer
# imp_mean=Imputer(missing_values='NaN',strategy='mean',axis=0)
# imp_mean.fit_transform(x)[0:5,:]
#
# x.median(axis=0)
#
# imp_median=Imputer(missing_values='NaN',strategy='median',axis=0)
# imp_median.fit_transform(x)[0:5,:]
#
# x.mode(axis=0)
#
# imp_mostfreq=Imputer(missing_values='NaN',strategy='mostfreq',axis=0)
# imp_mostfreq.fit_transform(x)[0:5,:]
# -----------------------------------//

# //-----------------------------------
# import seaborn as sns
# iris_df=sns.load_dataset('iris')
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(iris_df.iloc[:,0:4],iris_df.species,test_size=0.3)
# x_train.shape
# -----------------------------------//

# //-----------------------------------
# 실습을 위한 데이터
# import pandas as pd
# redwine=pd.read_csv('winequality-red.csv', delimiter=';')
# X=redwine.iloc[:,:-1]
# y=redwine.iloc[:,-1]
# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
#
# from sklearn.preprocessing import MinMaxScaler
# scaler=MinMaxScaler(feature_range=(0,1))
# X_train_scaled=scaler.fit_transform(X_train)
# X_test_scaled=scaler.fit_transform(X_test)
#
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# rf=RandomForestClassifier(n_estimators=10).fit(X_train_scaled,y_train)
# feature_importance_rf=pd.DataFrame(data=np.c_[X.columns.values, rf.feature_importances_], columns=['feature','importance'])
# feature_importance_rf.sort_values('importance',ascending=False,inplace=True)
# feature_importance_rf
#
#
# from sklearn.feature_selection import RFE
# select=RFE(RandomForestClassifier(n_estimators=5,random_state=42),n_features_to_select=5)
# select.fit(X_train_scaled, y_train)
# feature_importance_rfe=pd.DataFrame(data=np.c_[X.columns.values, select.get_support()], columns=['feature','importance'])
# feature_importance_rfe.sort_values('importance',ascending=False,inplace=True)
# feature_importance_rfe
# -----------------------------------//

# //-----------------------------------
# 실습을 위한 데이터
# from sklearn.datasets import load_boston
# boston=load_boston()
# X=boston.data
# y=boston.target
# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
#
# # 표준화
# from sklearn.preprocessing import MinMaxScaler
# scaler=MinMaxScaler(feature_range=(0,1))
# X_train_scaled=scaler.fit_transform(X_train)
# X_test_scaled=scaler.fit_transform(X_test)
#
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import cross_val_score
# model_boston=LinearRegression().fit(X_train_scaled,y_train)
# cross_val_score(model_boston, X_train_scaled,y_train,cv=5)
#
# model_boston.coef_
#
# import pandas as pd
# import numpy as np
# feature_importance_lr=pd.DataFrame(np.c_[boston.feature_names, model_boston.coef_.ravel()])
# feature_importance_lr.columns=['feature','coef']
# feature_importance_lr.sort_values('coef', ascending=False,inplace=True)
# feature_importance_lr
# -----------------------------------//

# //-----------------------------------
# from sklearn.datasets import load_iris
# from sklearn.feature_selection import SelectKBest, chi2
# X, y=load_iris(return_X_y=True)
# X.shape
#
# X_new=SelectKBest(chi2,k=1).fit_transform(X,y)
# X_new.shape
#
# X_new
# -----------------------------------//

# //-----------------------------------
# import pandas as pd
# import numpy as np
#
# redwine=pd.read_csv('winequality-red.csv', delimiter=';')
#
# # 데이터셋 샘플링
# train=redwine.sample(frac=0.7) # 학습 데이터 생성
# test=redwine.loc[~redwine.index.isin(train.index)] # 테스트 데이터 생성
# train.head()
#
# print(train.shape, test.shape)
#
# train_x=train.iloc[:,:-1] # 정답(label)을 제외한 학습데이터
# train_y=train.iloc[:,:-1] # 학습데이터 중에서 정답(label) 데이터
# test_x=test.iloc[:,-1] # 정답(label)을 제외한 테스트 데이터
# test_x=test.iloc[:,-1] # 테스트 데이터 중에서 정답(label) 데이터
#
# from sklearn.neural_network import MLPClassifier
#
# # hidden_layer_sizes=(50,30) : 퍼셉트론 50개와 30개짜리로 은닉층 두 개
# mlp=MLPClassifier(hidden_layer_sizes=(50,30))
# mlp.fit(train_x, train_y)
# -----------------------------------//

# //-----------------------------------
# from sklearn import datasets, linear_model
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# diabetes=datasets.load_diabetes()
# X=diabetes.data[:150]
# y=diabetes.target[:150]
#
# lasso=linear_model.Lasso()
# cv=KFold(5,shuffle=True,random_state=0)
# print(cross_val_score(lasso,X,y,cv=cv))
# -----------------------------------//

# //-----------------------------------
# from sklearn import svm, metrics
# # CSV 파일을 읽어 들이고 가공하기
# def load_csv(fname):
#     labels=[]
#     images=[]
#     with open(fname,'r') as f:
#         for line in f:
#             cols=line.split(',')
#             if len(cols) < 2: continue
#             labels.append(int(cols.pop(0))) # 라벨(정답) 추출
#             vals = list(map(lambda n:int(n)/256,cols)) # 테스트할 데이터 가공
#             images.append(vals)
#     return {'labels':labels,'images':images}
# data=load_csv('train.csv')
# test=load_csv('t10k.csv')
# # 학습하기
# clf=svm.SVC()
# clf.fit(data['images'],data['labels'])
# # 예측하기
# predict=clf.predict(test['images'])
# # 결과 확인하기
# ac_score=metrics.accuracy_score(test['labels'], predict)
# cl_report=metrics.classification_report(test['labels'], predict)
# print('정답률=', ac_score)
# print('리포트=')
# print(cl_report)
# -----------------------------------//

# //-----------------------------------
# import pandas as pd
# from sklearn import svm, metrics
# from sklearn.model_selection import GridSearchCV
# # MNIST 학습 데이터 읽어 들이기
# train_csv=pd.read_csv('train.csv')
# test_csv=pd.read_csv('t10k.csv')
# # 필요한 열 추출하기
# train_label=train_csv.iloc[:,0]
# train_data=train_csv.iloc[:,1:785]
# test_label=test_csv.iloc[:,0]
# test_data=test_csv.iloc[:,1:785]
# print('학습 데이터의 수=', len(train_label))
# # 그리드 서치 매개변수 설정
# params=[{'C':[1,10,100,1000], 'kernel':['linear']},
#        {'C':[1,10,100,1000], 'kernel':['rbf'],'gamma':[1,10,100,1000]}] # gamma:커...
# # 그리드 서치 수행
# svc=svm.SVC()
# clf=GridSearchCV(svc, params)
# clf.fit(train_data, train_label)
# print('학습기=',clf.best_estimator_)
# # 테스트 데이터 확인하기
# pre=clf.predict(test_data)
# ac_score=metrics.accuracy_score(pre,test_label)
# cl_report=metrics.classification_report(test_label,pre)
# print('정답률=',ac_score)
# print('리포트=')
# print(cl_report)
# -----------------------------------//

# //-----------------------------------
# from sklearn import  datasets, linear_model
# from sklearn.model_selection import train_test_split
# diabetes=datasets.load_diabetes()
# X=diabetes.data
# y=diabetes.target
# x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
# lasso=linear_model.Lasso()
# model=lasso.fit(x_train,y_train)
# model
#
# pred=model.predict(x_test)
# pred[:10]
#
# from sklearn.metrics import mean_squared_error
# mean_squared_error(y_test,pred)
#
# import math
# rmse=math.sqrt(mean_squared_error(y_test,pred))
# rmse
#
# from sklearn.metrics import mean_absolute_error
# mean_absolute_error(y_test,pred)
#
# from sklearn.metrics import explained_variance_score
# explained_variance_score(y_test,pred)
#
# from sklearn.metrics import r2_score
# r2_score(y_test, pred)
# -----------------------------------//

# //-----------------------------------
# import csv
# with open('basket.csv','r',encoding='UTF8') as cf:
#     transactions=[]
#     r=csv.reader(cf)
#     for row in r:
#         transactions.append(row)
# transactions
#
# from apyori import apriori
# rules=apriori(transactions, min_support=0.1,min_confidence=0.1)
# results=list(rules)
# type(results)
#
# results[0]
#
# results[10]
#
# print('lhs \trhs \t\tsupport\t\tconfidence\tlift')
# for row in results:
#     support=row[1]
#     ordered_stat=row[2]
#     for ordered_item in ordered_stat:
#         lhs=[x for x in ordered_item[0]]
#         rhs=[x for x in ordered_item[1]]
#         confidence=ordered_item[2]
#         lift=ordered_item[3]
#         print(lhs,' => ',rhs,'\t{:>5.4f}\t{:>5.4f}\t{:>5.4f}'.\
#               format(support,confidence,lift))
# -----------------------------------//

# //-----------------------------------
# make_classification함수 - 설정에 따른 분류용 가상 데이터를 생성하는 명령
# 인수
# n_samples : 표본 데이터의 수, 디폴트 100
# n_features : 독립 변수의 수, 디폴트 20
# n_informative : 독립 변수 중 종속 변수와 상관 관계가 있는 성분의 수, 디폴트 2
# n_redundant : 독립 변수 중 다른 독립 변수의 선형 조합으로 나타나는 성분의 수, 디폴트 2
# n_repeated : 독립 변수 중 단순 중복된 성분의 수, 디폴트 0
# n_classes : 종속 변수의 클래스 수, 디폴트 2
# n_clusters_per_class : 클래스 당 클러스터의 수, 디폴트 2
# weights : 각 클래스에 할당된 표본 수
# random_state : 난수 발생 시드
# 반환값
# X : [n_samples, n_features] 크기의 배열. 독립 변수
# y : [n_samples] 크기의 배열. 종속 변수
# %matplotlib inline
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_classification
#
# plt.title('Multi-class, two informative features, one cluster')
# X, y =make_classification(n_samples=30, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, n_classes=3, random_state=123)
# plt.scatter(X[:,0],X[:,1],marker='o',c=y,s=100,edgecolors='k',linewidths=2)
# plt.show()
#
# from sklearn.cluster import KMeans
# model=KMeans(n_clusters=3, init='random')
# model.fit(X)
#
# model.cluster_centers_
#
# pred=model.predict(X)
# pred
#
# c0, c1, c2 = model.cluster_centers_
# plt.scatter(X[model.labels_==0,0],X[model.labels_==0,1],s=100,marker='v',c='r')
# plt.scatter(X[model.labels_==1,0],X[model.labels_==1,1],s=100,marker='^',c='b')
# plt.scatter(X[model.labels_==2,0],X[model.labels_==2,1],s=100,marker='v',c='y')
# plt.scatter(c0[0],c0[1],s=200,c='r')
# plt.scatter(c1[0],c1[1],s=200,c='b')
# plt.scatter(c2[0],c2[1],s=200,c='y')
# plt.show()
#
# import pandas as pd
# import numpy as np
# df=pd.DataFrame(np.hstack([X,
#                            np.linalg.norm(X-c0,axis=1)[:,np.newaxis],
#                            np.linalg.norm(X-c1,axis=1)[:,np.newaxis],
#                            np.linalg.norm(X-c2,axis=1)[:,np.newaxis],
#                            model.labels_[:,np.newaxis]]),
#                 columns=['x0','x1','d0','d1','d2','class'])
# df
#
# def plot_clusters(model,data):
#     c0, c1, c2 = model.cluster_centers_
#     plt.scatter(data[model.labels_==0,0],X[model.labels_==0,1],s=20,marker='v',c='r')
#     plt.scatter(data[model.labels_==1,0],X[model.labels_==1,1],s=20,marker='^',c='b')
#     plt.scatter(X[model.labels_==2,0],X[model.labels_==2,1],s=20,marker='v',c='y')
#     plt.scatter(c0[0], c0[1],s=40,c='r')
#     plt.scatter(c1[0], c1[1],s=40,c='b')
#     plt.scatter(c2[0], c2[1],s=40,c='y')
#
# plt.figure(figsize=(10,10))
# model1= KMeans(n_clusters=3,init='random',n_init=1,max_iter=1,random_state=1)
# model1.fit(X)
# plt.subplot(3,2,1)
# plot_clusters(model1, X)
#
# model2= KMeans(n_clusters=3,init='random',n_init=1,max_iter=2,random_state=1)
# model2.fit(X)
# plt.subplot(3,2,2)
# plot_clusters(model2, X)
#
# model3= KMeans(n_clusters=3,init='random',n_init=1,max_iter=3,random_state=1)
# model3.fit(X)
# plt.subplot(3,2,3)
# plot_clusters(model3, X)
#
# model4= KMeans(n_clusters=3,init='random',n_init=1,max_iter=4,random_state=1)
# model4.fit(X)
# plt.subplot(3,2,4)
# plot_clusters(model4, X)
#
# model5= KMeans(n_clusters=3,init='random',n_init=1,max_iter=5,random_state=1)
# model5.fit(X)
# plt.subplot(3,2,5)
# plot_clusters(model5, X)
#
# model6= KMeans(n_clusters=3,init='random',n_init=1,max_iter=6,random_state=1)
# model6.fit(X)
# plt.subplot(3,2,6)
# plot_clusters(model6, X)
# -----------------------------------//

# //-----------------------------------
# x=[32,64,96,118,126,144,152,158]
# y=[18,24,61.5,49,52,105,130,125]
#
# from scipy import stats
# slope, intercept, r_value, p_value, std_err=stats.linregress(x,y)
# print("slope: {}\nintercept: {}\nr_value: {}\np_value: {}\nstd_err: {}"\
#       .format(slope,intercept,r_value,p_value,std_err))
#
# import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline
#
# plt.scatter(x,y)
# plt.plot(x, slope*np.array(x)+intercept,'-')
# plt.show()
# -----------------------------------//

# //-----------------------------------
# from sklearn.datasets import load_boston
# boston=load_boston()
# X=boston.data
# y=boston.target
#
# from sklearn.model_selection import train_test_split
# X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
#
# from sklearn.preprocessing import MinMaxScaler
# scaler =MinMaxScaler(feature_range=(0,1))
# X_train_scaled=scaler.fit_transform(X_train)
# X_test_scaled=scaler.fit_transform(X_test)
#
# from sklearn.linear_model import LinearRegression
# model_boston=LinearRegression().fit(X_train_scaled, y_train)
# model_boston
#
# model_boston.score(X_train_scaled, y_train)
# from sklearn.model_selection import cross_val_score
# r2=cross_val_score(model_boston,X_train_scaled,y_train,cv=5)
# r2
#
# r2.mean()
#
# import math
# from sklearn.metrics import mean_squared_error
# math.sqrt(mean_squared_error(y_train,model_boston.predict(X_train_scaled)))
#
# y_pred=model_boston.predict(X_test_scaled)
# y_pred
#
# import math
# from sklearn.metrics import mean_squared_error
#
# rmse_test=math.sqrt(mean_squared_error(y_test,y_pred))
# rmse_test
#
# from sklearn.metrics import r2_score
# r2_score(y_test, y_pred)
# -----------------------------------//

# //-----------------------------------
# import statsmodels.api as sm
# Boston=sm.datasets.get_rdataset('Boston',package='MASS')
# boston_df=Boston.data
# boston_df.head()
#
# import statsmodels.formula.api as smf
# formula='medv~'+'+'.join(boston_df.iloc[:,:-1].columns)
# model_boston=smf.ols(formula=formula,data=boston_df).fit()
# model_boston.summary()
#
# import statsmodels.formula.api as smf
# formula='medv ~ rad + zn + rm + chas + age -1'
# model_boston2=smf.ols(formula=formula,data=boston_df).fit()
# model_boston2.summary()
#
# y_pred=model_boston2.predict(boston_df)
#
# import matplotlib.pyplot as plt
# %matplotlib inline
# fig=plt.figure()
# plt.scatter(boston_df.iloc[:,-1], y_pred)
# plt.xlabel('Target y')
# plt.ylabel('Predicte y')
# plt.title('Prediction vs. Actual')
# plt.show()
#
# from scipy import stats
# import matplotlib.pyplot as plt
# %matplotlib inline
#
# fig=plt.figure()
# res=stats.probplot(y_pred,plot=plt)
# plt.title('Probability plot')
# plt.show()
#
#
# import statsmodels.api as sm
# Boston=sm.datasets.get_rdataset('Boston', package='MASS')
# boston_off=Boston.data
#
# formula='medv~'+'+'.join(boston_df.iloc[:,:-1].columns)
#
# from patsy import dmatrices
# y, X=dmatrices(formula,boston_df,return_type='dataframe')
#
# import pandas as pd
# vif=pd.DataFrame()
#
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# vif['VIF Factor']=[variance_inflation_factor(X.values,i)
#                    for i in range(X.shape[1])]
# vif['features']=X.columns
# vif
# -----------------------------------//

# //-----------------------------------
# from sklearn import datasets
# iris=datasets.load_iris()
# print(iris.DESCR)
#
# from sklearn.datasets import load_wine
# import pandas as pd
# wine =load_wine()
# print(wine.DESCR)
#
# from sklearn.datasets import load_breast_cancer
# cancer =load_breast_cancer()
# print(cancer.DESCR)
#
# from sklearn.datasets import load_digits
# digits =load_digits()
# print(digits.DESCR)
# -----------------------------------//

# //-----------------------------------
# from sklearn.datasets import make_classification
# X, y=make_classification(n_features=1,n_redundant=0,n_informative=1,n_clusters_per_class=1,random_state=1)
#
# from sklearn.linear_model import LogisticRegression
# model=LogisticRegression().fit(X,y)
#
# import numpy as np
# xx=np.linspace(-3,3,100)
# XX=xx[:,np.newaxis]
# prob=model.predict_proba(XX)[:,1]
#
# import matplotlib.pyplot as plt
# %matplotlib inline
# x_test=[[-0.2]]
# plt.subplot(211)
# plt.plot(xx, prob)
# plt.scatter(X,y,marker='o',c=y,s=100,edgecolors='k',linewidths=2)
# plt.scatter(x_test[0],model.predict_proba(x_test)[0][1:], marker='x',s=200,c='r',lw=5)
# plt.xlim(-3,3)
# plt.ylim(-.2,1.2)
# plt.legend(['$P(y=1|x_{test})$'])
# plt.subplot(212)
# plt.bar(model.classes_,model.predict_proba(x_test)[0])
# plt.xlim(-1,2)
# plt.gca().xaxis.grid(False)
# plt.xticks(model.classes_,['$P(y=0|x_{test})$', '$P(y=1|x_{test})$'])
# plt.title('Conditional probability distribution')
# plt.tight_layout()
# plt.show()
# -----------------------------------//

from sklearn.datasets import load_iris
import numpy as np
iris=load_iris()
idx=np.in1d(iris.target,[0,2])
X=iris.data[idx,0:2]
y=iris.target[idx]

from sklearn.linear_model import Perceptron
model=Perceptron(max_iter=100,eta0=0.1, random_state=1).fit(X,y)

import matplotlib.pyplot as plt
XX_min, XX_max=X[:,0].min()-1, X[:,0].max()+1
YY_min, YY_max=X[:,1].min()-1, X[:,1].max()+1
XX, YY=np.meshgrid(np.linspace(XX_min,XX_max,1000),np.linspace(YY_min,YY_max,1000))
ZZ=model.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
plt.contour(XX,YY,ZZ,colors='k')
plt.scatter(X[:,0],X[:,1],c=y,s=30,edgecolors='k',linewidths=1)

idx=[22,36,70,80]
plt.scatter(X[idx,0],X[idx,1],c='r',s=100,alpha=0.5)
for i in idx:
    plt.annotate(i, xy=(X[i,0],X[i,1] + 0.1))
plt.grid(False)
plt.title('Perceptron discriminant area')
plt.xlabel('x1')
plt.xlabel('x2')
plt.show()
















