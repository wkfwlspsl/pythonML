import pandas as pd
import numpy as np

# 1. winequality 데이터 불러오기
redwine=pd.read_csv('winequality-red.csv',delimiter=';')
print(redwine)

# 2. 데이터셋 샘플링
train = redwine.sample(frac=0.7)
test = redwine.loc[~redwine.index.isin(train.index)]
print(train.head())
print(train.shape, test.shape)

train_x = train.iloc[:,:-1]
train_y = train.iloc[:,-1]
test_x = test.iloc[:,:-1]
test_y = test.iloc[:,-1]

# 3. 분류모델 생성
from sklearn.neural_network import MLPClassifier

mlp=MLPClassifier(hidden_layer_sizes=(50,30))
mlp.fit(train_x, train_y)

print('Training score: %s' % mlp.score(train_x, train_y))

# 4. 모델 평가
import pandas as pd
pred=mlp.predict(test_x)
confusion_matrix=pd.crosstab(test_y, pred,
                             rownames=['True'],
                             colnames=['Predicted'], margins=True)
print(type(confusion_matrix))
print(confusion_matrix)

print(confusion_matrix.iloc[2:5, 0:3])

# 혼동행렬을 이용하여 정확도(accuracy) 계산
cm = confusion_matrix.as_matrix()
print(cm)
cm_row = confusion_matrix.shape[0]
cm_col = confusion_matrix.shape[1]
print(cm_row, cm_col)

accuracy = (cm[2][0] + cm[3][1] + cm[4][2]) / float(cm[cm_row-1][cm_col-1])
print(accuracy)