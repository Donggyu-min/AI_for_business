'''
1.	Use iris dataset that I introduced in class.
2.	Run linear regression with sklearn and statsmodel.
3.	Dependent variable: petal length
4.	Independent variables: petal width, sepal length and width.
5.	Submit a code file and a word file showing the coefficients from sklearn and the regression output table from statsmodels. Yes, the coefficients from both libraries must be the same.
'''

import pandas as pd
import sklearn.linear_model as sklm
import statsmodels.formula.api as smf
from sklearn import datasets

iris = datasets.load_iris()
# print(iris.DESCR)
x = iris.data
y = iris.target
# print(type(x), x.shape)
# print(type(y), y.shape)

df = pd.DataFrame(x, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
x_df = df[['sepal_width', 'sepal_length', 'petal_width']]
y_df = df[['petal_length']]
# print(df)

#ols with sklearn
sk_ols = sklm.LinearRegression()
sk_ols.fit(x_df, y_df)
print('\ncoefficient : ', sk_ols.coef_)
print('intercept : ', sk_ols.intercept_)
print('R-squared : ', sk_ols.score(x_df, y_df), '\n')

#ols with statsmodels
sm_ols = smf.ols('petal_length ~ sepal_width + sepal_length + petal_width', data=df).fit()
print(sm_ols.summary())
