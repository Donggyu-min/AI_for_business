import pandas as pd
import numpy as np
import mglearn
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


'''
1.	Use the Excel file data in A4data.xlsx, which has three variables, Country, Year, and Bbadopt. Bbadopt shows mobile broadband adoption rates of each country for four years.
2.	Read the data in Python and do the jobs below.
    A.	Calculate the average mobile broadband adoption rate for each country and put the average in Bbavg variable.
    B.	Calculate the speed of mobile broadband adoption for four years by subtraction 2013 data from 2016 data for each country and put the speed in Bbspeed variable.
    C.	Cluster all countries into groups with Bbavg and Bbspeed.
    D.	Cluster again all countries and this time use 2013 mobile broadband adoption rate instead of Bbavg.
    E.	Briefly explain the clustering results (in Korean). Submit your explanation with clustering figures and an ipynb file.
'''


def preprocessing(x):
    '''
    ***country별 평균값(avg_adopt), 속도(spd_adopt)구하기***
    1. groupby를 통해 x에 접근(x 개수 = country 개수)
    2. x는 row가 4개(2013, 2014, 2015, 2016)인 dataframe
    3. mean()함수를 통해 avg_adopt계산
    4. x[x['Year']==n]을 통해 각 2013/16년의 adopt데이터를 row 1개짜리 series로 추출
    5. type이 series니까 reset_index한 뒤 [0]으로 값에 접근
    * Bbadopt이 감소할 수도 있으니 max-min을 사용하지 않았음
    '''
    column_list = {}
    column_list['avg_adopt'] = x['Bbadopt'].mean()

    adopt_16 = x[x['Year'] == 2016]['Bbadopt'].reset_index(drop=True)[0]
    adopt_13 = x[x['Year'] == 2013]['Bbadopt'].reset_index(drop=True)[0]
    column_list['spd_adopt'] = adopt_16 - adopt_13
    column_list['adopt_13'] = adopt_13
    df = pd.Series(column_list, index=['avg_adopt', 'adopt_13', 'spd_adopt'])
    return df


def k_means(df):
    plt.clf()
    np_df = df.to_numpy()
    if df.columns.tolist()[0] == 'adopt_13':
        clusters = 4
    else:
        clusters = 3

    kmeans = KMeans(n_clusters = clusters).fit(df)
    print(kmeans.cluster_centers_)
    y = kmeans.labels_
    x = np_df
    rgb = np.array(['r', 'g', 'b', 'y'])

    plt.xlabel(df.columns.tolist()[0])
    plt.ylabel(df.columns.tolist()[1])
    plt.scatter(x[:,0], x[:,1], color = rgb[y])
    # plt.show()
    plt.savefig('figure/km_' + str(df.columns.tolist()[0]) + '.png')


def km_test(df):
    plt.clf()
    cost = []
    for i in range(1, 11):
        KM = KMeans(n_clusters = i, max_iter = 500)
        KM.fit(df)
        cost.append(KM.inertia_)
    plt.plot(range(1,11), cost, color = 'g', linewidth='3')
    plt.xlabel('# of clusters')
    plt.ylabel('cluster inertia')
    # plt.show()
    plt.savefig('figure/kmtest_' + str(df.columns.tolist()[0]) + '.png')
    #4 seems to be enough


def dbscan(df):
    plt.clf()
    np_df = df.to_numpy()
    dbscan = DBSCAN(eps=14, min_samples=3)
    clusters = dbscan.fit_predict(np_df)
    mglearn.discrete_scatter(np_df[:,0], np_df[:,1], clusters)

    plt.xlabel(df.columns.tolist()[0])
    plt.ylabel(df.columns.tolist()[1])
    # plt.show()
    plt.savefig('figure/dbscan_' + str(df.columns.tolist()[0]) + '.png')

if __name__ == '__main__':
    df = pd.read_excel('data/A4data.xlsx')
    df = df.groupby('Country').apply(preprocessing)
    df_avg = df[['avg_adopt', 'spd_adopt']]
    df_13 = df[['adopt_13', 'spd_adopt']]

    print(df)
    # print(df['spd_adopt'].max(axis=0), df['avg_adopt'].max(axis=0))

    km_test(df_avg)
    km_test(df_13)
    k_means(df_avg)
    k_means(df_13)
    dbscan(df_avg)
    dbscan(df_13)
