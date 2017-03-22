import re
import requests
import requests_cache
import pandas as pd
from bs4 import BeautifulSoup
requests_cache.install_cache('bball_ref_cache')
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import seaborn as sns


def bball_scraper(stat_type, start_year, end_year):
    bball_df = pd.DataFrame()
    url = 'http://www.basketball-reference.com/leagues/NBA_'
    for i in range(end_year-start_year + 1):
            url2 = url + r'%s_' %start_year + r'%s.html' %stat_type
            year = BeautifulSoup(requests.get(url2).content, "lxml")
            columns = year.find('thead').text.split('\n')
            stats = year.find('tbody')
            for figure in stats.find_all('tr', 'thead'):
                figure.decompose()
            data = stats.find_all('tr')
            player = [[td.getText() for td in data[i].findAll('td')] for i in range(len(data))]
            temp = pd.DataFrame(player)
            for index, row in temp.iterrows():
                if row[3] == 'TOT':
                    pname = row[0] 
                    temp = temp[(temp[0] != pname) | (temp[3] == 'TOT')]
            temp['Year'] = start_year
            bball_df = bball_df.append(temp)
            start_year = start_year + 1 
    columns = columns[3:-2]
    columns.append('Year')
    bball_df.columns = columns
    return bball_df

bball_per_game = bball_scraper('per_game', 2001, 2017)
bball_advanced = bball_scraper('advanced', 2001, 2017)
bball_100poss = bball_scraper('per_poss', 2001, 2017)
bball_per_game['MPT'] = bball_advanced['MP']

def kmeans(data): 
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=10, n_init=20)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the NBA Dataset')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    return reduced_data
    
bball_per_game2 = bball_per_game.merge(bball_advanced)
bball_per_game2 = bball_per_game2.apply(lambda x: pd.to_numeric(x, errors='ignore'))
bball_per_game2 = bball_per_game2.loc[bball_per_game2['MP'] > 1]
bball_per_game2.fillna(0, inplace=True)
bball_per_game2 = bball_per_game2.set_index(bball_per_game2['Player'])
bball_per_game2 = bball_per_game2.drop(['Player','Pos','Age','Tm','G','GS','MP', 'PER',
                                        'PF','FGA','FG','FG%', '3P','2P','FT', 'TRB',
                                        'eFG%', 'VORP', 'BPM', 'DBPM', 'OBPM', 'USG%',
                                        'OWS', 'DWS', 'WS', 'WS/48', 'Year','TRB%','FTr',
                                        'TS%', '3PAr'], axis = 1)
bball_per_game2 = bball_per_game2.drop(bball_per_game2[[-1,-2]], axis = 1)

df = bball_per_game2
from scipy.spatial.distance import pdist, squareform

dist = pdist(df, 'euclidean')
df_dist = pd.DataFrame(squareform(dist))


