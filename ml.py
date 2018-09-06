import math
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from scipy.stats import randint as sp_randint

from sklearn.decomposition import PCA

style.use("ggplot")

# loading the data
df = pd.read_csv('DJIA_adjcloses.csv', parse_dates=True, index_col=0)

# Visualizing the dataframe
# print(data.head())

# Dropping 'Not a Number' columns for Dow Chemicals (DWDP) and Visa (V)
df.drop(['DWDP', 'V'], axis=1, inplace=True)

# Copying the dataframe to add features
data = pd.DataFrame(df.copy())

# Daily Log Returns (%)
datareturns = np.log(data / data.shift(1)) * 100

# Data Raw
data_raw = datareturns

# Normalizing the Log returns
data = (datareturns - datareturns.mean()) / datareturns.std()

# Getting rid of the first row with NaN values.
data.dropna(how='all', inplace=True)

# take care of inf/NaN values by assigning them high values:
# *the algorithm won't take these outliers into consideration*
# data.fillna(999999, inplace=True)

# Getting rid of the NaN values.
data.dropna(how='any', inplace=True)

# Visualizing Log Returns for the DJIA 
# plt.figure(figsize=(16, 5))
# plt.title("Dow Jones Industrial Average Log Returns (%)")
# data.DJIA.plot()
# plt.grid(True);
# plt.legend()
# plt.show()

# Taking away the market benchmark DJIA
stock_tickers = data.drop(['DJIA'], axis=1)
stock_tickers_raw = data_raw.drop(['DJIA'], axis=1)
tickers = stock_tickers.columns.values
n_tickers = len(tickers)

# Dividing the dataset into training and testing sets
percentage = int(len(data) * 0.8)
X_train = stock_tickers[:percentage]
X_test = stock_tickers[percentage:]

X_train_raw = stock_tickers_raw[:percentage]
X_test_raw = stock_tickers_raw[percentage:]

# Applying Principle Component Analysis

# Creating covariance matrix and training data on PCA.
cov_matrix = X_train.loc[:,X_train.columns].cov()
pca = PCA()
pca.fit(cov_matrix)

def plotPCA(plot=False):

    # Visualizing Variance against number of principal components.
    cov_matrix_raw = X_train_raw.loc[:,X_train_raw.columns].cov()

    var_threshold = 0.95
    var_explained = np.cumsum(pca.explained_variance_ratio_)
    num_comp = np.where(np.logical_not(var_explained < var_threshold))[0][0] + 1  

    if plot:
        print('%d principal components explain %.2f%% of variance' %(num_comp, 100* var_threshold))

        # PCA percent variance explained.
        bar_width = 0.9
        n_asset = stock_tickers.shape[1]
        x_indx = np.arange(n_asset)
        fig, ax = plt.subplots()

        # Eigenvalues measured as percentage of explained variance.
        rects = ax.bar(x_indx, pca.explained_variance_ratio_[:n_asset], bar_width)
        ax.set_xticks(x_indx + bar_width / 2)
        ax.set_xticklabels(list(range(n_asset)), rotation=45)
        ax.set_title('Percent variance explained')
        ax.set_ylabel('Explained Variance')
        ax.set_xlabel('Principal Components')
        plt.show()

plotPCA(plot=False)

projected = pca.fit_transform(cov_matrix)
pcs = pca.components_

def PCWeights():
    '''
    Principal Components (PC) weights for each 28 PCs
    '''
    weights = pd.DataFrame()

    for i in range(len(pcs)):
        weights["weights_{}".format(i)] = pcs[:, i] / sum(pcs[:, i])

    weights = weights.values.T
    return weights

weights = PCWeights()
portfolio = portfolio = pd.DataFrame()

def plotEigenPortfolios(weights, plot=False, portfolio=portfolio):
    portfolio = pd.DataFrame(data ={'weights': weights.squeeze()*100}, index = stock_tickers.columns) 
    portfolio.sort_values(by=['weights'], ascending=False, inplace=True)
    
    if plot:
        print('Sum of weights of current eigen-portfolio: %.2f' % np.sum(portfolio))
        portfolio.plot(title='Current Eigen-Portfolio Weights', 
            figsize=(12,6), 
            xticks=range(0, len(stock_tickers.columns),1), 
            rot=45, 
            linewidth=3
            )
        plt.show()

    return portfolio

# Weights are stored in arrays, where 0 is the first PC's weights.
plotEigenPortfolios(weights=weights[0], plot=False)

# Sharpe Ratio

def sharpe_ratio(ts_returns, periods_per_year=252):
    '''
    Sharpe ratio is the average return earned in excess of the risk-free rate per unit of volatility or total risk.
    It calculares the annualized return, annualized volatility, and annualized sharpe ratio.
    
    ts_returns are  returns of a signle eigen portfolio.
    '''

    annualized_return = ts_returns.mean() * periods_per_year 
    annualized_vol = ts_returns.std() * np.sqrt(periods_per_year)
    annualized_sharpe = annualized_return / annualized_vol

    return annualized_return, annualized_vol, annualized_sharpe

def plotSharpe(eigen):

    '''
    Plots Principle components returns against real returns.
    '''

    eigen_portfolio_returns = np.dot(X_test_raw.loc[:, eigen.index], eigen / 100)
    eigen_portfolio_returns = pd.Series(eigen_portfolio_returns.squeeze(), index=X_test.index)
    returns, vol, sharpe = sharpe_ratio(eigen_portfolio_returns)
    print('Current Eigen-Portfolio:\nReturn = %.2f%%\nVolatility = %.2f%%\nSharpe = %.2f' % (returns*100, vol*100, sharpe))
    year_frac = (eigen_portfolio_returns.index[-1] - eigen_portfolio_returns.index[0]).days / 252

    df_plot = pd.DataFrame({'PC1': eigen_portfolio_returns, 'SPX': X_test_raw.loc[:, 'SPX']}, index=X_test.index)
    np.cumprod(df_plot + 1).plot(title='Returns of the market-cap weighted index vs. First eigen-portfolio', 
                             figsize=(12,6), linewidth=3)

# plotSharpe(eigen=plotEigenPortfolios(weights=weights[0], plot=False))




