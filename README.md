# Eigen-Portfolios
Automatized unsupervised machine learning Principal Component Analysis (PCA) on the Dow Jones Industrial Average index and it's respective 30 stocks to construct an optimized diversified intelligent portfolio. 

### Files
1) tiingoconnect.py: creates a function to extract historical financial data from the financial API Tiingo. 
2) getdata.py: uses three functions to extract the tickers from the components of the DJIA, downloads historical OHLC values for each ticker and the DJIA Index into individual csv batches, and compiles all the adjusted closing price for every ticker into the main csv file to use it as data for ml.py.
3) ml.py: takes as input the dataset obtained from getdata.py and computes logarithmic average and basic average returns for each company in the DJIA as well as for the index. It then creates a covariance matrix with every ticker and trains it with Principle Component Analysis to extract its principle components. With this, we can then compute the principle components to create Eigen Portfolios. We then pass a function to compute geometric average returns, valatility, and the sharpe ratio to output an optimized portfolio with the highest sharpe ratio (Highest average return and lowest volatility).
4) ml.ipnb: Jupyter Notebook implementation of ml.py to visualize outputs in order.
5) data.csv: dataset containing every ticker's adjusted closing price. (You can skip the steps of gathering data).

## Data
With the functions on getdata.py, we will be able to parse data with beautifulsoup4 through CNN Money to obtain the tickers of the Dow Jones Industrial Average Index. It will create a list of the 30 tickers and will pass the tickers as inputs for the financial API to download historical values.

Using Tiingo API, which gives you access to countless of historical information on stocks, we can download the necessary data for the Dow Jones Industrial Average. We download the tiingoconnect.py file and import it in the getdata.py file to download all the required information.

After downloading the information, we will have a pandas dataframe with basic information for every ticker (Open, High, Low, Close, Volume, and Adjusted values). With this we will choose all the adjusted closing values of each ticker and we will concatenate them into a big dataframe which we will use as our starting dataset for the ml.py file.

## Machine Learning
We import the pandas dataframe containing all the adjusted closing prices for all the companies in the DJIA as well as the DJIA's index. Because our starting date was the year 2000, the adjusted closing prices for Dow Chemicals and Visa appear as Not a Number values. For this we will take away both respective columns. We will end up with 28 columns of companies information and an additional one for the DJIA index.

### Logarithmic Returns
The price for an asset i at a given time t is pi,t and the logarithmic returns for each stock i will be written as:
ri,t = ln * (pi,t)/pi,t−1.

### Linear Returns
The price for an asset i at a given time t is pi,t and the linear returns for each stock i will be written as:
ri,t=(pi,t−pi,t−1)/pi,t−1.

### Normalized Returns
In order to create a covariance matrix that can be applied for the Principle Component Analysis, we will have to normalize the returns we choose to work with.
To normalize we substract the mean of each return and divide the residual by the return's standard deviation.

### Data Preprocessing
1) We preprocess the data by getting rid of all the Not a Number rows in the returns dataframe.
2) We split the training and testing sets with a predetermined percentage (in this case 80%, but can be modified at will).
3) We then create the covariance matrix of all the companies exclusing the column of the DJIA Index by computing the pandas .cov() function. With this we will have a 28 x 28 matrix.

### Principle Component Analysis
We create a function to compute Principle Component Analysis from Sklearn with a variance explained treshold of 95%.
This function computes an inversed elbow chart that shows the amount of principle components and how many of them explain the variance treshold. 

### Eigen Portfolios
We compute several functions to determine the weights of each principle component. We then visualize a scatterplot that visualizes an organized descending plot with the respective weight of every company at the current chosen principle component. 

### Sharpe Ratio
The sharpe ratio explains the annualized returns against the annualized volatility of each company in a portfolio. 
A high sharpe ratio explains higher returns and lower volatility for the specified portfolio.

1) Annualized Returns: We have to apply the geometric average of all the returns in respect to the periods per year (days of operations in the exchange in a year). 
2) Annualized Volatility: We have to take the standard deviation of the returns and multiply it by the square root of the periods per year. 
3) Annualized Sharpe: we compute the ratio by dividing the annualized returns against the annualized volatility. 

### Optimized Portfolio
We compute an iterable loop to compute the principle component's weights for each Eigen Portfolio, which then uses the sharpe ratio function to look for the portfolio with the highest sharpe ratio. Once we know which portfolio has the highest sharpe ratio, we can visualize its performance against the DJIA Index to understand how it outperforms it. 

## Acknowledgements
This project was inspired by the lectures on Portfolio Optimization on the course of Machine Learning in Finance by NYU Tandon School of Engineering. The theoretical basis and through implementation was modeled after Avellaneda & Lee (2008) "Statistical Arbitrage in the U.S. Equities Market."
