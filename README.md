# Eigen-Portfolios
Automatized unsupervised machine learning Principal Component Analysis (PCA) on the Dow Jones Industrial Average index and it's respective 30 stocks to construct an optimized diversified intelligent portfolio. 

### Files
1) tiingoconnect.py: creates a function to extract historical financial data from the financial API Tiingo. 
2) getdata.py: uses three functions to extract the tickers from the components of the DJIA, downloads historical OHLC values for each ticker and the DJIA Index into individual csv batches, and compiles all the adjusted closing for every ticker into the main csv file to use it as data for ml.py.
3) ml.py:

## Data
With the functions on getdata.py, we will be able to parse data with beautifulsoup4 through CNN Money to obtain the tickers of the Dow Jones Industrial Average Index. It will create a list of the 30 tickers and will pass the tickers as inputs for the financial API to download historical values.

Using Tiingo API, which gives you access to countless of historical information on stocks, we can download the necessary data for the Dow Jones Industrial Average. We download the tiingoconnect.py file and import it in the getdata.py file to download all the required information.

After downloading the information, we will have a pandas dataframe with basic information for every ticker (Open, High, Low, Close, Volume, and Adjusted values). With this we will choose all the adjusted closing values of each ticker and we will concatenate them into a big dataframe which we will use as our starting dataset for the ml.py file.

## Machine Learning


