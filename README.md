# Eigen-Portfolio
Automatized unsupervised machine learning Principal Component Analysis (PCA) on the Dow Jones Industrial Average index and it's respective 30 stocks to construct an optimized diversified intelligent portfolio. 

Combines unsupervised deep learning to classify a crisis.

Consists of 4 python files that make up the machine learning pipeline:

1) tiingoconnect.py: creates a function to extract historical financial data from the financial API Tiingo. 
2) getdata.py: uses three functions to extract the tickers from the components of the DJIA, downloads historical OHLC values for each ticker and the DJIA Index into individual csv batches, and compiles all the adjusted closing for every ticker into the main csv file to use it as data for ml.py.
