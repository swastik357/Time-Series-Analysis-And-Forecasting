In the CSV files, for NON-TRADING days:

(a) Prices and Volumes were originally NAN
(b) We FRONT FILLED prices
(c) We set Volumes to 0

I made some changes to original code because the format of column names in data downloaded from yahoo finance had changeds

data = yf.download('SPY', start="2018-01-01", end="2021-04-17")
 
Then, print(data.columns) gives:
 
MultiIndex([( 'Close', 'SPY'),
            (  'High', 'SPY'),
            (   'Low', 'SPY'),
            (  'Open', 'SPY'),
            ('Volume', 'SPY')],
           names=['Price', 'Ticker'])
 
As a result: df = df.join(data, how='outer') gives error MergeError: Not allowed to merge between different levels. (1 levels on the left, 2 on the right)
 
To resolve this, I did following pre-processing on data:
 
data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

Now print(data.columns) gives:
 
Index(['Close', 'High', 'Low', 'Open', 'Volume'], dtype='object')
 
Now df = df.join(data, how='outer') works fine.
 
df.head() :
 
	Close	High	Low	Open	Volume
2018-01-02	237.909164	237.944579	236.696475	237.085956	86655700.0
2018-01-03	239.414017	239.564509	238.077390	238.077390	90070400.0
2018-01-04	240.423019	240.909883	239.475900	240.060120	80636400.0
2018-01-05	242.025238	242.149149	240.724026	241.219723	83524000.0
2018-01-06	NaN	NaN	NaN	NaN	NaN

In related_schema, "AttributeName":"open_value" instead of "open" because open is a RESERVED KEYWORD on Amazon forecast.
      
