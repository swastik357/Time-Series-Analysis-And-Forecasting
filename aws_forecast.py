import pandas as pd
import yfinance as yf
import boto3
from datetime import datetime

# Get the Data

data = yf.download('SPY', start="2018-01-01", end="2021-04-17")

print(data.head())

# add in missing dates
date_range = pd.date_range(data.index[0], data.index[-1])
date_range

# make a new dataframe containing all dates
df = pd.DataFrame(index=date_range)
print(df.head())

data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
df = df.join(data, how='outer')
print(df.head())

# fill in missing data
df[['Open', 'High', 'Low', 'Close']] = \
    df[['Open', 'High', 'Low', 'Close']].fillna(method='ffill')
df['Volume'] = df['Volume'].fillna(0)
print(df.head(10))

# save it just in case it's needed later
df.to_csv("/Users/swastik./Downloads/simulationoutput/daily_price_full.csv")

# check format
testdf = pd.read_csv("/Users/swastik./Downloads/simulationoutput/daily_price_full.csv")
print(testdf.head(5))
# head daily_price_full.csv

# AWS Forecast requires a column called item_id
df['item_id'] = 'SPY'

# leave the last 30 points for forecast comparison
FORECAST_LENGTH = 30
train = df.iloc[:-FORECAST_LENGTH]

# AWS differentiates between "target time series" and "related time series"
train_target_series = train[['Close', 'item_id']]
train_related_series = train[['Open', 'High', 'Low', 'Volume', 'item_id']]

# Save the data which we will upload to S3 later
train_target_series.to_csv("/Users/swastik./Downloads/simulationoutput/daily_price_target_series.csv", header=None)
train_related_series.to_csv("/Users/swastik./Downloads/simulationoutput/daily_price_related_series.csv", header=None)

testdf = pd.read_csv("/Users/swastik./Downloads/simulationoutput/daily_price_target_series.csv")
print(testdf.head(5))

testdf = pd.read_csv("/Users/swastik./Downloads/simulationoutput/daily_price_related_series.csv")
print(testdf.head(5))
# Check format
# head -n 5 daily_price_target_series.csv

# # Check format
# head -n 5 daily_price_related_series.csv

# Create Dataset Group and Upload Data

# bucket names must be unique!
# you cannot use the same bucket as me
bucket_name = 'xxx'
region = 'yyy'
# get this from your AWS console
role_arn = 'zzz'

# Create S3 client
s3 = boto3.client('s3', region_name=region)

DATASET_FREQUENCY = "D"
TIMESTAMP_FORMAT = "yyyy-MM-dd"
dataset_group = "daily_forecast_dataset_group"

# create boto3 clients
forecast_client = boto3.client(service_name='forecast', region_name=region) 
forecastquery_client = boto3.client(service_name='forecastquery', region_name=region)

# create a dataset group
create_dataset_group_response = forecast_client.create_dataset_group(
  Domain="CUSTOM",
  DatasetGroupName=dataset_group
)

######################################################################################################################################################################################################

# wait until it's complete!
dataset_group_arn = create_dataset_group_response['DatasetGroupArn']
describe = forecast_client.describe_dataset_group(DatasetGroupArn=dataset_group_arn)
print(describe['Status'])
print(describe['CreationTime'])
print(describe['LastModificationTime'])
# Schema for target dataset
# Must match the columns of the CSV
target_schema = {
  "Attributes": [
    {
      "AttributeName":"timestamp",
      "AttributeType":"timestamp"
    },
    {
      "AttributeName":"target_value",
      "AttributeType":"float"
    },
    {
      "AttributeName":"item_id",
      "AttributeType":"string"
    }
  ]
}
# Give your dataset a name
target_dataset_name = "close_prices"
# Create a dataset
r = forecast_client.create_dataset(
  Domain="CUSTOM",
  DatasetType='TARGET_TIME_SERIES',
  DatasetName=target_dataset_name,
  DataFrequency=DATASET_FREQUENCY,
  Schema=target_schema)
# Check the response
target_dataset_arn = r['DatasetArn']
describe = forecast_client.describe_dataset(DatasetArn=target_dataset_arn)
print(describe['Status'])
print(describe['CreationTime'])
print(describe['LastModificationTime'])
# Schema for related dataset
# Must match the columns of the CSV
# OPTIONAL - you can do this with the target time series only
related_schema = {
  "Attributes": [
    {
      "AttributeName":"timestamp",
      "AttributeType":"timestamp"
    },
    {
      "AttributeName":"open_value",
      "AttributeType":"float"
    },
    {
      "AttributeName":"high_value",
      "AttributeType":"float"
    },
    {
      "AttributeName":"low_value",
      "AttributeType":"float"
    },
    {
      "AttributeName":"volume_value",
      "AttributeType":"float"
    },
    {
      "AttributeName":"item_id",
      "AttributeType":"string"
    }
  ]
}
# Give your dataset a name
related_dataset_name = "related_data"
# Create a dataset
r = forecast_client.create_dataset(
  Domain="CUSTOM",
  DatasetType='RELATED_TIME_SERIES',
  DatasetName=related_dataset_name,
  DataFrequency=DATASET_FREQUENCY,
  Schema=related_schema)
# Check the response
related_dataset_arn = r['DatasetArn']
describe = forecast_client.describe_dataset(DatasetArn=related_dataset_arn)
print(describe['Status'])
print(describe['CreationTime'])
print(describe['LastModificationTime'])
# Add your datasets to the dataset group
dataset_arns = [target_dataset_arn, related_dataset_arn]
forecast_client.update_dataset_group(
  DatasetGroupArn=dataset_group_arn,
  DatasetArns=dataset_arns
)
# Upload data to S3
s3r = boto3.resource('s3', region_name=region)
s3r.Bucket(bucket_name).Object(
    "daily_price_target_series.csv").upload_file("daily_price_target_series.csv")
s3r.Bucket(bucket_name).Object(
    "daily_price_related_series.csv").upload_file("daily_price_related_series.csv")
# Path to your data
s3_target_path = f"s3://{bucket_name}/daily_price_target_series.csv"
s3_related_path = f"s3://{bucket_name}/daily_price_related_series.csv"
# Launch an import job
target_import_job_response = forecast_client.create_dataset_import_job(
  DatasetImportJobName=dataset_group,
  DatasetArn=target_dataset_arn,
  DataSource= {
    "S3Config" : {
      "Path": s3_target_path,
      "RoleArn": role_arn
    } 
  },
  TimestampFormat=TIMESTAMP_FORMAT)
target_import_job_arn = target_import_job_response['DatasetImportJobArn']

# check if it's done - takes a few mins
describe = forecast_client.describe_dataset_import_job(DatasetImportJobArn=target_import_job_arn)
print(describe['Status'])
print(describe['CreationTime'])
print(describe['LastModificationTime'])
# Launch an import job for related dataset - don't have to wait for previous job
related_import_job_response = forecast_client.create_dataset_import_job(
  DatasetImportJobName=dataset_group,
  DatasetArn=related_dataset_arn,
  DataSource= {
    "S3Config" : {
      "Path": s3_related_path,
      "RoleArn": role_arn
    } 
  },
  TimestampFormat=TIMESTAMP_FORMAT)
related_import_job_arn = related_import_job_response['DatasetImportJobArn']

# check if it's done - takes a few mins
describe = forecast_client.describe_dataset_import_job(DatasetImportJobArn=related_import_job_arn)
print(describe['Status'])
print(describe['CreationTime'])
print(describe['LastModificationTime'])
# Make Predictor
algorithm_arn = 'arn:aws:forecast:::algorithm/Deep_AR_Plus'
predictor_name = "deep_ar_plus_predictor"
# By default, will return [0.1, 0.5, 0.9]
create_predictor_response = forecast_client.create_predictor(
  PredictorName=predictor_name,
  AlgorithmArn=algorithm_arn,
  ForecastHorizon=FORECAST_LENGTH,
  PerformAutoML=False,
  PerformHPO=False,
#   ForecastTypes=["0.10", "0.50", "0.9", "mean"],
  InputDataConfig={"DatasetGroupArn": dataset_group_arn},
  FeaturizationConfig={"ForecastFrequency": DATASET_FREQUENCY},
)
predictor_arn = create_predictor_response['PredictorArn']
# Wait for 'active' - may take a few hours
describe = forecast_client.describe_predictor(PredictorArn=predictor_arn)
print(describe['Status'])
print(describe['CreationTime'])
print(describe['LastModificationTime'])
# get accuracy metrics
forecast_client.get_accuracy_metrics(PredictorArn=predictor_arn)
# Generate Forecast
forecast_name = "deep_ar_plus_forecast"
create_forecast_response = forecast_client.create_forecast(
    ForecastName=forecast_name,
    PredictorArn=predictor_arn)
forecast_arn = create_forecast_response['ForecastArn']
# takes some time to become active
describe = forecast_client.describe_forecast(ForecastArn=forecast_arn)
print(describe['Status'])
print(describe['CreationTime'])
print(describe['LastModificationTime'])
# Check Forecast
def parse_aws_forecast(d10, d50, d90):
    ts = pd.Timestamp(d10['Timestamp'])
    val1 = d10['Value']
    val2 = d50['Value']
    val3 = d90['Value']
    return ts, val1, val2, val3
forecast_response = forecastquery_client.query_forecast(
    ForecastArn=forecast_arn,
    Filters={"item_id": 'SPY'})
# What's in it?
forecast_response
p10 = forecast_response['Forecast']['Predictions']['p10']
p50 = forecast_response['Forecast']['Predictions']['p50']
p90 = forecast_response['Forecast']['Predictions']['p90']

parsed = [parse_aws_forecast(d1, d2, d3) for d1, d2, d3 in zip(p10, p50, p90)]
forecast_df = pd.DataFrame(parsed, columns=['timestamp', 'p10', 'p50', 'p90'])
forecast_df.set_index('timestamp', inplace=True)
true_df = df[['Close']].copy()
true_df.columns = ['true']

full_df = true_df.join(forecast_df, how='outer')
full_df[['true', 'p10', 'p50', 'p90']].plot(figsize=(20, 10));
full_df.iloc[-100:][['true', 'p10', 'p50', 'p90']].plot(figsize=(20, 10));
