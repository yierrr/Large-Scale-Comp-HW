import boto3
import json
import random
import datetime

kinesis = boto3.client('kinesis', region_name='us-east-1')

def getReferrer():
    data = {}
    now = datetime.datetime.now()
    str_now = now.isoformat()
    data['EVENT_TIME'] = str_now
    data['TICKER'] = 'AAPL'
    price = random.random() * 100 
    data['PRICE'] = round(price, 2)
    return data

while True:
    data = getReferrer()
    kinesis.put_record(StreamName="stockstream",
                       Data=json.dumps(data),
                       PartitionKey="partitionkey")
