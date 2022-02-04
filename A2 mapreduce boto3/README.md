# assignment-2-yierrr
assignment-2-yierrr created by GitHub Classroom

Q1a (see here: https://github.com/lsc4ss-a21/assignment-2-yierrr/blob/main/Q1.ipynb)
```
#Creating DynamoDB table outside Lambda

import boto3
client = boto3.client('dynamodb')

dynamodb = boto3.resource('dynamodb')
table = dynamodb.create_table(
    TableName='psychrec',
    KeySchema=[
        {
            'AttributeName': 'username',
            'KeyType': 'HASH'
        }
    ],
    AttributeDefinitions=[
        {
            'AttributeName': 'username',
            'AttributeType': 'S'#string
        }
    ],
    ProvisionedThroughput={ #baseline concurrent capacity read and write
        'ReadCapacityUnits': 1,
        'WriteCapacityUnits': 1
    }    
)
```
```
# Code in Lambda

import time
import boto3
import json


dynamodb = boto3.resource('dynamodb')
client = boto3.client('dynamodb')#table created in jupyter?
table= dynamodb.Table('psychrec') 

s3 = boto3.resource('s3') 
s3.create_bucket(Bucket='psychlab-14070714')

time.sleep(10)

def lambda_handler(event, context):
    #saving to s3
    time = event["timestamp"]
    uid = event["user_id"]
    moo = event['mood']# string? int?
    key = time + '|' + uid
    eve = json.dumps(event)
    s3.Bucket('psychlab-14070714').put_object(Key=key,Body = eve) 
    
    #analyse sentiment
    txt = event["text"]
    comprehend = boto3.client('comprehend')
    response = comprehend.detect_sentiment(Text=txt,
                                       LanguageCode='en')
    sen = response['Sentiment']

    #saving to DB
    response = table.get_item(
        Key={
            'username': uid
        }
        )
    if 'Item' not in response.keys():  
        table.put_item(
           Item={
                'username': uid,
                'sentiment': sen,
                'mood': moo,
                'num_survey':1
            }
        )
    else:
        num = int(response['Item']['num_survey']) + 1
        table.update_item(
                Key={
                    'username': uid
                },
                UpdateExpression='SET num_survey = :val1, sentiment = :val2, mood = :val3',
                ExpressionAttributeValues={
                    ':val1': num,
                    ':val2': sen,
                    ':val3': moo,
                }
            )
        
    response = table.get_item(
        Key={
            'username': uid
        }
        )
    senti = response['Item']['sentiment']
    return senti
```
```
# Deleting S3 items and DynamoDB table outside Lambda
import boto3

client = boto3.client('dynamodb')
table= dynamodb.Table('psychrec') 
s3 = boto3.resource('s3') 

def cleanup(bucket_name):
    bucket = s3.Bucket(bucket_name)
    for item in bucket.objects.all():
        item.delete()
        
cleanup('psychlab-14070714')
table.delete()
```

Q1b
> This solution mainly uses services DynamoDB and S3 buckets. Regarding DynamoDB, according to its documentation (https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Introduction.html), DynamoDB employs sufficient servers to handle throughputs and storage while maintaining consistency. Data is stored and replicated on availability zones in different places, thus further protecting data from damages of storage facilities. Especially with point-to-time recovery, any table can be restored within 35 days. Moreover, DynamoDB also ensures seamless scalability without downtime or degragation, suitable for study with a large amount of participants. In addition, as privacy is vital to psychological research, the encryption option within DynamoDB can protect the participants from privacy leakage.DynamoDB is also suitable for long-term storage of data, which can be used for future replication or further research purposes. Regarding S3, according to its documentation (https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html), S3 enables users to input and retrieve data of different time, without dealing with damaged or partial data, further ruling out data loss concerns. Similar to DynamoDB, S3 also replicates data across multiple servers to prevent data damage. Additionally, the eventual consistency of S3 buckets allows for time between the implementation and the realization of data changes, allowing users to make amendments when accidental changes are made. For example, if a bucket is accidentally deleted, it may still remain in the list for a while.

Q2a(see here:https://github.com/lsc4ss-a21/assignment-2-yierrr/blob/main/Q2.py)
```
'''
What's the most used word in 5-star customer reviews on Amazon?

We can answer this question using the mrjob package to investigate customer
reviews available as a part of Amazon's public S3 customer reviews dataset.

For this demo, we'll use a small sample of this 100m+ review dataset that
Amazon provides (s3://amazon-reviews-pds/tsv/sample_us.tsv).

In order to run the code below, be sure to `pip install mrjob` if you have not
done so already.
'''

from mrjob.job import MRJob
from mrjob.step import MRStep
from nltk.corpus import stopwords
import re

WORD_RE = re.compile(r"[\w']+")
stop = set(stopwords.words('english'))

class MRMostUsedWord(MRJob):

    def mapper_get_words(self, _, txt):
        '''
        If a review's star rating is 5, yield all of the words in the review
        '''
        for word in WORD_RE.findall(txt):
            w = word.lower()
            if w not in stop:
                yield (word.lower(), 1)

    def combiner_count_words(self, word, counts):
        '''
        Sum all of the words available so far
        '''
        yield (word, sum(counts))

    def reducer_count_words(self, word, counts):
        ''' 
        Arrive at a total count for each word in the 5 star reviews
        '''
        yield None, (sum(counts), word)

    # discard the key; it is just None
    def reducer_find_max_word(self, _, word_count_pairs):
        '''
        Yield the word that occurs the most in the 5 star reviews
        '''
        lst=list(word_count_pairs)
        lst=sorted(lst, key=lambda x: x[0],reverse=True)
        
        lst = lst[:10]
        for pair in lst:
            yield pair

    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_words,
                   combiner=self.combiner_count_words,
                   reducer=self.reducer_count_words),
            MRStep(reducer=self.reducer_find_max_word)
        ]

if __name__ == '__main__':
    MRMostUsedWord.run()
```
which gives the top 10 words and counts:
> 956	"new"
953	"one"
823	"life"
648	"world"
538	"book"
524	"love"
460	"time"
417	"story"
413	"first"
393	"years"

Q3 (see here:https://github.com/lsc4ss-a21/assignment-2-yierrr/blob/main/Q3.ipynb)
```
import boto3
import time

session = boto3.Session()

kinesis = session.client('kinesis')
ec2 = session.resource('ec2')
ec2_client = session.client('ec2')


response = kinesis.create_stream(StreamName='stockstream',
                                 ShardCount=1
                                )

waiter = kinesis.get_waiter('stream_exists')
waiter.wait(StreamName='stockstream')


instances = ec2.create_instances(ImageId='ami-02e136e904f3da870',
                                 MinCount=1,
                                 MaxCount=2,
                                 InstanceType='t2.micro',
                                 KeyName='macs30123',
                                 SecurityGroupIds=['sg-02aab282c8cf4a938'],
                                 SecurityGroups=['launch-wizard-1'],
                                 IamInstanceProfile=
                                     {'Name': 'EMR_EC2_DefaultRole'},
                                )

waiter = ec2_client.get_waiter('instance_running')
waiter.wait(InstanceIds=[instance.id for instance in instances])


%%file producer.py
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
                       

%%file consumer.py
import boto3
import time
import json
from datetime import datetime

ec2_client = boto3.client('ec2')
l_ins=[]
response = ec2_client.describe_instances()
for reservation in response["Reservations"]:
    for instance in reservation["Instances"]:
        if instance['State']['Name']=='running':
            l_ins.append(instance["InstanceId"])


sns = boto3.client("sns", region_name="us-east-1")

response = sns.create_topic(Name="stockemail")
topic_arn = response["TopicArn"]

response = sns.subscribe(TopicArn=topic_arn, Protocol="email", Endpoint="yling12@uchicago.edu")


kinesis = boto3.client('kinesis', region_name='us-east-1')

shard_it = kinesis.get_shard_iterator(StreamName="stockstream",
                                     ShardId='shardId-000000000000',
                                     ShardIteratorType='LATEST'
                                     )["ShardIterator"]

no_stop = True
while no_stop:
    out = kinesis.get_records(ShardIterator=shard_it,
                              Limit=1)
    for o in out['Records']:
        jdat = json.loads(o['Data'])
        if jdat['PRICE'] >= 3:
            shard_it = out['NextShardIterator']
            time.sleep(0.2)
        else:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            sns.publish(TopicArn=topic_arn, 
                        Message="Stock price below $3, now is $"+str(jdat['PRICE'])+"Current Time = "+ current_time, 
                        Subject="Price Alert: Stock price below $3")
            
            ec2_client.terminate_instances(InstanceIds=l_ins)

            waiter = ec2_client.get_waiter('instance_terminated')
            waiter.wait(InstanceIds=l_ins)
            print("EC2 Instances Successfully Terminated")
            
            try:
                response = kinesis.delete_stream(StreamName='stockstream')
            except kinesis.exceptions.ResourceNotFoundException:
                pass

            waiter = kinesis.get_waiter('stream_not_exists')
            waiter.wait(StreamName='stockstream')
            print("Kinesis Stream Successfully Deleted")
            
            response = sns.list_subscriptions_by_topic(TopicArn=topic_arn)
            subscriptions = response["Subscriptions"]
            subscription_arn = subscriptions[0]['SubscriptionArn']
            
            sns.unsubscribe(SubscriptionArn=subscription_arn)
            sns.delete_topic(TopicArn=topic_arn)
            no_stop = False
            break
            
            
instance_dns = [instance.public_dns_name 
                 for instance in ec2.instances.all() 
                 if instance.state['Name'] == 'running'
               ]

code = ['producer.py', 'consumer.py']


import paramiko
from scp import SCPClient
ssh_producer, ssh_consumer = paramiko.SSHClient(), paramiko.SSHClient()

time.sleep(60)

instance = 0
stdin, stdout, stderr = [[None, None] for i in range(3)]
for ssh in [ssh_producer, ssh_consumer]:
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(instance_dns[instance],
                username = 'ec2-user',
                key_filename='/Users/sagelim/Desktop/academics/chi/21:22/lsc/LargeScaleComputing_A21/in-class-activities/06_Streaming_Batch_Processing/6M_Kinesis/macs30123.pem')
    
    with SCPClient(ssh.get_transport()) as scp:
        scp.put(code[instance])
    
    if instance == 0:
        stdin[instance], stdout[instance], stderr[instance] = \
            ssh.exec_command("sudo pip3 install boto3 testdata==4.0.1")
    else:
        stdin[instance], stdout[instance], stderr[instance] = \
            ssh.exec_command("sudo pip3 install boto3")

    instance += 1

producer_exit_status = stdout[0].channel.recv_exit_status() 
if producer_exit_status == 0:
    ssh_producer.exec_command("python3 %s" % code[0])
    print("Producer Instance is Running producer.py\n.........................................")
else:
    print("Error", producer_exit_status)

ssh_consumer.close; ssh_producer.close()

print("Connect to Consumer Instance by running: ssh -i \"macs30123.pem\" ec2-user@%s" % instance_dns[1])


! python3 consumer.py
```
which sends emails like this:
![rhoperiods](https://github.com/yierrr/Large-Scale-Comp-HW/blob/main/A2%20mapreduce%20boto3/email.png)

Q4
>We are interested in how twitter users in the US have viewed immigration in the past few years. We do this by first scraping tweets and replies from Twitter and then conducting text analysis. The primary question we seek to answer is how sentiments may evolve over time with changes in the global and political landscape-- whether the sentiment is positive or negative and if such sentiments are diminishing or increasing. We also try to explore how these sentiments may affect each other intertemporally, forming a dynamic process. We plan to use scraping resources like snscrape, which allows us to retrieve data from more than one year ago rather than one week as permitted by the standard Twitter API. Furthermore, regarding large-scale computing resources, we can use AWS Translate to translate tweets by immigrants in their native language to English and AWS Comprehend to arrive at sentiments. This is a suitable project since immigration is an important issue that can be best addressed through social science research, and with the enormous volume of twitter data, large-scale computing methods should provide the most efficient solution. We plan to finish scraping the data in 1 week and analysing the data in another week to make sure the final project can be finished in time. We plan to have two people scrape the data, one on tweets and replies in English and the other on tweets and replies in a foreign language (e.g., Spanish as Mexicans make up the largest share of immigrants to the US). The other two members of the group will then each conduct content and sentiment analysis on the collected data. After the workable data is established, all of us will contribute to the final write-up and presentation. If time permits, we would also like to employ pyspark and explore possible networks within the tweets to investigate topics like homophily. 


