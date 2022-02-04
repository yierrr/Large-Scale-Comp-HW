import boto3
import time
import json


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

response = sns.subscribe(TopicArn=topic_arn, Protocol="email", Endpoint="email")


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
            sns.publish(TopicArn=topic_arn, 
                        Message="Stock price below $3, now is $"+str(jdat['PRICE']), 
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

    
