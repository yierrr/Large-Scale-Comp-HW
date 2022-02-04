# assignment-3-yierrr
Q1 (see:https://github.com/lsc4ss-a21/assignment-3-yierrr/blob/master/Q1.ipynb)
```
from dask_yarn import YarnCluster
from dask.distributed import Client
import matplotlib.pyplot as plt
```
```
cluster = YarnCluster(environment="/home/hadoop/environment.tar.gz",
                      worker_vcores = 4,
                      worker_memory = "8GiB"
                      )
```
```
cluster.scale(4)
client = Client(cluster)
client
```
```
import dask.dataframe as dd

df = dd.read_parquet("s3://amazon-reviews-pds/parquet/product_category=Books/*.parquet",
                     storage_options={'anon': True, 'use_ssl': False},
                     engine='fastparquet')
[df.columns]
```
>[Index(['marketplace', 'customer_id', 'review_id', 'product_id',
        'product_parent', 'product_title', 'star_rating', 'helpful_votes',
        'total_votes', 'vine', 'verified_purchase', 'review_headline',
        'review_body', 'review_date', 'year'],
       dtype='object')]

Exploration 1:
```
place_high_star = (df[['star_rating', 'marketplace']].loc[df['star_rating']>3]
              .groupby('marketplace')
              .sum())
phs_df = place_high_star.compute()
phs_df
%matplotlib inline
phs_df.plot(kind="bar")
plt.title('High star ratings by marketplaces');
```
![good_by_mkt.png](https://github.com/lsc4ss-a21/assignment-3-yierrr/blob/master/good_by_mkt.png)
>This captures the distribution of high ratings (larger than 3) across countries. It can be seen that the US contributes the most high ratings, which could serve as a predictor of high ratings.

Exploration 2:
```
place_help_total = (df[['total_votes', 'marketplace','helpful_votes']].groupby('marketplace')
                                                       .sum())
pht_df = place_help_total.compute()
pht_df
pht_df['ratio']=pht_df['helpful_votes']/pht_df['total_votes']
pht_df[['ratio']].plot(kind="bar")
plt.title('Helpful vote fractions by marketplaces');
```
![help_frac_by_mkt.png](https://github.com/lsc4ss-a21/assignment-3-yierrr/blob/master/help_frac_by_mkt.png)
>This shows how the ratios of helpful votes in all votes are distributed across countries. As shown in the figure, these ratios are mostly similar, but Japan has the highest helpful-total ratio, indicating that using the ratio of helpful votes as a predictor should not be affected/interacted by the countries.

Exploration 3:
```
noveri = (df[['marketplace','verified_purchase']].loc[df['verified_purchase']=='N']
                                                           .groupby('marketplace')                                                        
                                                              .count())
nv_df = noveri.compute()

place_purch = (df[['marketplace','verified_purchase']].loc[df['verified_purchase']=='Y']
                                                           .groupby('marketplace')
                                                           .count())
pp_df = place_purch.compute()
pp_df['not_verified']=nv_df['verified_purchase']
pp_df['ratio']=pp_df['verified_purchase']/(pp_df['not_verified']+pp_df['verified_purchase'])
pp_df[['ratio']].plot(kind="bar")
plt.title('Verified purchase fractions by marketplaces');
```
![veri_by_mkt.png](https://github.com/lsc4ss-a21/assignment-3-yierrr/blob/master/veri_by_mkt.png)
>This shows how the percentages of verified purchases are distributed across countries. As shown in the figure, the percentage of verified purchase is extremely low in Germany, while similar in other countries, with the highest in Japan, indicating that whether a purchase is verified should be taken into consideration when predicting ratings, especially with the case of Germany.

Exploration 4:
```
helpful_by_star = (df[['star_rating', 'helpful_votes']].loc[df['verified_purchase']=='Y']
                                                       .groupby('star_rating')
                                                       .sum())
helpful_df = helpful_by_star.compute()
helpful_df
helpful_df.plot(kind="bar")
plt.title('Helpful votes and star ratings after verifying purchases',y=1.12);
```
![help_star_veri.png](https://github.com/lsc4ss-a21/assignment-3-yierrr/blob/master/help_star_veri.png)
>This shows how star ratings are correlated with helpful votes after purchases have been verified. As shown in the figure, a high correlation between 5-star rating and helpful votes can be conlucded, potentially predicting high ratings like 5 star.

Exploration 5:
```
star_loc = (df[['star_rating', 'marketplace','verified_purchase']].loc[df['verified_purchase']=='Y']
                                            .loc[df['star_rating']>3]
                                            .groupby('marketplace')
                                            .sum())
sl_df = star_loc.compute()


all_star_loc = (df[['star_rating', 'marketplace','verified_purchase']].loc[df['verified_purchase']=='Y']
                                                  .groupby('marketplace')
                                                  .sum())
asl_df = all_star_loc.compute()
sl_df['all_rating']=asl_df['star_rating']
sl_df['ratio']=sl_df['star_rating']/(sl_df['all_rating']+sl_df['star_rating'])
sl_df[['ratio']].plot(kind="bar")
plt.title('High star rating ratios after verifying purchases across countries', y=1.1);
```
![high_ratio_veri_mkt.png](https://github.com/lsc4ss-a21/assignment-3-yierrr/blob/master/high_ratio_veri_mkt.png)
>This shows the ratios of high ratings in all star ratings across countries. As shown in figure, the ratios are approximately similar, indicating countries may not be a good predictor for the ratios.

Q2-4 (see:https://github.com/lsc4ss-a21/assignment-3-yierrr/blob/master/Q2-4.ipynb)
```
sc.install_pypi_package("boto3")
import boto3

s3 = boto3.resource('s3')
bucket = 'amazon-reviews-pds'
bucket_resource = s3.Bucket(bucket)

for obj in bucket_resource.objects.all():
    if 'parquet' in obj.key and 'Books' in obj.key:
        print(obj.last_modified,"\t", round(obj.size * 1e-9), "GB\t",
              obj.key, "\n")
data = spark.read.parquet('s3://amazon-reviews-pds/parquet/product_category=Books/*.parquet')
sc.install_pypi_package("pandas==1.0.3")
sc.install_pypi_package("scipy==1.4.1")
sc.install_pypi_package("matplotlib==3.2.1")
sc.install_pypi_package("seaborn==0.10.1")
```


Q2
```
# original data plus good_review dummy
data_good = data.withColumn('good_review', (data.star_rating >= 4).cast("integer"))
#counting good reviews
good_count = (data_good.groupBy('good_review')
             .count())
l_val = good_count.collect()
num_good = 0
num_bad = 0

if l_val[0][0] == 1:
    num_good = l_val[0][1]
    num_bad = l_val[1][1]
else:
    num_good = l_val[1][1]
    num_bad = l_val[0][1]
frac_bg = num_bad/num_good
# balanced data with good_review dummy
new_data = data_good.sampleBy("good_review", fractions={0: 1, 1: float(frac_bg)}, seed=1234)
new_count = (new_data.groupBy('good_review')
             .count())
new_count.collect()[0][1]/new_count.collect()[1][1]
```
>0.999779401940467
```
import seaborn as sns
import matplotlib.pyplot as plt

df = new_count.toPandas()

sns.barplot(x='good_review', y='count', data=df)
plt.title('Numbers of good and non-good reviews')
%matplot plt
```
![q2-bal.png](https://github.com/lsc4ss-a21/assignment-3-yierrr/blob/master/q2-bal.png)
The sample is much more balanced than before.

Q3
a)
```
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import size
from pyspark.ml import Pipeline
tokenizer = Tokenizer(inputCol='review_body', outputCol='text')
remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol='word_lst')
pipeline = Pipeline(stages=[tokenizer,remover])
new_data = new_data.na.drop()
new_data = pipeline.fit(new_data).transform(new_data)
new_data = new_data.withColumn('body_length', size('word_lst'))
# Additional 1: Chose length of review body as a covariate because usually longer reviews should 
# be given to more favored books.
new_data = new_data.withColumn('verified', (new_data.verified_purchase == "Y").cast('integer'))
# Additional 2: Chose whether a purchase is verified because not verified reviews can be noisy
# and may not actually relate to quality/ratings of the books.
new_data = new_data.withColumn('In_US', (new_data.marketplace == "US").cast('integer'))
# Additional 3: Chose the marketplace to be in the US as most good reviews come from the US.
new_data = new_data.withColumn('total_votes', (new_data.total_votes).cast("integer"))
```
>1. length of review body: Usually longer reviews should be given to higher-rated books than lower-rated ones.
>2. verified purchase: Not verified reviews can be noisy and may not actually relate to quality/ratings of the books.
>3. marketplace (in US or not): Most good reviews come from the US in the existent data.

b)
```
train, test = new_data.randomSplit([0.7, 0.3])
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
features = ['total_votes', 
            'body_length', 
            'verified',
            'In_US']
assembler = VectorAssembler(inputCols = features, outputCol = 'features',handleInvalid='skip')
lr = LogisticRegression(featuresCol='features', labelCol='good_review')
pipeline = Pipeline(stages = [assembler, lr])
```
c)By designing a pipeline, we are setting the actions to be taken when the data is passed in without actually giving the data or executing anything. The dataframe will not be processed when setting the pipeline. It will only be processed when the data is fitted into the pipeline, and in a manner as designed in the pipeline. This is similar to Dask where we can delay actions and compute them later.


Q4
```
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import numpy as np
paramGrid = ParamGridBuilder()\
    .addGrid(lr.regParam,  np.arange(0, .1, 0.01))\
    .addGrid(lr.elasticNetParam, [0, 1])\
    .build()

evaluator = BinaryClassificationEvaluator().setLabelCol('good_review')
cv = CrossValidator(estimator = pipeline,
                    estimatorParamMaps = paramGrid,
                    evaluator = evaluator,
                    numFolds = 5, seed = 0)
# Downsizing the sample for computation time concern.
train_sample = train.sample(fraction=0.00001)
test_sample = test.sample(fraction=0.00001)
#fit model
model = cv.fit(train_sample)

print("Train AUC:", evaluator.evaluate(model.transform(train_sample), {evaluator.metricName: "areaUnderROC"}))
print("Test AUC:", evaluator.evaluate(model.transform(test_sample), {evaluator.metricName: "areaUnderROC"}))
```
>Train AUC: 0.7793560606060607
>Test AUC: 0.4954545454545455
```
trainingSummary = model.bestModel.stages[-1].summary
print("Training AUC: " + str(trainingSummary.areaUnderROC))

print("\nFalse positive rate by label (Training):")
for i, rate in enumerate(trainingSummary.falsePositiveRateByLabel):
    print("label %d: %s" % (i, rate))
print("\nTrue positive rate by label (Training):")
for i, rate in enumerate(trainingSummary.truePositiveRateByLabel):
    print("label %d: %s" % (i, rate))    
print("\nTraining Accuracy: " + str(trainingSummary.accuracy))
```
>Training AUC: 0.7793560606060604
False positive rate by label (Training):
label 0: 0.25
label 1: 0.45454545454545453
True positive rate by label (Training):
label 0: 0.5454545454545454
label 1: 0.75
Training Accuracy: 0.6521739130434783

As can be seen above, the model works well (and much better) with training data than testing data, leading to very poor predictions of new data. AUC for test data is only about 0.5 and there exist far-from-zero false positive and false negative rates. This may be because that the model only has four covariates independent from each other, while there can be great interaction term effect. Another way that might improve this is to use a larger dataset to train the model (without the limitations of computation power). Removing the restriction of linearity of the model may also help with prediction.
