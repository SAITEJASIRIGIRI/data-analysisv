#!/usr/bin/env python
# coding: utf-8

# # Installing pyspark Module

# In[1]:


get_ipython().system('pip install pyspark')


# # Beginning a SparkSession & Building a spark instance

# In[2]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('classification').getOrCreate()


# In[3]:


from pyspark.sql.functions import count, mean, when, lit, create_map, regexp_extract
from itertools import chain


# # Loading and Reading the Dataset

# In[4]:


data1 = spark.read.csv('train.csv',                     header=True, inferSchema=True)
data2 = spark.read.csv('test.csv',                      header=True, inferSchema=True)


# # Viewing the Dataframe Schema 

# In[5]:


data1.printSchema()


# # Showing First 10 Rows of the Dataframe 

# In[6]:


data1.show(10)


# # Transforming Spark Dataframe by Limiting and Converting It to a Pandas Dataframe

# In[7]:


data1.limit(4).toPandas()


# #  Selecting 4 Columns for Inspect within Spark

# In[8]:


data1.select('Pclass', 'Fare', 'Survived', 'Age').show(4)


# # Applying the Summary() Method

# In[9]:


data1.select('Pclass', 'Fare', 'Survived', 'Age').summary().show()


# # Showing Total Number of Rows and Columns present in the Dataframe

# In[10]:


print('No. of columns present in the dataframe: \t', len(data1.columns))
print('No. of rows present in the dataframe: \t', data1.count())


# # Count of People Who Survived

# In[11]:


data1.groupBy('Survived').count().show()


# # Finding Average Fare and Age

# In[12]:


data1.groupBy('Survived').mean('Age', 'Fare').show()


# # Showing Number of Survival According to Sex

# In[13]:


data1.groupBy('Survived').pivot('Sex').count().show()


# # Displaying Number of Survival According to Number of Siblings

# In[14]:


data1.groupBy('Survived').pivot('SibSp').count().show()


# # Viewing Survival Number According to Class Type

# In[15]:


data1.groupBy('Survived').pivot('Pclass').count().show()


# # Showing Survival Number According to Embarked

# In[16]:


data1.groupBy('Survived').pivot('Embarked').count().show()


# # Determining the Number of Survival Based on Parch

# In[17]:


data1.groupBy('Survived').pivot('Parch').count().show()


# # Checking for Null Values

# In[18]:


for col in data1.columns:
    print(col.ljust(20), data1.filter(data1[col].isNull()).count())


# # Extracting Summary for Embarked and Fare 

# In[19]:


data1.select('Embarked', 'Fare').summary('max', '50%', 'mean').show()


# In[20]:


data1 = data1.fillna({'Embarked': 'S', 'Fare':14.45})


# # Extracting the Title Using the Regular Expression and Observing the Count and Average Age

# In[21]:


data1 = data1.withColumn('Title', regexp_extract(data1['Name'],                '([A-Za-z]+)\.', 1))

data1.groupBy('Title').agg(count('Age'), mean('Age')).sort('count(Age)').show()


# # Keeping Four Titles and Mapping Other With One of the First Three

# In[22]:


title_dic = {'Mr':'Mr', 'Miss':'Miss', 'Mrs':'Mrs', 'Master':'Master',              'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr',             'Don': 'Mr', 'Mme': 'Miss', 'Jonkheer': 'Mr', 'Lady': 'Mrs',             'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs',              'Dr':'Mr', 'Rev':'Mr'}

mapping = create_map([lit(x) for x in chain(*title_dic.items())])

data1 = data1.withColumn('Title', mapping[data1['Title']])
data1.groupBy('Title').mean('Age').show()


# #  Creating a Function which Imputes column Age with the Average age

# In[23]:


def ageimpute(data, title, age):
    return data.withColumn('Age',                          when((data['Age'].isNull()) & (data['Title']==title),                               age).otherwise(data['Age']))


# # Imputing the Age

# In[24]:


data1 = ageimpute(data1, 'Mr', 33.02)
data1 = ageimpute(data1, 'Mrs', 35.98)
data1 = ageimpute(data1, 'Miss', 21.86)
data1 = ageimpute(data1, 'Master', 4.75)


# # Creating FamilySize column and Dropping columns SibSp and Parch

# In[25]:


data1 = data1.withColumn('FamilySize', data1['Parch'] + data1['SibSp']).            drop('SibSp', 'Parch')


# # Dropping Unwanted Columns

# In[26]:


data1 = data1.drop('Name', 'PassengerID', 'Ticket', 'Title', 'Cabin')


# # Viewing Trimmed Dataframe

# In[27]:


data1.show(6)


# # Checking for Missing Values

# In[28]:


for col in data1.columns:
    print(col.ljust(20), data1.filter(data1[col].isNull()).count())


# # Importing Model building Libraries

# In[29]:


from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression,                    RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


# # Changing the Embarked and Sex Column from String to Numeric index

# In[30]:


stringIndex = StringIndexer(inputCols=['Embarked', 'Sex'], 
                       outputCols=['EmbNum', 'SexNum'])

stringIndex_model = stringIndex.fit(data1)

data1_ = stringIndex_model.transform(data1).drop('Sex', 'Embarked')
data1_.show(6)


# # Use of VectorAssembler 

# In[31]:


vec_asmbl = VectorAssembler(inputCols=data1_.columns[1:], 
                           outputCol='features')

data1_ = vec_asmbl.transform(data1_).select('Survived', 'features')
data1_.show(6, truncate=False)


# # Splitting the Data for Training and Testing

# In[32]:


train_data, valid_data = data1_.randomSplit([0.8, 0.2])


# # Showing the Train Data for Top 6 Rows

# In[33]:


train_data.show(6, truncate=False)


# # Using MulticlassClassificationEvaluator

# In[34]:


evaluator = MulticlassClassificationEvaluator(labelCol='Survived', 
                                          metricName='accuracy')


# # Building Logistic Regression Model 

# In[35]:


ridge = LogisticRegression(labelCol='Survived', 
                        maxIter=120, 
                        elasticNetParam=0, 
                        regParam=0.03)

model = ridge.fit(train_data)
pred = model.transform(valid_data)
evaluator.evaluate(pred)


# # Developing Random Forest Classifier Model

# In[36]:


rf = RandomForestClassifier(labelCol='Survived', 
                           numTrees=200, maxDepth=5)

model = rf.fit(train_data)
pred = model.transform(valid_data)
evaluator.evaluate(pred)


# # Building GBT Classifier Model 

# In[37]:


gb = GBTClassifier(labelCol='Survived', maxIter=100, maxDepth=3)

model = gb.fit(train_data)
pred = model.transform(valid_data)
evaluator.evaluate(pred)


# # Showing the Test Data

# In[38]:


data2.show(6)


# # Checking for Missing Values inside Test Data

# In[39]:


for col in data2.columns:
    print(col.ljust(20), data2.filter(data2[col].isNull()).count())


# # Creating a FamilySize Feature and Dropping the Unwanted

# In[40]:


data2 = data2.fillna({'Embarked': 'S', 'Fare':14.45})
data2 = data2.withColumn('FamilySize', data2['Parch'] + data2['SibSp']).            drop('Parch', 'SibSp')


# #  Imputing Missing Age

# In[41]:


data2 = data2.withColumn('Title', regexp_extract(data2['Name'],                '([A-Za-z]+)\.', 1))

data2 = data2.withColumn('Title', mapping[data2['Title']])

data2.groupBy('Title').agg(count('Age'), mean('Age')).sort('count(Age)').show()


# # Showing Top 6 Rows after Dropping 4 variables

# In[42]:


data2 = ageimpute(data2, 'Mr', 33.02)
data2 = ageimpute(data2, 'Mrs', 35.98)
data2 = ageimpute(data2, 'Miss', 21.86)
data2 = ageimpute(data2, 'Master', 4.75)

data2 = data2.drop('Ticket', 'Cabin', 'Title', 'Name')
data2.show(6)


# # Checking for Null Values

# In[43]:


for col in data2.columns:
    print(col.ljust(20), data2.filter(data2[col].isNull()).count())


# # Grid-search and Cross-validation

# In[44]:


pipeline_rf = Pipeline(stages=[stringIndex, vec_asmbl, rf])

paramGrid = ParamGridBuilder().            addGrid(rf.maxDepth, [3, 4, 5]).            addGrid(rf.minInfoGain, [0., 0.01, 0.1]).            addGrid(rf.numTrees, [1000]).            build()

selected_model = CrossValidator(estimator=pipeline_rf, 
                                estimatorParamMaps=paramGrid, 
                                evaluator=evaluator, 
                                numFolds=5)

model_final = selected_model.fit(data1)
pred_train = model_final.transform(data1)
evaluator.evaluate(pred_train)


# #  In-sample Accuracy

# In[45]:


pred_test = model_final.transform(data2)

predictions = pred_test.select('PassengerId', 'prediction')
predictions = predictions.                withColumn('Survived', predictions['prediction'].                cast('integer')).drop('prediction')
predictions.show(6)


# In[ ]:




