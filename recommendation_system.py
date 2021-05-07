from __future__ import print_function

import sys
import numpy as np

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext


from operator import add
from pyspark.ml.feature import RegexTokenizer, CountVectorizer
from pyspark.ml.feature import StopWordsRemover, VectorAssembler
from pyspark.ml.feature import IDF
from pyspark.ml import Pipeline, PipelineModel

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F



def cosineSimilarity(vector1, vector2):
    '''
    Function to compute cosine similarity between 2 vectors
    -------------------------------------------------------
    Input: 
        Vector-1: Each restaurants feature vector 
        Vector-2: myRestaurant Feature vector 
                  for which recommendations to be made
    
    Returns: A similarity score (between 0 and 1)
    '''
    #  sum(A . B) / (sqrt(sum (A**2)  * sqrt(sum (B**2)))
    numerator = np.dot(vector1, vector2)
    denominator = np.sqrt(np.dot(vector1, vector1)) * np.sqrt(np.dot(vector2, vector2))
    
    return float(numerator/denominator)


def getUserRecommendations(restaurant_ids, all_business_tfidf) :
    '''
    Function to recommend restaurants based on
    cosine similarity score between user reviewed restaurant and others.
    ----------------------------------------------------------------
    Input:  IDs of max 5 restaurants for which recoemmendation is needed
    Return: A dataframe with 5 recommendations for each of user reviews restaurants. 

    '''
    
    # Creating a dataframe to merge results of different input restaurant ids
    schema = StructType([   
                            StructField("business_id", StringType(), True)
                            ,StructField("similarity_score", IntegerType(), True)
                        ])

    similar_restuarants_final = spark.createDataFrame([], schema)


    for rest_id in user_restaurants:

        # Collecting feature values i.e. review texts for user rated restaurants
        user_restaurant_features = all_business_tfidf.map(lambda x: x[1] if x[0] == rest_id else None) \
                                                               .filter(lambda x: x != None).collect()[0]

        # Computing similarity between user rated restaurants and other restaurants reviews
        similar_restaurants_rdd = all_business_tfidf.map(lambda x: (x[0], cosineSimilarity(x[1], user_restaurant_features)))

        # Convert the results into df
        similar_restaurants_df = similar_restaurants_rdd.toDF(schema = ['business_id', 'similarity_score'])

        # Filter out those restaurants from this which are already been rated by our user
        similar_restaurants_df = similar_restaurants_df.filter(similar_restaurants_df.business_id != rest_id)

        similar_restaurants_df = similar_restaurants_df.orderBy('similarity_score', ascending=False).limit(5)


        similar_restuarants_final = similar_restuarants_final.union(similar_restaurants_df)

    # Data might contain duplicate restaurant ids, so removing those
    similar_restuarants_final.dropDuplicates(['business_id']).orderBy('similarity_score', ascending=False).limit(10)
    
    return similar_restuarants_final


def getRestaurantDetails(sim_rest):
    '''
    Function to get the recommended restaurant details based on ids.
    ---------------------------------------------------------------
    Returns -> Name of restaurant
               Category of restaurant
               Rating
               Similarity score with input restaurant id
               Total review count given to that restaurant
    '''
    
    restaurant_details = sim_rest.join(business_df, on='business_id', how = 'inner') \
                                 .select(sim_rest.business_id, \
                                       sim_rest.similarity_score, business_df.name, \
                                       business_df.categories, business_df.stars, business_df.review_count,
                                       business_df.latitude, business_df.longitude)
    
    return restaurant_details


def keyWordsRecommendation(keyword, all_business_tfidf):
    '''
    Function to get Top-10 recommendations based on a keyword
    given by user.
    
    Input:  Keyword
    Output: Dataframe with 10 entries of recommended restaurants
    
    '''
    
    input_word_df = sc.parallelize([(0, keyword)]).toDF(['business_id', 'text'])
    
    # For getting recommendation based on a keywords, first we need to transform it 
    # into word vector. So, we will load our pipelined model that we saved earlier
    input_word_df = pipeline_model.transform(input_word_df)
    
    # Get word2vectors data for this keyword
    input_word_tfidf = input_word_df.select('tfidf_vec').collect()[0][0]
    
    # Get similarity
    similar_restaurants_rdd = all_business_tfidf.map(lambda x: (x[0], cosineSimilarity(x[1], input_word_tfidf)))
    
    sim_rest_by_keyword = similar_restaurants_rdd.toDF(['business_id', 'similarity_score']) \
                                             .orderBy('similarity_score', ascending=False).limit(10)
    
    
    return sim_rest_by_keyword
    
    


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: wordcount <file> <output> ", file=sys.stderr)
        exit(-1)
        
    # Creating instance of Spark Sessions
    sc = SparkContext()
    spark = SparkSession.builder.appName('FinalProject').getOrCreate()
    sqlContext = SQLContext(sc)    
    
    ####################################################################
    #                           Load the data
    ####################################################################
    # Read smaller data generated from sampling
    business_df = spark.read.parquet(sys.argv[1])
    business_count = business_df.count()
    
    reviews_df = spark.read.parquet(sys.argv[2])
    reviews_count = reviews_df.count()
    
    users_df = spark.read.parquet(sys.argv[3])
    users_count = users_df.count()
    
    print(f'Business COunt: {business_count}')
    print(f'Reviews COunt: {reviews_count}')
    print(f'Users COunt: {users_count}')
    
    business_df.show(2)
    reviews_df.show(2)
    users_df.show(2)
    
    ##################################################################
    #                      Text-Processing
    ##################################################################
    # Create review text df from reviews data
    reviews_text = reviews_df.select('business_id', 'text')
    
    # Group all reviews per business
    reviews_by_business = reviews_text.rdd.map(tuple).reduceByKey(add)
    columns = ['business_id', 'text']
    reviews_by_business_df = spark.createDataFrame(reviews_by_business, schema = columns)
    #reviews_by_business_df.show(3)
    
    # count should be the total number of business
    print(f'Total grouped restaurants: {reviews_by_business_df.count()}')
    
    # Now, we will do the text processing
    # Remove the stop words from text, and create the tf idf matrix 
    # Will build a pipeline for this task
    
    tokenizer        = RegexTokenizer(pattern = '\w+', inputCol = 'text', outputCol = 'tokens', toLowercase=True, gaps = False)
    stopWordsRemover = StopWordsRemover(inputCol = 'tokens', outputCol = 'nostopwords')
    countVectorizer  = CountVectorizer(inputCol='nostopwords', outputCol='rawFeatures', vocabSize=1000)
    tfiDF            = IDF(inputCol='rawFeatures', outputCol='tfidf_vec')
    
    pipeline         = Pipeline(stages=[tokenizer, stopWordsRemover, countVectorizer, tfiDF])
    
    # Fit the model
    pipeline_model = pipeline.fit(reviews_by_business_df)
    
    # save the pipeline model
    pipeline_model.write().overwrite().save(sys.argv[4])
    
    # Load the pipeline model
    pipeline_model = PipelineModel.load(sys.argv[4])
    
    # convert the review data into feature vectors
    transformed_reviews_by_business = pipeline_model.transform(reviews_by_business_df)
    
    #transformed_reviews_by_business.select('text', 'tokens', 'nostopWords', 'tfidf_vec').show(5)
    
    ###################################################################
    #                 Making Recommendations to a User based on his
    #                            past reviews
    ##################################################################
     
    #from pyspark.sql.functions import rand
    #usr_id = reviews_df.select('user_id').orderBy(rand()).limit(1).collect()
    #my_user = [val.user_id for val in usr_id][0]
    
    # Selecting a random user
    my_user = '6FvrfCqKu59ItrYM8BF8qg'
    
    # Now, we will make User profile based on the reviews he has given in the past  
    # Selecting all those restaurants that user has rated with more than 2 ratings 
    # and limiting to just 2 reviewed restaurants
    user_reviews = reviews_df.filter( (reviews_df.user_id == my_user) & (reviews_df.stars > float(2.0)) )\
                            .select(reviews_df.business_id).distinct().limit(2)
    
    user_restaurants = [val.business_id for val in user_reviews.collect()]
    
    user_rest_details = user_reviews.join(business_df, on='business_id', how = 'inner')
    
    print(f'\nRestaurants previously reviewed by user: {my_user}')
    user_rest_details.select('business_id', 'name', 'categories', 'stars').orderBy('stars', ascending=False).show()
    
    '''
    Now, for making the recommendations
    We will first fetch the feature values for all the restaurant ids 
    and then find the similarity of our user's rated restaurants feature values with the rest ones 
    '''
    
    # Fetch all business word vectors -> (business_id => [reviews word vectors] ) 
    all_business_tfidf = transformed_reviews_by_business.select('business_id', 'tfidf_vec') \
                                                        .rdd.map(lambda x: (x[0], x[1]))
    
    # Get recommendations for user based on his reviews ones
    similar_restuarants = getUserRecommendations(user_restaurants, all_business_tfidf)
    
    
    # Get restaurant details for our similar df like name, categories, ratings and review count    
    similar_restuarants_details = getRestaurantDetails(similar_restuarants)
    
    
    print(f'\nRestaurants recommended for User: {my_user}')
    similar_restuarants_details = similar_restuarants_details.drop('latitude', 'longitude') \
                                                              .orderBy('similarity_score', ascending=False)
    
    similar_restuarants_details.show(10)
    
    
    ###################################################################
    #           Making Recommendations based on a keyword
    #              given by User using the cosine
    #                similarity of review texts
    ###################################################################
    # Key word similarity in review text for recommendation
    
    key_word = 'Indian'
    
    print(f'\nRestaurants similar to keyword - {key_word}')
    sim_business_keyword = keyWordsRecommendation(key_word, all_business_tfidf)
    
    # Get details of these similar restaurants
    sim_business_keyword_details = getRestaurantDetails(sim_business_keyword)
    sim_business_keyword_details.drop('input_business_id', 'latitude', 'longitude') \
                                .orderBy('stars', ascending=False).show()
                                
  
    sc.stop()