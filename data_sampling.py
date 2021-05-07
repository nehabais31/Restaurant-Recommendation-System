
from __future__ import print_function

import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: wordcount <file> <output> ", file=sys.stderr)
        exit(-1)
        
    # Creating instance of Spark Sessions
    sc = SparkContext()
    spark = SparkSession.builder.appName('FinalProject').getOrCreate()
    sqlContext = SQLContext(sc)
    
    # Read the data
    df_business = spark.read.json(sys.argv[1])
    df_reviews  = spark.read.json(sys.argv[2])
    df_users    = spark.read.json(sys.argv[3])
    
    # Filter business data
    df_business_MA = df_business.select('business_id', 'name',  'address', 'city', \
                                          'state', 'postal_code', 'latitude', 'longitude', 'stars', 'review_count', 'categories') \
                                    .filter( (df_business.state == 'MA') & (df_business.categories.contains('Restaurants') ) )

    business_count = df_business_MA.count()
    print(f'MA Business count: {business_count}')
    
    # Sampling data to run on local
    sampled_business_MA = df_business_MA.sample(0.1, 123)
    
    sampled_business_count = sampled_business_MA.count()
    print(f'Sampled Business count: {sampled_business_count}')
    sampled_business_MA.show(2)
    
    
    # Extract reviews of restaurants in MA
    df_reviews_MA = df_reviews.join(sampled_business_MA, on = 'business_id', how = 'inner') \
                              .select(df_reviews.business_id, df_reviews.user_id, df_reviews.review_id, df_reviews.stars, df_reviews.text)
        
    df_reviews_count = df_reviews_MA.count()
    print(f'Business Reviews count: {df_reviews_count}')
    df_reviews_MA.show(2)
    
    # Extract MA user data
    df_users_MA = df_users.join(df_reviews_MA, on = 'user_id', how = 'inner') \
                              .select(df_users.user_id, df_users.name, df_users.review_count, df_users.yelping_since, \
                                      df_users.useful, df_users.funny , df_users.cool , df_users.fans , df_users.average_stars)
        
    df_users_MA_count = df_users_MA.count()
    print(f'Business Users count: {df_users_MA_count}')
    df_users_MA.show(2)
    

    # Store smaller datasets 
    sampled_business_MA.coalesce(1).write.parquet(sys.argv[4])
    df_reviews_MA.coalesce(1).write.parquet(sys.argv[5])
    df_users_MA.coalesce(1).write.parquet(sys.argv[6])

    sc.stop()
