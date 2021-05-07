# Restaurant Recommendation System  
## Using Content Based Filtering Method





The idea is to make a recommendation sytem for recommending top restuarants of Massachussets to a user based on his previous ratings or feedbacks.  
Also, generating some recommendations based on a specific keyword, like what are the top similar Indian Restaurants or what are some top similar pizza restaurants.  
Cosine similarity metric is used to find the similar items.  
The dataset used is Yelp Data containing details of various businesses, reviews and users' profiles.  My project is narrowed down to restaurant category of Massachussets state.    
  
The Large data can be downloaded from :  
https://www.kaggle.com/yelp-dataset/yelp-dataset?select=yelp_academic_dataset_business.json
Required Files to download:  
  1. yelp_academic_dataset_business.json  
  2. yelp_academic_dataset_review.json  
  3. yelp_academic_dataset_user.json  
Download the above files and save in a folder named Large_Yelp_datasets.




# Submit your python scripts .py 

3 scripts:  
  1. data_sampling.py ( This is for reference showing how I filtered and sampled my data to run on local machine)  
  2. recommendation_system.py -  Main task file implementing the entire recommendation system using the sampled data from above script
  3. Restaurant_Recommendation_System.ipynb - Jupyter Notebook for the recommendation task ( Content is similar to the .py script)  

# Other Documents. 
  1. Powerpoint slides  
  2. Visualization Image for Result


# How to run  
### 1. Download large datasets from kaggle and save in a folder named Large_Yelp_datasets  
###    2. Then run data_sampling.py to sample the data and selecting only Massachessets data.  
###    3. At last run recommendation_system.py using the small sampled files generated from Step-2.  

```python

spark-submit <script name> <Input file 1> <Input File 2> <Input File 3> <Output folder 1> <Output folder 2> <Output folder 3>
  
Script name:    data_sampling.py
Input File 1:   Large_Yelp_datasets/yelp_academic_dataset_business.json 
Input File 2:   Large_Yelp_datasets/yelp_academic_dataset_review.json  
Input File 3:   Large_Yelp_datasets/yelp_academic_dataset_user.json  
Output Folder1: Small_Data/business_MA
Output Folder2: Small_Data/reviews_MA
Output Folder3: Small_Data/users_MA

```


```python

spark-submit <script name> <Input file 1> <Input File 2> <Input File 3> < Output folder to save Pipeline Model>
  
Script name:    recommendation_system.py
Input File 1:   Small_Data/business_MA  
Input File 2:   Small_Data/reviews_MA  
Input File 3:   Small_Data/users_MA  
Output Folder:  pipeline_model

```

