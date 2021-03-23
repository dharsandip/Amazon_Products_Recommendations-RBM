In this project we have built a recommendation system for the electronics products of Amazon using RBM (Restricted Boltzmann Machine) which is an energy-based, stochastic, unsupervised deep learning algorithm. The dataset is downloaded from Kaggle (https://www.kaggle.com/saurav9786/amazon-product-reviews). In fact they downloaded the same from Source - Amazon Reviews data (http://jmcauley.ucsd.edu/data/amazon/). Here, recommendation system has been build with collaborative filtering using RBM model. Since the dataset is very huge (contains more than 7.8 million rows and too many products and ratings) and difficult to hangle in my laptop, here we are taking a subset (10000 lines of original dataset) of that for our current study. 

Attribute Information:

? userId : Every user identified with a unique id (First Column)

? productId : Every product identified with a unique id(Second Column)

? Rating : Rating of the corresponding product by the corresponding user(Third Column)

? timestamp : Time of the rating ( Fourth Column)


Results:

Training loss (RMSE) after epoch=30: 0.4224

In the testing/evaluation part, we have made Single User's predictions (Product Recommendations by RBM from re-constructed Inputs for the products that he/she did not buy/rate). At the same time RBM model made sure that the original rating that he/she gave for the electronic products that he/she purchased/rated, was retained/reproduced. 
In the original product ratings, -1 means user did not purchase/rate the product, 1 rating means user purchased/liked the product and rated positively. 0 means user purchased but did not like the product. In RBM recommendations, it recommends the products to the user that user did not purchase/rate. In RBM recommendations, 0 means RBM does not recommend the product to the user and 1 means RBM recommends the product to the user.   


