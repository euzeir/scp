# scp
This is the project for the exame of Scalable and Cloud Programming for the academic year 2020-2021.
Our objective in this project is to implement well known big data and machine learning algorithms and to show how to deploy them into a cluster of computers (AWS or GC).

In more details we are going to implement the following algorithms:

- Regression::
	1. Linear Regression:
		Purpose: Predicting the price of a house based on a set of features.
		Dataset: "housing.csv" (a dataset of 5000 rows and 6 columns)

- Classification::
	2. Logistic Regression:
		Purpose: Predicting if a client will subscribe (1) or not (0) into a bank.
		Dataset: "banking.csv" (a dataset of 41189 rows and 21 columns)
		[https://raw.githubusercontent.com/madmashup/targeted-marketing-predictive-engine/master/banking.csv]

- Clustering::
	K-Means:
		Purpose: Cluster a dataset of multidimentional points
		Dataset: https://data.humdata.org/dataset/catalog-of-earthquakes1970-2014

- Graphs::
	BFS:
		Purpose: BFS algorithm on social networks to find the hops from one starting node to a target node.
		Datasets: http://networkrepository.com/index.php
				  https://snap.stanford.edu/data/	

The project should be developed using Scala as programming language and Apache Spark.


Required:
	- Scala version 2.12.12
	- Spark version 3.0.1

Eduart Uzeir && Domenico Coriale
