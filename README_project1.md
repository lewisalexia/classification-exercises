# To Churn or not to Churn
# Project Description
 
Telco has been experiencing high amounts of customer churn in the past month. 
 
# Project Goal
 
* Discover drivers of churn in Telco's customer base.
* Use drivers to develop a machine learning model to classify customers as ending in churn or not ending in churn.
* Churn is defined as a customer leaving the company. 
* This information could be used to further our understanding of which elements contribute to or detract from a customer's decision to churn.
 
# Initial Thoughts
 
My initial hypothesis is that customers are churning because of monthly contract availability and associated cost.
 
# The Plan
 
* Aquire data from MySQL Server (using Codeup credentials in env.py file).
 
* Prepare data
   * Look at the data frame's info and note:
		* nulls
		* corresponding value counts
		* object types
		* numerical types
		* names of columns
 
* Explore data in search of drivers of churn
   * Answer the following initial questions:
       * How often does churn occur?
       * Does contract type affect churn?
       * Does age affect churn?
       * Does being single or having dependents affect churn?
       * Does payment type affect churn?
       * Does internet type affect churn?
       
* Develop a Model to predict if a customer will churn:
   * Use drivers identified in the explore phase to build predictive models of different types.
   * Evaluate models on the train and validate datasets.
   * Select the best model based on the highest accuracy.
   * Evaluate the best model on the test data.
 
* Draw conclusions
	* Merge findings back onto the customer base to identify customers with a high chance of churning.
	* Make recommendations to minimize churn.
	* Identify takeaways from company data.	 
 
# Data Dictionary

**INSERT TABLE HERE**
 
# Steps to Reproduce
1) Clone this repo.
2) Acquire the data from MySQL servers using your own Codeup credentials stored in an env.py file.
3) Put the data in the file containing the cloned repo.
4) Run notebook.
 
# Takeaways and Conclusions
* 
* 
* 
* 
* 
* 
* 
* 
 
# Recommendations
* 
* 
* 
