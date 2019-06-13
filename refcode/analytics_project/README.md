Final Project Analytics Course University of Maryland Fall 2016 - KDD Cup 2014 Data Set
 
Forked from: https://github.com/yoonkim/kdd_2014
Code for KDD Cup 2014 (Winning Entry)

https://www.kaggle.com/c/kdd-cup-2014-predicting-excitement-at-donors-choose

Original Instructions
========
Let -path be the folder with all the data 

To run: 

1. python kdd_2014_data_model1.py -path

2. Run kdd_2014_model1.R (change folder <- path) 

3. download ESLI data from http://nces.ed.gov/ccd/elsi/tableGenerator.aspx (Public School, Years 2011-2012, columns school id and school type) 

4. Run kdd_2014_model2.R (change folder <- path) until line 126 

5. python kdd_2014_data_model2.py -path

6. Run kdd_2014_model2.R from line 126 until the end

7. Final prediction is (0.5*model1+0.5*model2)*discount
