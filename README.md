# regression-project
#### Welcome to this initial exploration of Zillow Taxable_Value Data for Los Angeles, Orange, and Ventura Counties in Southern California!  The context of this project is limited to properties in the aforementioned counties which were subject to a property sale in 2017.
#### The goals of this initial exploration are as follows:
- Discover a means to reliably predict the Assessed_Taxable_Value of a property based upon previous observation points.
- Discover a means whereby a relatively noisy dataset can be subsetted into meaningful subsets (based on domain knowledge) in order to extract meaningful signal.
- Reduce dimensionality via selection tools and identifying multicolinearity via pair-wise plots of each of the predictors.
- Identify possible routes of future fruitful exploration and modeling based upon discoveries found within a time-limited context.
- Create a Machine Learning Model which explains the greatest amount of the variance of the target given a very-short time window for EDA and modeling.

#### PROJECT DESCRIPTION:
- This project is designed to provide recommendations for improvements upon MODEL Z, a model currently fielded by the Zillow Data Science Team.
- With MODEL Z, the Zillow Data Science Team is looking to refine predictive capability within a particular subset of Southern California property data, specifically LA County, Orange Count, and Ventura County.
- With some exploratory data analysis and prototype ML modeling, we can make recommendations to adapt MODEL Z so that it can perform with an increased capacity to explain the variance of the target (Assessed_Taxable_Value) greater than the current iteration.

#### Project Planning:
- Plan: Questions and Hypotheses
- Acquire: Compile several custom datasets from Zillow SQL server
- Prepare: Encode required columns,removed outliers, removed entries with NULL values for important predictors and split into ML subsets (Train/Validate/Test). Finally derived missing Year_Built data based upon the training split, and imputed that value to the validate and test split.
- Explore: Given a noisy, non-homogenous data set, discover meaningful subsets which can be utilized to create a statistically usable model, which can then be adapted/altered to become usable for increasingly-complex datasets. 
- Model: Firstly, create an MVP model which is a Vanilla Multiple Linear Regression model specifically using SQFT/BEDS/BATHS.  Given the short-fused nature of the project, large, non-refined steps will be taken to find meaningful recommendations.  For example, a transition from a Multiple Linear Model to a Random Forest Regressor may not be accompanied by the expected tuning of important hyper-parameters for the new model.  We are looking to obtain a Vanilla-to-Vanilla comparison of the various models, while also knowing that significantly different results will come with further tuning.
- Deliver: Please refer to this doc as well as the Final_Report.ipynb file for the finished version of the presentation, in a ddition to a brief 5-min presentation to the Zillow DS Team.

#### Initial hypotheses and questions:
#### What meaningful subsets of the dataset can be leveraged to create a model that displays a significant improvement in performance when compared to MODEL Z?
- Do properties in Los Angeles County have a statistically significant different mean than the overall population?
- Do properties in Orange County have a statistically significant different mean than the overall population?
- Does a Statistically significant correlation exist between SQFT feature and Baths feature?
- Does a Statistically significant correlation exist between SQFT feature and Taxable_Value feature?


#### Data Dictionary: note outliers default defined as 1.5 * IQR, except as specified 


beds:           | integer    |   number of bedrooms (outliers defined as < 1 and > 5) 

baths:          | float      |   number of bathrooms (outliers defined as < 1 and > 4)           

sqft:           | float      |   livable square footage as calculated by county assessors office 

taxable_value:  | float      |   TARGET: Assessed Taxable Value as calculated by county assessors office

built:          | float      |   Year of property construction

lotsqft:        | float      |   square footage of land subject to tax calculations 

fips:           | object     |   Federal county code: 6037 LA County, 6059 Orange County, 6111 Ventura County (California)

city:           | object     |   Propietary Code used to denote City, mapping is currently unavailable

o_sqft:         | float      |   an attempt to combine SQFT-BEDS-BATHS into one data point to reduce effects of colinearity

o_sqft = SQFT / (BEDS + BATHS)  
purpose is to compare the amount of square footage attributable to common areas such 
as kitchen, living room, dining room, etc., while controlling for the effects of number of BED/BATH contribution to SQFT.


#### Instructions for those who wish to reproduce this work or simply follow along:
You Will Need (ALL files must be placed in THE SAME FOLDER!):
- 1. final_report.ipynb file from this git repo
- 2. wrangle_zillow.py file from this git repo 
- 3. env.py file to be generated by IT.  You must use your own env.py file with your own sign-in credentials.  Please contact IT if you are in need of obtaining this.  Otherwise, feel free to create your env.py script with any text editor by following these steps:

- In your newly opened file, we will place 2 Blocks of Info.  Copy/Paste the following three lines:

            host = 'address for database server'
            user = 'your assigned user name'
            pwd = 'the password associated with your user name'

            # The 2nd Block of Info: (Copy/Paste the 2nd Block below the 1st Block)
            def get_db_url(db_name,u_name=user,pwd=pwd,h_name=host):
                return f'mysql+pymysql://{u_name}:{pwd}@{h_name}/{db_name}'

Ensure:
- All 3 files are in the SAME FOLDER
- wrangle_zillow.py and env.py each have the .py extension in the file name
- The two Blocks of Info in the env.py file are all aligned at the left-most index of the page EXCEPT the final line of code beginning with "return".  This line should be 4 whitespaces (or 1 TAB) indented from the rest.
- If your choose to CREATE your env.py file, ensure that each of the 3 string values (host,uname,pwd) are wrapped in single-quotes ' ' as in the example.

Any further assistance required, please email me at myemail@somecompany.com.

#### Findings, Recommendations, and Takeaways:


- Property taxes are assessed at county level of granularity.  Our models need to begin at that level.  Let us reduce the noise factor in the dataset by subsetting by county.
- With that being said, let us now choose the county with the greatest number of datapoints, but also the least homogeneous-LA County in order to find an algorithm that performs best in this noisy environment.
- Baseline stats derived from y-target MEAN: RMSE=243k r2=0
- Linear Model #1 using SQFT/BEDS/BATHS stats: slight improvement from baseline r2=0.142
- DecisionTreeRegressor and RandomForestRegressor displayed improvement over LM.
- Random Forest with 55 trees and max_depth of 10 (CodeName; JUNIOR) achieved 0.242 r-squared on validation set
- Applied Random Forest 55/10 (JUNIOR) to VENTURA County subset (less noise and more homogenous), 0.420 r-squared on validation set, but appears to suffer from significant OVERFITTING (0.618 on train)
- FUTURE GOAL:  1. Create a model that works.
                2. Adapt as needed for iteratively differing housing environments (ex, SoCal is different from suburban Pittsburgh)
- How to get there: 1. Get better data: too noisy, not homogenous enough...for example need to incorporate GEO data in a meaningful 
                    way...convert lat/longs into something uselful.
                    2. Obtain more domain knowledge about specific market, assessed taxes by county, etc...numbers devoid of context are useless clutter (noise).  For example, the typical county tax assessor does not care that you have a fireplace or a hot tub...those are market concerns, not taxation concerns.
                    3. Work with different algorithms to find better results, specifically tree-based approaches, Gradient Boosted Trees, XGBoost, ADABoost.  Then exploit the ability to tune hyper-parameters.

