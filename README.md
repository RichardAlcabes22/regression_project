# regression-project
#### Welcome to this initial exploration of Zillow Taxable_Value Data for Los Angeles County, Orange County, and Ventura County 2017!
#### The goals of this initial exploration are as follows:
- Discover 
- Discover 
- Discover 
- Identify 
- Create a Machine Learning Model which can accurately 

#### PROJECT DESCRIPTION:
- This project is designed to provide an improvement upon xxxx efforts.  Currently, Telco is unable to accurately identify customers who are likely to churn.
- Therefore, Telco is currently unable to provide Targeted Loyalty Offers to those customers in an effort to retain their business.
- With some analysis and prototype ML modeling, we can ientify Customers who are likely to churn, with a success rate greater than the current rate of 0%.

#### Initial hypotheses and questions:
#### What % of Telco Customers have No "Perceived Commitment" to Telco in combination with low "Switching Costs"?
- Do customers with "Month-to-Month" contracts churn out at a statistically significant greater rate than the overall population?
- Do customers with "Paperless Billing" enabled churn out at a statistically significant greater rate than the overall population?
- Do customers without "Dependents" churn out at a statistically significant greater rate than the overall population?
- Do customers who comprise the largest customer segment churn out at a statistically significant greater rate than the overall population?

#### HYPOTHESIS: Customers in this segmnent churn at greater rates than the overall Telco customer base

#### Data Dictionary 
customer_id:        |          integer          |         each customer has unique ID

gender:             |         string             |       Male/Female           

senior_citizen:     |         enumerated integer    |    1-YES 2-NO

partner:            |          string            |        Yes/No

dependents:         |          string             |       Yes/No

tenure:             |          integer            |      number of months customer has been with Telco

phone_service / online_security / online_backup /
device_protection / tech_support / streaming_tv / 
streaming_movies / paperless_billing:  
                    |         string             |     Yes/No

multiple_lines:     |         string             |      Yes/No/No phone service

contract_type:      |         string              |      Month-to-month/One year/Two year

internet_service_type:   |     string            |      DSL/Fiber optic/None

payment_type:       |         string              |      Bank transfer(auto)/Credit card/ 
                                                        (auto)/Electronic check/Mailed check

#### Project Planning:
- Plan: Questions and Hypothesis
- Acquire: Compile custom dataset from Telco SQL server
- Prepare: Encode required columns and split into ML subsets (Train/Validate/Test)
- Explore: Discover categories with significant percentages of customers as well as those with Churn rates which are significantly greater than the Customer average.
- Model: Design a prototype ML model which is initially designed to identify the greatest percentage of customers who are likely to churn.  We will place initial emphasis on Accuracy and Recall for the Positive (churn) class.  Type II errors lead to churn, but Type I errors also lead to wasted resources.  We will prioritize the minimization of Type II errors.
- Deliver: Please refer to this doc as well as the Final_Report.ipynb file for the finished version of the presentation.


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

- Telco currently has a large segment (~22%) of its customer base that exhibits three characteristics of churn, and an unmeasured pluarality of customers who exhibit at least one characteristic. 
- While, this customer segment is the largest in Telco, it also suffers from a Churn-rate that is greater than the mean Churn Rate for the overall customer base
- A prototype ML model has been developed to ID a significant percentage of customers who Churn.  This represents an improvement upon the current situation in which 0% of customers who churn are identified.

# NOTE:
## the PREDICTIONS .csv file code can be found at the end of the project_exploration notebook, rather than in the Final_Report itself.
