# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 00:39:45 2016

@author: attam
"""
import csv
import math
from matplotlib import pyplot as plt
import pandas as pd
#import textblob as tb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from collections import Counter
import re
from collections import Counter
from collections import defaultdict
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import linear_model
from sklearn import tree

o = pd.read_csv('outcomes.csv')
d = pd.read_csv('donations.csv')
c = pd.merge(d, o, how='left', on='projectid')
#c.columns
c=c.dropna(axis=0)

r = pd.read_csv("resources.csv")
f = pd.merge(r, o, how='left', on='projectid')
#d.columns
f=f.dropna(axis=0)
d1 = pd.merge(f, c, how='left', on='projectid')

p=pd.read_csv('projects.csv')
e = pd.merge(p, o, how='left', on='projectid')
e = e.dropna()
d2 = pd.merge(e, d1, how='left', on='projectid')
d2.columns
d2=d2.dropna(axis=0)

def focus_edu(edu):
    if edu in ('Environmental Science','Mathematics','Applied Sciences',):         
        return 'Science'     
    elif edu in ('Civics & Government','History & Geography','Social Sciences','Economics'):         
        return 'Social Sciences'
    elif edu in ('Extracurricular','Character Education','Sports','College & Career Prep','Community Service'):         
        return 'Extracurricular'
    elif edu in ('Literature & Writing','Visual Arts','Music','Performing Arts','Literacy','Foreign Languages'):         
        return 'Art/Literature'
    elif edu in ('Nutrition','Health & Wellness','Gym & Fitness'):         
        return 'Health' 
    else:         
        return 'Others'

# Process State
def group_states(state):
    if state in ('AL','AR',
       'IN', 'IL', 'KY', 'KS', 'LA', 'MO', 'MN', 'MS', 'NE', 'ND', 'OK', 'SD', 'TX', 'TN', 'WI'):
        return 'CST'
    elif state in ('FL', 'NJ', 'MD',
                     'NY', 'IA', 'CT', 'DE', 'GA', 'MA', 'ME', 'MI', 'NH', 'NC', 'OH', 'PA', 'RI', 'SC', 'VA', 'VT', 'DC', 'WV'):
        return 'EST'
    elif state in ('CA', 'NV', 'OR', 'WA'):
        return 'PST'
    elif state in('AZ', 'CO', 'ID', 'MT', 'NM', 'UT', 'WY'):
        return 'MST'
    else:
        return 'other'
d2['state_bin']=d2['donor_state'].map(group_states)

# Process dollar amount
def group_amount_dollar(amount):
    if amount in('10_to_100'):
        return 'Between 10 and 100'
    elif amount in('under_10'):
        return 'Below 10'
    else:
        return 'above 100'    
d2['amount_bin']=d2['dollar_amount'].map(group_amount_dollar)

# Process payment method
def group_payment_method(payment):
    if payment in('paypal', 'creditcard', 'amazon', 'check'):
        return 'Online'
    elif payment in('promo_code_match', 'double_your_impact_match', 'almost_home_match'):
        return 'Discount'
    else:
        return 'no cash received'    
d2['payment_bin']=d2['payment_method'].map(group_payment_method)

#process for honoree
def group_for_honoree(value):
    if value=='f':
        return 'false_for_honoree'
    elif value=='t':
        return 'true_for_honoree'
    else:
        return value    
d2['honoree_bin']=d2['for_honoree'].map(group_for_honoree)

#label_obj = preprocessing.LabelEncoder()
#df['honoree_bin'] = label_obj.fit_transform(df['for_honoree'])
#df['honoree_bin'].unique()

#process via_giving_page
def group_via_giving_page(value):
    if value in('f'):
        return 'false_via_giving_page'
    elif value in('t'):
        return 'true_via_giving_page'
    else:
        return value    
d2['via_giving_page_bin']=d2['via_giving_page'].map(group_via_giving_page)

#process payment_was_promo_matched
def group_payment_was_promo_matched(value):
    if value in('f'):
        return 'false_payment_was_promo_matched'
    elif value in('t'):
        return 'true_payment_was_promo_matched'
    else:
        return value    
d2['payment_was_promo_matched_bin']=d2['payment_was_promo_matched'].map(group_payment_was_promo_matched)


#process payment_included_web_purchased_gift_card
def group_payment_included_web_purchased_gift_card(value):
    if value in('f'):
        return 'false_payment_included_web_purchased_gift_card'
    elif value in('t'):
        return 'true_payment_included_web_purchased_gift_card'
    else:
        return value    
d2['payment_included_web_purchased_gift_card_bin']=d2['payment_included_web_purchased_gift_card'].map(group_payment_included_web_purchased_gift_card)

#process payment_included_campaign_gift_card
def group_payment_included_campaign_gift_card(value):
    if value in('f'):
        return 'false_payment_included_campaign_gift_card'
    elif value in('t'):
        return 'true_payment_included_campaign_gift_card'
    else:
        return value    
d2['payment_included_campaign_gift_card_bin']=d2['payment_included_campaign_gift_card'].map(group_payment_included_campaign_gift_card)

#process payment_included_acct_credit
def group_payment_included_acct_credit(value):
    if value in('f'):
        return 'false_payment_included_acct_credit'
    elif value in('t'):
        return 'true_payment_included_acct_credit'
    else:
        return value    
d2['payment_included_acct_credit_bin']=d2['payment_included_acct_credit'].map(group_payment_included_acct_credit)

#process donation_included_optional_support
def group_donation_included_optional_support(value):
    if value in('f'):
        return 'false_donation_included_optional_support'
    elif value in('t'):
        return 'true_donation_included_optional_support'
    else:
        return value    
d2['donation_included_optional_support_bin']=d2['donation_included_optional_support'].map(group_donation_included_optional_support)

#process is_teacher_acct
def group_is_teacher_acct(value):
    if value in('f'):
        return 'false_is_teacher_acct'
    elif value in('t'):
        return 'true_is_teacher_acct'
    else:
        return value    
d2['is_teacher_acct_bin']=d2['is_teacher_acct'].map(group_is_teacher_acct)

# Process item_unit_price
def group_item_unit_price(amount):
    if amount>0 and amount<100:
        return 'Between 0 and 100'
    elif amount>=100 and amount<1000:
        return 'Between 100 and 1000'
    elif amount>=1000 and amount<10000:
        return 'Between 1000 and 10000'
    elif amount>=10001 and amount<20000:
        return 'Between 10000 and 20000'
    elif amount>=20001 and amount<35000:
        return 'Between 20000 and 35000'
    else:
        return 'above 35000'    
d2['item_unit_price_bin']=d2['item_unit_price'].map(group_item_unit_price)

# Process item_quantity
def group_item_quantity(amount):
    if amount>0 and amount<10:
        return 'Between 0 and 10'
    elif amount>=11 and amount<100:
        return 'Between 10 and 100'
    elif amount>=101 and amount<500:
        return 'Between 100 and 500'
    elif amount>=501 and amount<1000:
        return 'Between 500 and 1000'
    else:
        return 'above 1000'    
d2['item_quantity_bin']=d2['item_quantity'].map(group_item_quantity)

d2 =  d2.drop(['school_ncesid',
 'school_latitude',
 'school_longitude',
 'school_city',
 'school_state',
 'school_zip','school_ncesid',
 'school_latitude',
 'school_longitude',
 'school_city',
 'school_state',
 'school_zip','school_metro',
 'school_district',
 'school_county',
 'school_charter',
 'school_magnet',
 'school_year_round','school_nlns',
 'school_kipp',
 'school_charter_ready_promise',
 'teacher_prefix',
 'teacher_teach_for_america',
 'teacher_ny_teaching_fellow','total_price_including_optional_support',
 'students_reached',
 'eligible_double_your_impact_match',
 'eligible_almost_home_match',
 'date_posted', 'secondary_focus_subject',
 'secondary_focus_area'], axis = 1)
 
#d2["is_exciting_"] = 0
#d2["is_exciting_"][d2["is_exciting"]=="t"] = 1
d2["at_least_1_teacher_referred_donor_"] = 0
d2["at_least_1_teacher_referred_donor_"][d2["at_least_1_teacher_referred_donor"]=="t"] = 1
d2["great_chat_"] = 0
d2["great_chat_"][d2["great_chat"]=="t"] = 1
d2["fully_funded_"] = 0
d2["fully_funded_"][d2["fully_funded"]=="t"] = 1
d2["at_least_1_green_donation_"] = 0
d2["at_least_1_green_donation_"][d2["at_least_1_green_donation"]=="t"] = 1
d2["donation_from_thoughtful_donor_"] = 0
d2["donation_from_thoughtful_donor_"][d2["donation_from_thoughtful_donor"]=="t"] = 1
d2["three_or_more_non_teacher_referred_donors_"] = 0
d2["three_or_more_non_teacher_referred_donors_"][d2["three_or_more_non_teacher_referred_donors"]=="t"] = 1
d2["one_non_teacher_referred_donor_giving_100_plus_"] = 0
d2["one_non_teacher_referred_donor_giving_100_plus_"][d2["one_non_teacher_referred_donor_giving_100_plus"]=="t"] = 1
d2["teacher_referred_count_"] = 0
d2["teacher_referred_count_"][d2["teacher_referred_count"]>=1] = 1
d2["non_teacher_referred_count_"] = 0
d2["non_teacher_referred_count_"][d2["non_teacher_referred_count"]>=1]=1
d2['primary_subject_bin'] = d2['primary_focus_subject'].apply(focus_edu)

d2 =  d2.drop([
 'at_least_1_teacher_referred_donor',
 'great_chat',
 'fully_funded',
 'at_least_1_green_donation',
 'donation_from_thoughtful_donor','three_or_more_non_teacher_referred_donors',
 'one_non_teacher_referred_donor_giving_100_plus',
 'teacher_referred_count',
 'non_teacher_referred_count','primary_focus_subject'], axis = 1)
 
d2.count()
d2=d2.drop('donor_city',1)
d2=d2.drop('donation_timestamp',1)
d2=d2.drop('donation_message',1)
d2=d2.drop('donor_zip',1)
d2=d2.drop('donationid',1)
d2=d2.drop('donor_acctid',1)
d2=d2.drop('donation_to_project',1)

d2 = d2[d2.item_unit_price > 0]

d2=d2.drop('resourceid',1)
d2=d2.drop('vendorid',1)
d2=d2.drop('item_name',1)
d2=d2.drop('item_number',1)

d3=pd.get_dummies(d2[['primary_subject_bin','is_exciting','item_unit_price_bin','item_quantity_bin','vendor_name','project_resource_type','payment_was_promo_matched_bin' ,'donation_included_optional_support_bin', 'is_teacher_acct_bin', 'payment_bin','amount_bin', 'state_bin']], prefix='dummy', drop_first=True)
##d2=pd.get_dummies(d2[['is_exciting','item_unit_price_bin','item_quantity_bin','vendor_name','project_resource_type']], prefix='dummy', drop_first=True)
#df2=pd.get_dummies(df[['is_exciting','payment_was_promo_matched_bin' ,'donation_included_optional_support_bin', 'is_teacher_acct_bin', 'payment_bin','amount_bin', 'state_bin']], prefix='dummy', drop_first=True)
#df2=pd.get_dummies(df[['is_exciting','honoree_bin','via_giving_page_bin','payment_was_promo_matched_bin','payment_included_web_purchased_gift_card_bin', 'payment_included_campaign_gift_card_bin', 'payment_included_acct_credit_bin', 'donation_included_optional_support_bin', 'is_teacher_acct_bin', 'amount_bin', 'payment_bin','state_bin']], prefix='dummy', drop_first=True)
##d2=pd.get_dummies(d2[['is_exciting','payment_was_promo_matched_bin' ,'donation_included_optional_support_bin', 'is_teacher_acct_bin', 'payment_bin','amount_bin', 'state_bin']], prefix='dummy', drop_first=True) 

#'fulfillment_labor_materials',
#       'total_price_excluding_optional_support','great_messages_proportion',

#d3=d3.drop('dummy_t', axis=1).columns
predictor_names = ['dummy_Extracurricular', 'dummy_Health', 'dummy_Others',
       'dummy_Science', 'dummy_Social Sciences',
       'dummy_Between 100 and 1000', 'dummy_Between 1000 and 10000',
       'dummy_Between 10000 and 20000', 'dummy_Between 20000 and 35000',
       'dummy_Between 10 and 100', 'dummy_Between 100 and 500',
       'dummy_Between 500 and 1000', 'dummy_above 1000', 'dummy_AKJ Books',
       'dummy_Abilitations', 'dummy_Amazon', 'dummy_Barnes and Noble',
       'dummy_Best Buy for Business', 'dummy_Blick Art Materials',
 #      'dummy_Brodhead Garrett - Removed as Vendor Summer '09',
       'dummy_CDW-G', 'dummy_Cannon Sports',
       'dummy_Carolina Biological Supply Company', 'dummy_Childcraft',
       'dummy_Classroom Direct', 'dummy_Delta Education',
       'dummy_Ellison Educational Equipment',
       'dummy_Encyclopedia Britannica', 'dummy_Frey Scientific',
       'dummy_Grainger',
 #      'dummy_Hammond & Stephens - Removed as Vendor Summer '09',
       'dummy_Highsmith', 'dummy_Kaplan Early Learning',
       'dummy_Kaplan Early Learning Company', 'dummy_Kid Carpet',
       'dummy_Lakeshore Learning Materials', 'dummy_LeapFrog SchoolHouse',
       'dummy_Learning ZoneXpress', 'dummy_MakerBot', 'dummy_Nasco',
       'dummy_Office Depot', 'dummy_Quill.com', 'dummy_Recorded Books',
       'dummy_Sargent-Welch', 'dummy_Sax Arts & Crafts',
#       'dummy_Sax Family & Consumer Science - Removed as Vendor Dec '09',
       'dummy_Scholastic Classroom Magazines',
 #      'dummy_School Outfitters - Removed As Vendor Summer '09',
       'dummy_School Specialty', 'dummy_Sportime',
 #      'dummy_Super Duper Publications', u'dummy_Teacher's School Supply',
 #      'dummy_Teachers' School Supply', u'dummy_Time for Kids',
       'dummy_Weekly Reader', 'dummy_Woodwind and Brasswind', 'dummy_Other',
       'dummy_Supplies', 'dummy_Technology', 'dummy_Trips',
       'dummy_Visitors', 'dummy_true_payment_was_promo_matched',
       'dummy_true_donation_included_optional_support',
       'dummy_true_is_teacher_acct', 'dummy_Online',
       'dummy_no cash received', 'dummy_Between 10 and 100',
       'dummy_above 100', 'dummy_EST', 'dummy_MST', 'dummy_PST',
       'dummy_other']
       
predictors=d3[predictor_names]
target=d3['dummy_t']
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=1)
logr_model = linear_model.LogisticRegression()
parametrized_classifier=logr_model.fit(X_train, y_train)
predictions=logr_model.predict(X_test) 
print(metrics.classification_report(y_test,predictions)) 
coeff_value_list=logr_model.coef_[0] 
print(metrics.confusion_matrix(y_test,predictions))

dt_model = tree.DecisionTreeClassifier()
#(criterion='entropy')
dt_model.fit(X_train, y_train) 
dt_model
dt_model.feature_importances_
predictions=dt_model.predict(X_test)
print(metrics.classification_report(y_test,predictions))
print(metrics.confusion_matrix(y_test,predictions))