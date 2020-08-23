# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 00:06:06 2020

@author: dmbes
"""


# pickle vectorizer and model itself.


#%%
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#%%
# Read in Data and convert to list
df = pd.read_csv('screen_name_train_200K.csv')
sn = df['screen_name'].tolist()
y = df['class'].tolist()

# Add domains now
other_df = pd.read_csv('dns.csv')

# Get all the urls to be lowercase.
other_df['QNAME_DNS'] = other_df['QNAME_DNS'].str.lower() 

# Cut out the ones that didn't resolve to anything. If the ISVALIDURL_DNS == FALSE, cut it.
other_df = other_df[~other_df['ISVALIDURL_DNS']==False]

# Cut out ones that end is .army.mil
clean1 = other_df[~other_df['QNAME_DNS'].str.contains(".army.mil", na=False)]

# Cut out ones that end in .mail.mil
clean2 = clean1[~clean1['QNAME_DNS'].str.endswith(".mail.mil", na=False)]

# Cut out ones that end in .disa.mil
clean3 = clean2[~clean2['QNAME_DNS'].str.endswith(".disa.mil", na=False)]

# Cut out reverse DNS lookups
clean4 = clean3[~clean3['QNAME_DNS'].str.endswith("in-addr.arpa", na=False)]

# Cut out the ones that start with www
clean5 = clean4[~clean4['QNAME_DNS'].str.startswith("www", na=False)]

# Cut out ones that end in mcaffee.com
clean6 = clean5[~clean5['QNAME_DNS'].str.endswith("mcafee.com", na=False)]

# Cut out ones that have "akamai" in them.
clean7 = clean6[~clean6['QNAME_DNS'].str.contains("akamai", na=False)]

# Get the number of times each domain name was queried.
queries_per_fqdn = clean7.groupby('QNAME_DNS').count()




# Get the number of times each domain was hit, sort them most to least queries
sorted_q_by_fqdn = queries_per_fqdn.sort_values("Id", ascending=False)
reduced = sorted_q_by_fqdn.reset_index()[['QNAME_DNS', 'Id']].copy()
#reduced.to_csv('remaining_fqdns.csv')

#%%
# Vectorize

# encode strings to see how many instances of each bigram in each screen name (or domain name)
vectorizer = CountVectorizer(analyzer='char', ngram_range=(2,2) )
X = vectorizer.fit_transform(sn)




domains = queries_per_fqdn.reset_index()['QNAME_DNS'].tolist()
X2 = vectorizer.transform(domains)
#print(X2.shape)

#%%
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.33, random_state=42)




#%%
# Convert to TFIDF
tfidf = TfidfTransformer().fit(X_train)
X_train = tfidf.transform(X_train)

#tfidf = TfidfTransformer().fit(X_train)
X2 = tfidf.transform(X2)

#%%
# Logistic Regression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)

#%%
'''
X_test = tfidf.transform(X_test)
pred = clf.predict(X_test)
'''
X_test = tfidf.transform(X_test)
pred = clf.predict(X2)
pred2 = clf.predict_proba(X2)


# put into panads dataframe, export to csv file.
df = pd.DataFrame(data=pred)
df2 = pd.DataFrame(data=pred2)
pd.set_option("display.max_rows", None, "display.max_columns", None)
#print(df[0])

reduced['PREDICTION'] = df[0]
reduced['PROBABILITY'] = df2[0]
mapping = {reduced.columns[1]:'NUM_QUERIES'}
reduced = reduced.rename(columns=mapping)
reduced['HIGHEST THREAT'] = reduced['QNAME_DNS'].loc[(reduced['PROBABILITY'] >= 0.95)]
reduced.to_csv('output.csv')
# get all the string with random probability > 95% (or whatever, play with threshold)

