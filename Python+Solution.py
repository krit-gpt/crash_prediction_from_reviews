
# coding: utf-8

# In[186]:


import pandas as pd
import string,re
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from Class_replace_impute_encode import ReplaceImputeEncode
from Class_tree import DecisionTree
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import sys


# ### Loading the data

# In[187]:


df = pd.read_excel("HondaComplaints.xlsx")
sw = pd.read_excel("afinn_sentiment_words.xlsx")


# ### Deleting the duplicate entries

# In[188]:


df['description']= df['description'].str.replace('AIR BAG', 'AIRBAG')
df= df.drop_duplicates('description')


# In[189]:


df.head()


# In[190]:


sentiment_dic = {}
for i in range(len(sw)):
    sentiment_dic[sw.iloc[i][0]] = sw.iloc[i][1]


# ### Data Preprocessing
# 
# - Creating a list of Synonyms and Custom Words to make the analysis better and more effective.
# - Tagging the words as Parts of Speech - Nouns, Verbs, etc.
# - Stemmed the words.

# In[191]:


def my_analyzer(s):
    # Synonym List
    syns = {'irrresponsible' :'irresponsible',
              'wont':'would not', 'cant':'can not', 'cannot':'can not', \
              'couldnt':'could not', 'shouldnt':'should not', \
              'wouldnt':'would not', 'anticipate':'anticipate', 'airbag':'airbags'}
    
    # Preprocess String s  
    s= s.lower()
    s = s.replace('-', ' ')
    s = s.replace(',', '. ')
    s = s.replace('_', ' ')
    s = s.replace("'nt", " not")
    s = s.replace("n't", " not")
    # Tokenize 
    tokens = word_tokenize(s)
    tokens = [word.replace(',','') for word in tokens ]
    tokens = [word for word in tokens if ('*' not in word) and               ("''" != word) and ("``" != word) and               (word!='description') and (word !='dtype')               and (word != 'object') and (word!="'s")]
    
    # Map synonyms
    for i in range(len(tokens)):
        if tokens[i] in syns:
            tokens[i] = syns[tokens[i]]
            
    # Remove stop words
    stemmer = SnowballStemmer("english")
    punctuation = list(string.punctuation)+['..', '...']
    pronouns = ['i', 'he', 'she', 'it', 'him', 'they', 'we', 'us', 'them']
    custom_stop= [["quot", "amp", "could", "also", "even", "really", "one",                     "would", "get", "getting", "go", "going", "..", "...",                     "us", "area", "vegas","oct", "place", "want", "get",                     "take", "end", "la", "gal", "get", "next", "though",                     "non", "seem", "use", "sep", "w/", "jul", "get", "go","almost", "say", "tell",                    "own", "car", "xxx", "son"  ]]
    stop = stopwords.words('english') + punctuation + pronouns+ custom_stop
    filtered_terms = [word for word in tokens if (word not in stop) and                   (len(word)>2) and (not word.replace('.','',1).isdigit())                   and (not word.replace("'",'',2).isdigit()) and re.sub('[^a-z\s]',' ', str(word))]
    # Lemmatization & Stemming - Stemming with WordNet POS
    # Since lemmatization requires POS need to set POS
    tagged_words = pos_tag(filtered_terms, lang='eng')
    # Stemming with for terms without WordNet POS
    
    wn_tags = {'N':wn.NOUN, 'J':wn.ADJ, 'V':wn.VERB, 'R':wn.ADV}
    wnl = WordNetLemmatizer()
    stemmed_tokens = []
    for tagged_token in tagged_words:
        term = tagged_token[0]
        pos  = tagged_token[1]
        pos  = pos[0]
        try:
            pos   = wn_tags[pos]
            stemmed_tokens.append(wnl.lemmatize(term, pos=pos))
        except:
            stemmed_tokens.append(stemmer.stem(term))
    return stemmed_tokens


# In[192]:


def my_preprocessor(s):
    # Preprocess String s
    s = s.lower()
    # Replace special characters with spaces
    s = s.replace('-', ' ')
    s = s.replace('_', ' ')
    s = s.replace(',', '. ')
    # Replace not contraction with not
    s = s.replace("'nt", " not")
    s = s.replace("n't", " not")
    return(s)


# ### Topic Modeling
# 
# The below sections performs Topic Modeling to segregate each review into clusters.

# In[193]:


def display_topics(topic_vectorizer, terms, n_terms=15, word_cloud=False, mask=None):
    for topic_idx, topic in enumerate(topic_vectorizer):
        message = "Topic #%d: " %(topic_idx+1)
        print(message)
        abs_topic = abs(topic)
        topic_terms_sorted =         [[terms[i], topic[i]]             for i in abs_topic.argsort()[:-n_terms - 1:-1]]
        k = 5
        n = int(n_terms/k)
        m = n_terms - k*n
        for j in range(n):
            l = k*j
            message = ''
            for i in range(k):
                if topic_terms_sorted[i+l][1]>0:
                    word = "+"+topic_terms_sorted[i+l][0]
                else:
                    word = "-"+topic_terms_sorted[i+l][0]
                message += '{:<15s}'.format(word)
            print(message)
        
        if m> 0:
            l = k*n
            message = ''
            for i in range(m):
                if topic_terms_sorted[i+l][1]>0:
                    word = "+"+topic_terms_sorted[i+l][0]
                else:
                    word = "-"+topic_terms_sorted[i+l][0]
                message += '{:<15s}'.format(word)
            print(message)
    return


# In[194]:


def term_dic(tf, terms, scores=None):
    td = {}
    for i in range(tf.shape[0]):
    # Iterate over the terms with nonzero scores
        term_list = tf[i].nonzero()[1]
        if len(term_list)>0:
            if scores==None:
                for t in np.nditer(term_list):
                    if td.get(terms[t]) == None:
                        td[terms[t]] = tf[i,t]
                    else:
                        td[terms[t]] += tf[i,t]
            else:
                for t in np.nditer(term_list):
                    score = scores.get(terms[t])
                    if score != None:
                    # Found Sentiment Word
                        score_weight = abs(scores[terms[t]])
                        if td.get(terms[t]) == None:
                            td[terms[t]] = tf[i,t] * score_weight
                        else:
                            td[terms[t]] += tf[i,t] * score_weight
    return td


# In[195]:


n_reviews = len(df['description'])
cv = CountVectorizer(max_df=0.7, min_df=4, max_features=None,analyzer=my_analyzer)
tf = cv.fit_transform(df['description'])
terms = cv.get_feature_names()
print('{:.<22s}{:>6d}'.format("Number of Reviews", n_reviews))
print('{:.<22s}{:>6d}'.format("Number of Terms", len(terms)))


# ### Calculating Term Frequency for each word in the corpus

# In[196]:


td = term_dic(tf, terms)
print("The Corpus contains a total of ", len(td), " unique terms.")
print("The total number of terms in the Corpus is", sum(td.values()))
term_sums = tf.sum(axis=0)
term_counts = []
for i in range(len(terms)):
    term_counts.append([terms[i], term_sums[0,i]])
def sortSecond(e):
    return e[1]
term_counts.sort(key=sortSecond, reverse=True)
print("\nTerms with Highest Frequency:")
for i in range(10):
    print('{:<15s}{:>5f}'.format(term_counts[i][0], term_counts[i][1]))


# ### Calculating the TF-IDF score for each word in the corpus

# In[197]:


print("\nConducting Term/Frequency Matrix using TF-IDF")
# Default for norm is 'l2', use norm=None to supress
tfidf_vect = TfidfTransformer(norm=None, use_idf=True) #set norm=None
# tf matrix is (n_reviews)x(m_features
tf = tfidf_vect.fit_transform(tf)
term_idf_sums = tf.sum(axis=0)
term_idf_scores = []
for i in range(len(terms)):
    term_idf_scores.append([terms[i], term_idf_sums[0,i]])
print("The Term/Frequency matrix has", tf.shape[0], " rows, and",    tf.shape[1], " columns.")
print("The Term list has", len(terms), " terms.")
term_idf_scores.sort(key=sortSecond, reverse=True)
print("\nTerms with Highest TF-IDF Scores:")
for i in range(10):
    j = i
    print('{:<15s}{:>8.2f}'.format(term_idf_scores[j][0],     term_idf_scores[j][1]))


# ### Performing Topic Clustering

# In[198]:


n_topics = 7

uv = LatentDirichletAllocation(n_components=n_topics, learning_method='online', random_state=12345)
U = uv.fit_transform(tf)

print("\n********** GENERATED TOPICS **********")

display_topics(uv.components_, terms, n_terms=15)

# Store topic selection for each doc in topics[]
topics = [0] * n_reviews
for i in range(n_reviews):
    max = abs(U[i][0])
    topics[i] = 0
    for j in range(n_topics):
        x = abs(U[i][j])
        if x > max:
            max = x
            topics[i] = j
U_rev_scores = []
for i in range(n_reviews):
    u = [0] * (n_topics+1)
    u[0] = topics[i]
    for j in range(n_topics):
        u[j+1] = U[i][j]
    U_rev_scores.append(u)
rev_scores = U_rev_scores

# Integrate Topic Scores into Main Data Frame (df)
cols = ["topic"]
for i in range(n_topics):
    s = "T"+str(i+1)
    cols.append(s)
df_rev = pd.DataFrame.from_records(rev_scores, columns=cols)
print(df_rev.head())
print(df.head())


# ### Managing the Topic Clusters and the topic probabilities into the dataset, to be used as features later down the road.

# In[199]:


df=df.join(df_rev)


# In[201]:





# In[202]:


df.head()


# ### Topic Distribution

# In[203]:


print(" TOPIC DISTRIBUTION")
print('{:<6s}{:>4s}{:>12s}'.format("TOPIC", "N", "PERCENT"))
print("----------------------")
topic_counts = df['topic'].value_counts(sort=False)
for i in range(len(topic_counts)):
    percent = 100*topic_counts[i]/n_reviews
    print('{:>3d}{:>8d}{:>9.1f}%'.format((i+1), topic_counts[i], percent))


# In[204]:


df['abs'].value_counts()


# ## Sentiment Analysis

# In[205]:


cv = CountVectorizer(max_df=0.95, min_df=1, max_features=None, preprocessor=my_preprocessor, ngram_range=(1,2))
tf = cv.fit_transform(df['description'])
s_terms = cv.get_feature_names()
n_reviews = tf.shape[0]
n_terms = tf.shape[1]
print('{:.<22s}{:>6d}'.format("Number of Reviews", n_reviews))
print('{:.<22s}{:>6d}'.format("Number of Terms", n_terms))


# In[206]:


def sent_score(tf,df, text): 

    
    min_sentiment = +5
    max_sentiment = -5
    avg_sentiment, min, max = 0,0,0
    min_list, max_list = [],[]
    sentiment_score = [0]*n_reviews
    for i in range(n_reviews):
            # Iterate over the terms with nonzero scores
                n_sw = 0
                term_list = tf[i].nonzero()[1]
                if len(term_list)>0:
                    for t in np.nditer(term_list):
                        score = sentiment_dic.get(s_terms[t])
                        if score != None:
                            sentiment_score[i] += score * tf[i,t]
                            n_sw += tf[i,t]
                if n_sw>0:
                    sentiment_score[i] = sentiment_score[i]/n_sw
                if sentiment_score[i]==max_sentiment and n_sw>3:
                    max_list.append(i)
                if sentiment_score[i]>max_sentiment and n_sw>3:
                    max_sentiment=sentiment_score[i]
                    max = i
                    max_list = [i]
                if sentiment_score[i]==min_sentiment and n_sw>3:
                    min_list.append(i)
                if sentiment_score[i]<min_sentiment and n_sw>3:
                    min_sentiment=sentiment_score[i]
                    min = i
                    min_list = [i]
                avg_sentiment += sentiment_score[i]
    avg_sentiment = avg_sentiment/n_reviews
    print( "Overall Average Sentiment: ", avg_sentiment)
    df['Sentiment_Score']= sentiment_score
    return


# In[207]:


sent_score(tf,df,'description')


# In[223]:


df['Year'].value_counts()


# ## Decision Tree

# In[209]:


# import os
# os.chdir('C:\Texas A&M Spring Semester\STAT 656')


# In[211]:


## Since all these variables are not necessary we can drop them.
df.drop(['State','NhtsaID','description'], axis=1,inplace=True)


# In[212]:


##Since many scikit learn methods will not accept string values 
##The following values should be converted to numeric
cat_map={'Y':1, 'N':2}
cat_map3={'HONDA':1, 'ACURA':2}
cat_map4={'TL':1,'ODYSSEY':2,'CR-V':3,'CL':4,'CIVIC':5,'ACCORD':6}
df['Model']=df['Model'].map(cat_map4)
df['crash']=df['crash'].map(cat_map)
df['cruise']=df['cruise'].map(cat_map)
df['abs']=df['abs'].map(cat_map)
df['Make']=df['Make'].map(cat_map3)


# In[224]:


attribute_map = {
    'Year':[2,(2001,2002,2003),[0,0]],
    'Make':[2,(1,2),[0,0]],
    'Model':[2,(1,2,3,4,5,6),[0,0]],
    'crash':[1,(1,2),[0,0]],
    'cruise':[1,(1,2),[0,0]],
    'abs':[1,(1,2),[0,0]],
    'mileage':[0,(1, 200000),[0,0]],
    'mph':[0,(0, 80),[0,0]],
    'topic':[2,(0.0,1.0,2.0,3.0,4.0,5.0,6.0),[0,0]],
    'T1':[0,(-1e+8,1e+8),[0,0]],
    'T2':[0,(-1e+8,1e+8),[0,0]],
    'T3':[0,(-1e+8,1e+8),[0,0]],
    'T4':[0,(-1e+8,1e+8),[0,0]],
    'T5':[0,(-1e+8,1e+8),[0,0]],
    'T6':[0,(-1e+8,1e+8),[0,0]],
    'T7':[0,(-1e+8,1e+8),[0,0]]
}


# In[228]:


##Data Preprocessing Starts
feature_names=np.asarray(df.columns)
initial_missing=df.isnull().sum()
print('The number of observation in the new dataset are :-',df.shape[0])
#new_df.describe()
# Initialize number missing in attribute_map
for k,v in attribute_map.items():
    for feature in feature_names:
        if feature==k:
            v[2][0] = initial_missing[feature]
            break


# In[229]:


#Initializing outliers and setting all outliers as missing value
for i in (df.index):
    # For each observations, Iterate over all attributes.
    # k is the attributes name and v is its metadata
    for k, v in attribute_map.items():
        # Check if the data is missing
             
        if v[0]==0: # Interval Attribute
            l_limit = v[1][0] # get lower limit from metadata
            u_limit = v[1][1] # get upper limit from metadata
            # If the observation is outside the limits, its an outlier
            if df.loc[i, k]>u_limit or df.loc[i,k]<l_limit:
                v[2][1] += 1        # Number of outliers in metadata
                df.loc[i,k] = None  # Set outlier to missing
                
        else: # Categorical Attribute or Other
            
            in_cat = False
            # Iterate over the allowed categories for this attribute
            for cat in v[1]:
                if df.loc[i,k]==cat: # Found the category, not outlier
                    in_cat=True
            if in_cat==False:  # Did not find this category in the metadata
                df.loc[i,k] = None  # This data is not recognized, its an outlier
                v[2][1] += 1        # Increment the outlier counter for this attribute


# In[231]:


print("\nNumber of missing values and outliers by attribute:")
feature_names = np.array(df.columns.values)
for k,v in attribute_map.items():
    print(k+":\t%i missing" %v[2][0]+ "  %i outlier(s)" %v[2][1])


# In[232]:


# Each of these lists will contain the names of the attributes in their level
interval_attributes = []
nominal_attributes  = []
binary_attributes   = []
onehot_attributes   = []
# Iterate over the data dictionary
for k,v in attribute_map.items():
    if v[0]==0:
        interval_attributes.append(k)
    else:
        if v[0]==1:
            binary_attributes.append(k)
        else:
            nominal_attributes.append(k)
            for i in range(len(v[1])):
                str = k+("%i" %i)
                onehot_attributes.append(str)


# In[233]:


n_interval = len(interval_attributes)
n_binary   = len(binary_attributes)
n_nominal  = len(nominal_attributes)
n_onehot   = len(onehot_attributes)
print("\nFound %i Interval Attributes, " %n_interval,       "%i Binary," %n_binary,        "and %i Nominal Attribute\n" %n_nominal)


# ### Data Preprocessing
# 
# - Finding the missing values.
# - Imputing the missing values.
# 

# In[234]:



##Filling missing values, Imputation of the dataframe
# Assigning the nominal and binary data from the dataframe into a numpy array
from sklearn import preprocessing
#print("Original DataFrame:\n", df[0:5])
# Assigning the interval data from the dataframe into a numpy array
interval_data = df.as_matrix(columns=interval_attributes)
# Creating the Imputer for the Interval Data
interval_imputer = preprocessing.Imputer(strategy='mean')
# Imputing the missing values in the Interval data
imputed_interval_data = interval_imputer.fit_transform(interval_data)
nominal_data = df.as_matrix(columns=nominal_attributes)
binary_data  = df.as_matrix(columns=binary_attributes)
# Creating Imputer for Categorical Data
cat_imputer = preprocessing.Imputer(strategy='most_frequent')
# Imputing the missing values in the Categorical Data
imputed_nominal_data = cat_imputer.fit_transform(nominal_data)
imputed_binary_data  = cat_imputer.fit_transform(binary_data)


# In[235]:


# Bring Interval and Categorial Data Together
# The Imputed Data
data_array= np.hstack((imputed_interval_data, imputed_binary_data,                        imputed_nominal_data))
col = []
for i in range(n_interval):
    col.append(interval_attributes[i])
for i in range(n_binary):
    col.append(binary_attributes[i])
for i in range(n_nominal):
    col.append(nominal_attributes[i])
df_imputed = pd.DataFrame(data_array,columns=col)
#print("\nImputed DataFrame:\n", df_imputed[0:5])
df_imputed.describe()


# ### Building one hot vectors

# In[236]:


##Nominal Attributes
##Creating an instance of the OneHotEncoder & Selecting Attributes
onehot = preprocessing.OneHotEncoder()
hot_array = onehot.fit_transform(imputed_nominal_data).toarray()


# In[237]:


# I have not scaled the interval data as the range for interval attributes is not that big.
# The Imputed and Encoded Data

data_array = np.hstack((imputed_interval_data, imputed_binary_data, hot_array))
#col = (interval_attributes, cat_attributes)
col = []
for i in range(n_interval):
    col.append(interval_attributes[i])
for i in range(n_binary):
    col.append(binary_attributes[i])
for i in range(n_onehot):
    col.append(onehot_attributes[i])
df_imputed_scaled = pd.DataFrame(data_array,columns=col)
df_imputed_scaled.columns


# In[239]:


df_imputed_scaled.describe()


# ### Cross Validating to find the best Decision Tree Depth

# In[242]:


varlist = ['crash', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
X = df_imputed_scaled.drop(varlist, axis=1)
y = df_imputed_scaled['crash'] 


# In[245]:


max_depth=[5,6,7,8,10,12,15,20,25]
for i in max_depth:
    dtc = DecisionTreeClassifier(criterion='gini', max_depth=i,     min_samples_split=5, min_samples_leaf=5)
    dtc = dtc.fit(X,y)
    score_list = ['accuracy', 'recall', 'precision', 'f1']
    mean_score = []
    std_score = []
    print("For max_depth=",i)
    print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
    for s in score_list:
        dtc_10 = cross_val_score(dtc, X, y, scoring=s, cv=10)
        mean = dtc_10.mean()
        std = dtc_10.std()
        mean_score.append(mean)
        std_score.append(std)
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))


# In[246]:


print("Max depth for the Decision Tree is 6")


# In[248]:


X_train, X_validate, y_train, y_validate = train_test_split(X,y,test_size = 0.3, random_state=7)
score_list = ['accuracy', 'recall', 'precision', 'f1']


# In[249]:


dtc_train = DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_split=5, min_samples_leaf=5,
class_weight='balanced').fit(X_train, y_train)
print("\nTable of the metrics for 70/30 split")
DecisionTree.display_binary_split_metrics(dtc_train, X_train,y_train,X_validate, y_validate)


# In[250]:


def dtc_graph():
    dot_data = tree.export_graphviz(dtc_train, out_file=None,
    feature_names=list(X.columns),
    class_names=['0','1'],
    filled=True, rounded=True,
    special_characters=True)
    graph = graphviz.Source(dot_data)
    return(graph)
dtc_graph()


# In[ ]:


##Splitting data
X_train, X_validate, y_train, y_validate = train_test_split(X,y,test_size = 0.3, random_state=7)
score_list = ['accuracy', 'recall', 'precision', 'f1']


# ## Web Scraping

# In[26]:


import re
import pandas as pd
import requests
import newspaper
from newspaper import Article
from newsapi import NewsApiClient # Needed for using API Feed
from time import time


# In[27]:


agency_urls = {
'huffington': 'http://huffingtonpost.com',
'reuters': 'http://www.reuters.com',
'cbs-news': 'http://www.cbsnews.com',
'usa-today': 'http://usatoday.com',
'cnn': 'http://cnn.com',
'npr': 'http://www.npr.org',
'wsj': 'http://wsj.com',
'fox': 'http://www.foxnews.com',
'abc': 'http://abc.com',
'abc-news': 'http://abcnews.com',
'abcgonews': 'http://abcnews.go.com',
'nyt': 'http://nytimes.com',
'washington-post': 'http://washingtonpost.com',
'us-news': 'http://www.usnews.com',
'msn': 'http://msn.com',
'pbs': 'http://www.pbs.org',
'nbc-news': 'http://www.nbcnews.com',
'enquirer': 'http://www.nationalenquirer.com',
'la-times': 'http://www.latimes.com'
}


# In[28]:


def clean_html(html):
# First we remove inline JavaScript/CSS:
    pg = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
    # Then we remove html comments. This has to be done before removing regular
    # tags since comments can contain '>' characters.
    pg = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", pg)
    # Next we can remove the remaining tags:
    pg = re.sub(r"(?s)<.*?>", " ", pg)
    # Finally, we deal with whitespace
    pg = re.sub(r"&nbsp;", " ", pg)
    pg = re.sub(r"&rsquo;", "'", pg)
    pg = re.sub(r"&ldquo;", '"', pg)
    pg = re.sub(r"&rdquo;", '"', pg)
    pg = re.sub(r"\n", " ", pg)
    pg = re.sub(r"\t", " ", pg)
    pg = re.sub(r" ", " ", pg)
    pg = re.sub(r" ", " ", pg)
    pg = re.sub(r" ", " ", pg)
    return pg.strip()


# In[29]:


def newsapi_get_urls(search_words, agency_urls):
    if len(search_words)==0 or agency_urls==None:
        return None
    print("Searching agencies for pages containing:", search_words)
    # This is my API key, each user must request their own
    # API key from https://newsapi.org/account
    api = NewsApiClient(api_key='6f174feb5d05447d920d538d45718afa')
    api_urls = []
    # Iterate over agencies and search words to pull more url's
    # Limited to 1,000 requests/day - Likely to be exceeded
    for agency in agency_urls:
        domain = agency_urls[agency].replace("http://", "")
        print(agency, domain)
        for word in search_words:
            try:
                articles = api.get_everything(q=word, language='en',                sources=agency, domains=domain)
            except:
                print("--->Unable to pull news from:", agency, "for", word)
                continue
# Pull the URL from these articles (limited to 20)
            d = articles['articles']
            for i in range(len(d)):
                url = d[i]['url']
                api_urls.append([agency, word, url])


    df_urls = pd.DataFrame(api_urls, columns=['agency', 'word', 'url'])
    n_total = len(df_urls)
    # Remove duplicates
    df_urls = df_urls.drop_duplicates('url')
    n_unique = len(df_urls)
    print("\nFound a total of", n_total, " URLs, of which", n_unique,    " were unique.")
    return df_urls


# In[30]:


def request_pages(df_urls):
    web_pages = []
    for i in range(len(df_urls)):
        u = df_urls.iloc[i]
        url = u[2]
        short_url = url[0:50]
        short_url = short_url.replace("https//", "")
        short_url = short_url.replace("http//", "")
        n = 0
        # Allow for a maximum of 5 download failures
        stop_sec=3 # Initial max wait time in seconds
        while n<3:
            try:
                r = requests.get(url, timeout=(stop_sec))
                if r.status_code == 408:
                    print("-->HTML ERROR 408", short_url)
                    raise ValueError()
                if r.status_code == 200:
                    print("Obtained: "+short_url)
                else:
                    print("-->Web page: "+short_url+" status code:",                     r.status_code)
                n=99
                continue # Skip this pag
            except:
                n += 1
                # Timeout waiting for download
                t0 = time()
                tlapse = 0
                print("Waiting", stop_sec, "sec")
                while tlapse<stop_sec:
                    tlapse = time()-t0
        if n != 99:
            # download failed skip this page
            continue
            # Page obtained successfully
        html_page = r.text
        page_text = clean_html(html_page)
        web_pages.append([url, page_text])
    df_www = pd.DataFrame(web_pages, columns=['url', 'text'])
    n_total = len(df_urls)
    # Remove duplicates
    df_www = df_www.drop_duplicates('url')
    n_unique = len(df_urls)
    print("Found a total of", n_total, " web pages, of which", n_unique,    " were unique.")
    return df_www


# In[31]:


search_words = ['takata']
df_urls = newsapi_get_urls(search_words, agency_urls)
print("Total Articles:", df_urls.shape[0])


# In[216]:


df_www = request_pages(df_urls)


# In[217]:


df_www= df_www.drop_duplicates('text')


# In[218]:


df_www.to_csv('df_www.csv')


# In[222]:


df_www['text']= df_www['text'].str.replace("air bag", "airbag")
df_www['text']= df_www['text'].str.replace("air bags", "airbag")
df_www['text']= df_www['text'].str.replace("airbags", "airbag")
df_www['text']= df_www['text'].str.replace("Air bag", "airbag")
df_www['text']= df_www['text'].str.replace("Air bags", "airbag")


# In[231]:


df_www.head()


# In[223]:


for i in range(df_www.shape[0]):
    short_url = df_www.iloc[i]['url']
    short_url = short_url.replace("https://", "")
    short_url = short_url.replace("http://", "")
    short_url = short_url[0:60]
    page_char = len(df_www.iloc[i]['text'])
    print("{:<60s}{:>10d} Characters".format(short_url, page_char))


# In[224]:


n_reviews = len(df_www['text'])
cv = CountVectorizer(max_df=0.7, min_df=4, max_features=None,analyzer=my_analyzer)
tf = cv.fit_transform(df_www['text'])
terms = cv.get_feature_names()
print('{:.<22s}{:>6d}'.format("Number of Articles", n_reviews))
print('{:.<22s}{:>6d}'.format("Number of Terms", len(terms)))


# In[225]:


td = term_dic(tf, terms)
print("The Corpus contains a total of ", len(td), " unique terms.")
print("The total number of terms in the Corpus is", sum(td.values()))
term_sums = tf.sum(axis=0)
term_counts = []
for i in range(len(terms)):
    term_counts.append([terms[i], term_sums[0,i]])
def sortSecond(e):
    return e[1]
term_counts.sort(key=sortSecond, reverse=True)
print("\nTerms with Highest Frequency:")
for i in range(10):
    print('{:<15s}{:>5f}'.format(term_counts[i][0], term_counts[i][1]))


# In[226]:


print("\nConducting Term/Frequency Matrix using TF-IDF")
# Default for norm is 'l2', use norm=None to supress
tfidf_vect = TfidfTransformer(norm=None, use_idf=True) #set norm=None
# tf matrix is (n_reviews)x(m_features
tf = tfidf_vect.fit_transform(tf)
term_idf_sums = tf.sum(axis=0)
term_idf_scores = []
for i in range(len(terms)):
    term_idf_scores.append([terms[i], term_idf_sums[0,i]])
print("The Term/Frequency matrix has", tf.shape[0], " rows, and",    tf.shape[1], " columns.")
print("The Term list has", len(terms), " terms.")
term_idf_scores.sort(key=sortSecond, reverse=True)
print("\nTerms with Highest TF-IDF Scores:")
for i in range(10):
    j = i
    print('{:<15s}{:>8.2f}'.format(term_idf_scores[j][0],     term_idf_scores[j][1]))


# In[227]:


n_topics = 7

uv = LatentDirichletAllocation(n_components=n_topics, learning_method='online', random_state=12345)
U = uv.fit_transform(tf)

print("\n********** GENERATED TOPICS **********")

display_topics(uv.components_, terms, n_terms=15)
# Store topic selection for each doc in topics[]
topics = [0] * n_reviews
for i in range(n_reviews):
    max = abs(U[i][0])
    topics[i] = 0
    for j in range(n_topics):
        x = abs(U[i][j])
        if x > max:
            max = x
            topics[i] = j
U_rev_scores = []
for i in range(n_reviews):
    u = [0] * (n_topics+1)
    u[0] = topics[i]
    for j in range(n_topics):
        u[j+1] = U[i][j]
    U_rev_scores.append(u)
rev_scores = U_rev_scores
# Integrate Topic Scores into Main Data Frame (df)
cols = ["topic"]
for i in range(n_topics):
    s = "T"+str(i+1)
    cols.append(s)
df_rev = pd.DataFrame.from_records(rev_scores, columns=cols)
df_www = df_www.join(df_rev)


# In[228]:


df_www.head()


# ## Sentiment Analysis

# In[229]:


sent_score(tf,df_www, 'text')


# In[230]:


df_www.head()

