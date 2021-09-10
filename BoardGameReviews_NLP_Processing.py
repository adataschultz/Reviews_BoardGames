# -*- coding: utf-8 -*-
'''
@author: aschu
'''
###############################################################################
############### Board Game Reviews NLP Preprocessing & EDA ####################
###############################################################################

print('\nBoard Game Reviews NLP Preprocessing & EDA') 
print('======================================================================')
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
import dask.dataframe as dd
import string, unicodedata
import contractions
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import spacy
from spacy_cld import LanguageDetector
import langid
import time
from nltk import FreqDist
from wordcloud import WordCloud,STOPWORDS

# Set path
path = r'D:\BoardGameReviews\Data'
os.chdir(path)

# Read data
df = pd.read_csv('BGR_NLP.csv', index_col=False, low_memory=False)

# Download word sets for cleaning
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

# Define class for cleaning the comments
class cleantext():
    
    def __init__(self, text = 'test'):
        self.text = text
        
    def remove_between_square_brackets(self):
        self.text = re.sub('\[[^]]*\]', '', self.text)
        return self

    def remove_numbers(self):
        self.text = re.sub('[-+]?[0-9]+', '', self.text)
        return self

    def replace_contractions(self):
        '''Replace contractions in string of text'''
        self.text = contractions.fix(self.text)
        return self
    
    def remove_special_characters(self, remove_digits=True):
        self.text = re.sub('[^a-zA-z0-9\s]','', self.text)
        return self
    
    def get_words(self):
        self.words = nltk.word_tokenize(self.text)
        return self

    def remove_non_ascii(self):
        '''Remove non-ASCII characters from list of tokenized words'''
        new_words = []
        for word in self.words:
            new_word = unicodedata.normalize('NFKD',
                                             word).encode('ascii',
                                                          'ignore').decode('utf-8',
                                                                           'ignore')
            new_words.append(new_word)
        self.words = new_words
        return self

    def to_lowercase(self):
        '''Convert all characters to lowercase from list of tokenized words'''
        new_words = []
        for word in self.words:
            new_word = word.lower()
            new_words.append(new_word)
        self.words = new_words
        return self

    def remove_stopwords(self):
        '''Remove stop words from list of tokenized words'''
        new_words = []
        for word in self.words:
            if word not in stopwords.words('english'):
                new_words.append(word)
        self.words = new_words
        return self

    def stem_words(self):
        '''Stem words in list of tokenized words'''
        stemmer = LancasterStemmer()
        stems = []
        for word in self.words:
            stem = stemmer.stem(word)
            stems.append(stem)
        self.words = stems
        return self

    def lemmatize_verbs(self):
        '''Lemmatize verbs in list of tokenized words'''
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in self.words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        self.words = lemmas
        return self
        
    def remove_punctuation(self):
        '''Remove punctuation from list of tokenized words'''
        new_words = []
        for word in self.words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        self.words = new_words
        return self

    def join_words(self):
        self.words = ' '.join(self.words)
        return self
    
    def do_all(self, text):
        
        self.text = text
        self = self.remove_numbers()
        self = self.replace_contractions()
        self = self.get_words()
        self = self.remove_punctuation()
        self = self.remove_non_ascii()
        self = self.remove_special_characters()
        self = self.remove_stopwords()
        self = self.stem_words()
        self = self.lemmatize_verbs()
        
        return self.words

# Shorter class name for following use
ct = cleantext()

# Define function for using for computations in Dask
def dask_this(df):
    res = df.apply(ct.do_all)
    return res

# Convert from pandas df to dask df
ddata = dd.from_pandas(df, npartitions=10)

# Run the cleaning with Dask workers 
print('Time for comments to be cleaned..')
search_time_start = time.time()
ddata['cleancomment'] = ddata['comment'].map_partitions(dask_this).compute(scheduler='processes',
                                                                           num_workers=50)
print('Finished cleaning comments in:', time.time() - search_time_start)

#Convert back to pandas df
df = ddata.compute()

del ddata

###############################################################################
###############################################################################
# Test two approaches for finding English words so other languages can be removed
# Approach 1 which yielded more English words
print('Time for cleaning function Languages-Spacy..')
search_time_start = time.time()

cleancomments= df['cleancomment']
languages_spacy = []

for e in cleancomments:
    doc = nlp(e)
    # cheking if the doc._.languages is not empty
    # then appending the first detected language in a list
    if(doc._.languages):
        languages_spacy.append(doc._.languages[0])
    # if it is empty, we append the list by unknown
    else:
        languages_spacy.append('unknown')
 
print('Finished cleaning function Languages-Spacy in:', time.time() - search_time_start)

df['languages'] = languages_spacy

print('Count of Languages in Comments using Approach 1')
print(df[['languages']].value_counts())

# Subset only comments in English
df = df.loc[df['languages'] == 'en']
print('Dimensions of data with only English comments')
print(df.shape) 

###############################################################################
# Approach 2 that yielded less English words
df1 = df

# Load English words from spacy
nlp = spacy.load('en')
language_detector = LanguageDetector()
nlp.add_pipe(language_detector)

# Convert from pandas df to dask df
ddata = dd.from_pandas(df1, npartitions=10)

print('Time for cleaning with Langid..')
search_time_start = time.time()
ddata['language'] = ddata['cleancomment'].apply(langid.classify,
                                                meta='object').compute(scheduler='processes',
                                                                       num_workers=50)
print('Finished cleaning with Langid in:', time.time() - search_time_start)
df1 = ddata.compute()

ids_langid  = df1['language']

# Generate label for language
langs = ids_langid.apply(lambda tuple: tuple[0], meta='object')

# Number unique language labels
print('Number of tagged languages (estimated):')
print(len(langs.unique()))

# Percent of the total dataset in English
print('Percent of data in English (estimated):')
print((sum(langs=='en')/len(langs))*100) 

# Remove since this approach yielded less observations containing English
del df1

###############################################################################
# Subset by Geekscore average into different sets
df1 = df[df.Rating_Group == 'Low']
df2 = df[df.Rating_Group == 'High']

# Use clean comment
df1_clean = df1['cleancomment']
df2_clean = df2['cleancomment']

# Defines a function for finding a list of words in the comments
def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

ratings_lower_words = get_all_words(df1_clean)
ratings_higher_words = get_all_words(df2_clean)

# Find 10 most common words 
freq_dist_higher = FreqDist(ratings_higher_words)
print(freq_dist_higher.most_common(10))

# Find 10 most common words 
freq_dist_lower = FreqDist(ratings_lower_words)
print(freq_dist_lower.most_common(10))

###############################################################################
# Word cloud visualization
def wordcloud_draw(data, color = 'black'):
    words = ' '.join(data)
    cleaned_word = ' '.join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
print('Higher Rating Games')
wordcloud_draw(df2_clean,'white')

print('Lower Rated Games')
wordcloud_draw(df1_clean, 'white')

###############################################################################
# Write to csv for modeling
df.to_csv('cleanedcomments_nlp.csv')

###############################################################################



