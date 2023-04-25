#FILE INI UNTUK COBA ITERASIIN SATU2 APA YG SALAH DENGAN KODE

import pandas as pd
import string 
import re #regex library
import numpy as np
import nltk
from nltk.tokenize import word_tokenize 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from textblob import TextBlob, Word, Blobber
import matplotlib.pyplot as plt

import seaborn as sns

file_path = "Reviewgabunganno5.CSV" #csv

#membaca csvnya
df = pd.read_csv(file_path)
df.head()
#print(df.head)

#memastikan semuanya string
df['Content'] = df['Content'].astype(str)

#yang null dihapus
df = df[~df['Content'].isnull()]
print(len(df))

#menghilangkan simbol2  
def clean(txt):
    txt = txt.str.replace("()", "")
    txt = txt.str.replace('(<a).*(>).*()', '')
    txt = txt.str.replace('(&amp)', '')
    txt = txt.str.replace('(&gt)', '')
    txt = txt.str.replace('(&lt)', '')
    txt = txt.str.replace('(\xa0)', ' ')  
    return txt
df['Content'] = clean(df['Content'])
#print(df['Content'])

#bikin lowercase semua
df['review1'] = df['Content'].apply(lambda x: " ".join(x.lower() for x in x.split()))
#menghilangkan punctuation
df['review1'] = df['review1'].str.replace('[^\w\s]', '')


#hilangin stopwords 
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('indonesian')
df['review1'] = df['review1'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


#Remove rare words
freq = pd.Series(' '.join(df['review1']).split()).value_counts()
less_freq = list(freq[freq ==1].index)
#print(less_freq)

#df['review1'] = df['review1'].apply(lambda x: " ".join(x for x in x.split() if x not in less_freq))
#print(df['review1'])

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

#Ketahuan sastrawi yg bikin kodenya lama (cry)
df['review1'] = df['Content'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
print(df['review1'])
df['review1'] = df['review1'].str.replace('[^\w\s]', '')
df['review1'].head()

#DATA ANALITIK TIMEEEEE

#jumlah kata dalam review
df['review_len'] = df['Content'].astype(str).apply(len)
df['word_count'] = df['Content'].apply(lambda x: len(str(x).split()))

#polarity : menentukan sentimen text positif atau negatif (-1: negatif, 1 : positif, 0 : netral)
df['polarity'] = df['review1'].map(lambda text: TextBlob(text).sentiment.polarity) #cari mathematicsnya
print(df.head())

#Distribusi len, word count, polarity
df[["review_len", "word_count", "polarity"]].hist(bins=20, figsize=(15,20))

#Rating vs Polatitas
plt.figure(figsize = (10, 8))
sns.set_style('whitegrid')
sns.set(font_scale = 1.5)
sns.boxplot(x ='Rating', y ='polarity', data=df)
plt.xlabel("Rating")
plt.ylabel("Polatiry")
plt.title("Product Ratings vs Polarity")
plt.show()

#Polaritas ratings 
mean_pol = df.groupby('Rating')['polarity'].agg([np.mean])
mean_pol.columns = ['mean_polarity']
fig, ax = plt.subplots(figsize=(8, 6))
plt.bar(mean_pol.index, mean_pol.mean_polarity, width=0.3)
#plt.gca().set_xticklabels(mean_pol.index, fontdict={'size': 14})
for i in ax.patches:
    ax.text(i.get_x(), i.get_height()+0.01, str("{:.2f}".format(i.get_height())))
plt.title("Polarity of Ratings", fontsize=22)
plt.ylabel("Polarity", fontsize=16)
plt.xlabel("Rating", fontsize=16)
plt.ylim(0, 0.35)
plt.show()

#jumlah rating
plt.figure(figsize=(8, 6))
sns.countplot(x='Rating', data=df)
plt.xlabel("Rating")
plt.title("Number of data of each rating")
plt.show()

#rating vs pantang tulisan
plt.figure(figsize=(10, 6))
sns.pointplot(x='rating', y='review_len', data=df)
plt.xlabel("Rating")
plt.ylabel("Review Length")
plt.title("Product Rating vs Review Length")
plt.show()

#Top Product
product_pol = df.groupby('Product')['polarity'].agg([np.mean])
product_pol.columns = ['polarity']
product_pol = product_pol.sort_values('polarity', ascending=False)
product_pol = product_pol.head(20)
print(product_pol)











