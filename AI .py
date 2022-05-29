#Import các thư viện cần dùng
#pandas và numpy cho các quá trình xử lý data cơ bản
import pandas as pd
import numpy as np
#nltk để có thư viện xử lý ngôn ngữ tự nhiên
import nltk
#seaborn và matplotlib để trực quan hóa data
#Xử lý các url, hashtag bằng regex function
import re
import seaborn as sns
import matplotlib.pyplot as plt
#style để chọn style cho matplot, ở đây chúng ta dùng style
from matplotlib import style
style.use('ggplot')
#textblob để xử lý dữ liệu ký tự
from textblob import TextBlob
#tokenize để chia nhỏ văn bản thành các phần
from nltk.tokenize import word_tokenize
#PorterStemmer để biến đổi từ
from nltk.stem import PorterStemmer
#stop words để remove các stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
#WordCloud để trực quan hóa các từ được tìm kiếm nhiều sau khi chia thành positive, negative và neutral
from wordcloud import WordCloud

df = pd.read_csv('/content/vaccination_tweets.csv')

df.head()

df.info()

df.isnull().sum()

df.columns

#Chỉ dùng cột text nên ta sẽ xóa hết các cột còn lại
text_df = df.drop(['id', 'user_name', 'user_location', 'user_description', 'user_created',
       'user_followers', 'user_friends', 'user_favourites', 'user_verified',
       'date', 'hashtags', 'source', 'retweets', 'favorites',
       'is_retweet'], axis=1)
text_df.head(20)

#In 5 dòng đầu của dữ liệu trong cột text
print(text_df['text'].iloc[0],"\n")
print(text_df['text'].iloc[1],"\n")
print(text_df['text'].iloc[2],"\n")
print(text_df['text'].iloc[3],"\n")
print(text_df['text'].iloc[4],"\n")

text_df.info()

#Tạo một phương thức giúp xử lý đồng nhất dữ liệu
def data_processing(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+https\S+", '',text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','',text)
    text = re.sub(r'[^\w\s]','',text)
#Xóa đi các stop words
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)

#Xử lý dữ liệu bằng phương thức đã được thiết lập
nltk.download('punkt')
text_df.text = text_df['text'].apply(data_processing)
print(text_df.text)

#Xóa đi các dữ liệu đã bị lặp trong cột text (11020 GIẢM XUỐNG THÀNH 11013)
text_df = text_df.drop_duplicates('text')
print(text_df)

#Tạo phương thức biến đổi từ (stemming)
stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data

#Thực hiện stemming cho dữ liệu
text_df['text'] = text_df['text'].apply(lambda x: stemming(x))

text_df.head()

print(text_df['text'].iloc[0],"\n")
print(text_df['text'].iloc[1],"\n")
print(text_df['text'].iloc[2],"\n")
print(text_df['text'].iloc[3],"\n")
print(text_df['text'].iloc[4],"\n")


text_df.info()

#Thiết lập phương thức tính popularity của dữ liệu bằng TextBlob
def polarity(text):
    return TextBlob(text).sentiment.polarity

#Áp dụng phương thức popularity đã được thiết lập để xử lý dữ liệu
text_df['polarity'] = text_df['text'].apply(polarity)
print(text_df)

text_df.head(20)

#Tạo 1 phương thức để chia dữ liệu thành negative, positive và neutral
def sentiment(label):
    if label <0:
        return "Negative"
    elif label ==0:
        return "Neutral"
    elif label>0:
        return "Positive"

#Áp dụng phương thức sentiment cho bảng dữ liệu
text_df['sentiment'] = text_df['polarity'].apply(sentiment)

text_df.head(20)

#Sử dụng countplot để biểu diễn trực quan hóa sự phân bố của các mục sentiment
fig = plt.figure(figsize=(5,5))
sns.countplot(x='sentiment', data = text_df)

#Biểu diễn theo phần trăm
fig = plt.figure(figsize=(7,7))
colors = ("yellowgreen", "gold", "red")
wp = {'linewidth':2, 'edgecolor':"black"}
tags = text_df['sentiment'].value_counts()
explode = (0.1,0.1,0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors = colors,
         startangle=90, wedgeprops = wp, explode = explode, label='')
plt.title('Distribution of sentiments')

#Biểu diễn hình ảnh trực quan hóa các từ phổ biến trong các tweet có trong file dữ liệu theo mục positive, negative, neutral
pos_tweets = text_df[text_df.sentiment == 'Positive']
pos_tweets = pos_tweets.sort_values(['polarity'], ascending= False)
pos_tweets.head(20)

text = ' '.join([word for word in pos_tweets['text']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in positive tweets', fontsize=19)
plt.show()

neg_tweets = text_df[text_df.sentiment == 'Negative']
neg_tweets = neg_tweets.sort_values(['polarity'], ascending= False)
neg_tweets.head(20)

text = ' '.join([word for word in neg_tweets['text']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in negative tweets', fontsize=19)
plt.show()

neutral_tweets = text_df[text_df.sentiment == 'Neutral']
neutral_tweets = neutral_tweets.sort_values(['polarity'], ascending= False)
neutral_tweets.head(20)

text = ' '.join([word for word in neutral_tweets['text']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in neutral tweets', fontsize=19)
plt.show()