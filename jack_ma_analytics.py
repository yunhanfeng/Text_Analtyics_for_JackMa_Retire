import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 


# load in lm positive word
with open('LM_pos_words.txt', 'r') as f:
    pos = f.read().replace('\n',',')

tokenizer = RegexpTokenizer(r'\w+')
pos_token = tokenizer.tokenize(pos)
pos_token = [word.lower() for word in pos_token]

# load in lm negative word
with open('LM_neg_words.txt', 'r') as f:
    neg = f.read().replace('\n',',')
tokenizer = RegexpTokenizer(r'\w+')
neg_token = tokenizer.tokenize(neg)
neg_token = [word.lower() for word in neg_token]


# load in jack text
with open('Jack Ma_retire.txt', 'r',encoding="cp1252") as f:
    lines = f.read().replace('\n', '')

tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(lines)
len(tokens)

fdist = nltk.FreqDist(tokens)

for word ,frequency in fdist.most_common(10):
    print(u'{};{}'.format(word, frequency))
stop_word = []
for word, freqency in fdist.most_common(10):
    stop_word.append(word)
    
tokens_n_s = [word for word in tokens if word not in stop_word]
tokens_n_s = [word for word in tokens_n_s if not word.isnumeric()]
tokens_n_s = [word.lower() for word in tokens_n_s]

token_pos_ma = [word for word in tokens_n_s if word in pos_token]
token_neg_ma = [word for word in tokens_n_s if word in neg_token]

len(tokens_n_s)    # 758
len(token_pos_ma)  # 31
len(token_neg_ma)  # 7

# Jack Ma i
(31-7) / 758   ## 0.03166

# Jack Ma ii
7 / 758         ## 0.011535

#### negator not yet done
# y15 i adjusted
(21-3-(16+3)) / 1387  ## 0.009234

# y15 ii adjusted
(16+3) / 1387     ## 0.01370

# negator
negat = ['no', 'not', 'never']

token_not_15 = [word for word in tokens_n_s if word in negat]

# write a function of 3 words after a word
def nextword(target, source):
    alist = []
    for i, w in enumerate(source):
        if w == target:
            alist.append(source[i+1])
            alist.append(source[i+2])
            alist.append(source[i+3])
            return alist

# keep slicing the original text
first_not = nextword('not', tokens_n_s)
first_no = nextword('no', tokens_n_s)

tokens_n_s.index('not')
new_15_1 = tokens_n_s[535:]
second_not = nextword('not', new_15_1)

new_15_1.index('not')
new_15_2 = new_15_1[128:]
third_not = nextword('not', new_15_2)

new_15_2.index('not')
new_15_3 = new_15_2[263:]
fourth_not = nextword('not', new_15_3)

new_15_3.index('not')
new_15_4 = new_15_3[36:]
fifth_not = nextword('not', new_15_4)

new_15_4.index('not')
new_15_5 = new_15_4[212:]
sixth_not = nextword('not', new_15_5)

after_negator = first_not + first_no + second_not + third_not + fourth_not + fifth_not + sixth_not

after_negator_pos = [word for word in after_negator if word in pos_token]
################################################################################################################
# load in tesla private text
with open('Elon_private.txt', 'r',encoding="cp1252") as f:
    lines_17 = f.read().replace('\n', '')
    
tokenizer = RegexpTokenizer(r'\w+')
tokens17 = tokenizer.tokenize(lines_17)
len(tokens17)

fdist17 = nltk.FreqDist(tokens17)

for word ,frequency in fdist17.most_common(10):
    print(u'{};{}'.format(word, frequency))

stop_word17 = []

for word, freqency in fdist17.most_common(10):
    stop_word17.append(word)

tokens17_n_s = [word for word in tokens17 if word not in stop_word17]
tokens17_n_s = [word for word in tokens17_n_s if not word.isnumeric()]
tokens17_n_s = [word.lower() for word in tokens17_n_s]

token_pos_17 = [word for word in tokens17_n_s if word in pos_token]
token_neg_17 = [word for word in tokens17_n_s if word in neg_token]

len(tokens17_n_s)    # 460
len(token_pos_17)  # 10
len(token_neg_17)  # 4

# Elon_private i
(10-4) / 460    ## 0.01304

# Elon_private ii
4 / 460         ## 0.00869

# y17 sentiment is higher than y15, meaning the prospect in y17 is better, which
# is also shown in the stock price

tokens17_not = [word for word in tokens17_n_s if word in negat]

first_not7 = nextword('not', tokens17_n_s)
first_no7 = nextword('no', tokens17_n_s)

tokens17_n_s.index('not')
new_17_1 = tokens17_n_s[1037:]
second_not7 = nextword('not', new_17_1)

new_17_1.index('not')
new_17_2 = new_17_1[38:]
third_not7 = nextword('not', new_17_2)

new_17_2.index('not')
new_17_3 = new_17_2[173:]
fourth_not7 = nextword('not', new_17_3)
        
after_negator17 = first_not7 + first_no7 + second_not7 + third_not7 + fourth_not7
after_negator_pos17 = [word for word in after_negator17 if word in pos_token]

#############
# use the business dic
dict1 = {'Jack Ma': [0.03166, 0.011535], 'Elon Musk': [0.01304,0.00869]}
indexname = ['Sentiment','Negative']
df = pd.DataFrame(dict1, index = indexname)

pic1 = df.plot(kind = 'bar', rot = 45)

# use the build-in sentiment
dict2 = {'Jack Ma':[0.221, 0.007], 'Elon Mask': [0.149, 0.039]}
indexname2 = ['Positive','Negative']
df2 = pd.DataFrame(dict2, index = indexname2)

pic2 = df2.plot(kind = 'bar', rot =45)

#############
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

scores = sid.polarity_scores(lines)
for key in sorted(scores):
    print('{0}: {1}, '.format(key, scores[key]), end='')
    
scores_elon = sid.polarity_scores(lines_17)
for key in sorted(scores_elon):
    print('{0}: {1}, '.format(key, scores_elon[key]), end='')


#############
stopwords = set(STOPWORDS)
wordcloud_ma = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(lines) 

# cloud map - Jack ma retire
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud_ma) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 

wordcloud_elon = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(lines_17)

# cloud map - elon private
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud_elon) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() 



