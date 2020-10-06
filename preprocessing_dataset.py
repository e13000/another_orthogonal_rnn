import json
import nltk
from pprint import pprint as pp
from math import ceil

raw_data_filename = 'science_titles.json'

with open(raw_data_filename, mode='r', encoding='utf-8') as f:
    dataset = json.load(f)

# Get titles of posts which have more than 10 upvotes and 10 comments
news_titles = [post[0].lower() for (_, post) in dataset.items()]
news_scores = [post[2] for (_, post) in dataset.items()]
news_ncomments = [post[3] for (_, post) in dataset.items()]

# We need to mark the starting point and ending point of a title
title_start_tk = 'TITLE_START_TOKEN'
title_end_tk = 'TITLE_END_TOKEN'
news_titles = [title_start_tk+' '+str(title)+' '+ title_end_tk for title in news_titles]

tokenized_titles = [nltk.word_tokenize(title) for title in news_titles]
# tokenized_titles is a list of a list, let's convert it to a list of words
words = [word for tokenized_title in tokenized_titles for word in tokenized_title]

words_fdist = nltk.FreqDist(words)
# Number of words that occur at least FREQ_THRESHOLD times

FREQ_THRESHOLD = 200
vocab_size = ceil(len([word for (word, freq) in words_fdist.items() if freq >= FREQ_THRESHOLD]) / 100) * 100 - 1
# vocab_size = 7999
frequent_words = words_fdist.most_common(n=vocab_size)
vocabulary = [v[0] for v in frequent_words]

rare_word_tk = 'RARE_WORD_TOKEN'
# Replace words that occur less than FREQ_THRESHOLD by RARE_WORD_TOKEN
filtered_tokenized_titles = [[word if word in vocabulary else rare_word_tk for word in tokenized_title]
                             for tokenized_title in tokenized_titles]

vocabulary.insert(0, rare_word_tk) # Add rare word token to the vocabulary

# index to word, i.e., decoding a number to a word
i2w = vocabulary
w2i = {word: index for (index, word) in enumerate(i2w)}
encoded_titles = [[w2i[word] for word in filtered_tokenized_title]
                  for filtered_tokenized_title in filtered_tokenized_titles]

RARE_WORD_THRESHOLD = 0
# Delete titles which have more than RARE_WORD_THRESHOLD rare words
# encoded_titles = [encoded_title
#                   for encoded_title in encoded_titles
#                   if encoded_title.count(w2i[rare_word_tk]) <= RARE_WORD_THRESHOLD]

encoded_titles[:], news_scores[:], news_ncomments[:] = \
                    zip(*[(encoded_title, score, ncomment)
                        for (encoded_title, score, ncomment) in zip(encoded_titles, news_scores, news_ncomments)
                        if encoded_title.count(w2i[rare_word_tk]) <= RARE_WORD_THRESHOLD])


decoded_titles = [[i2w[i] for i in encoded_title]
                  for encoded_title in encoded_titles]

processed_data_filename = 'processed_'+raw_data_filename

with open(processed_data_filename, mode='w', encoding='utf-8') as f:
    json.dump({'i2w': i2w,
               'data': encoded_titles,
               'score': news_scores,
               'ncomments': news_ncomments}, f)

pp(vocab_size+1)
pp(len(encoded_titles))
pp(w2i['that'])
pp(encoded_titles[10])
pp(decoded_titles[10])
pp(news_scores[10])
pp(news_ncomments[10])