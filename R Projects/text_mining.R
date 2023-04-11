### Homework 6
### IS 571
### Brandon Kimball

library(tidyverse)
library(tidytext)
library(stringr)
library(wordcloud)
library(topicmodels)


# Loading the data
sotu <- read_csv("/Users/brandonk87/Downloads/R Class/Text Mining/state_of_the_union.csv")


##### Question 1 #####
# Selecting only George Washington speeches
sotu_gw <- sotu %>% filter(president == "George Washington")
# Tokenizing text
tidy_gw_sotu <- sotu_gw %>% unnest_tokens(word, text)
# Removing stop words
data("stop_words")
tidy_gw_sotu <- tidy_gw_sotu %>% anti_join(stop_words)

# Afinn sentiment
gw_afinn_sentiment <- tidy_gw_sotu %>% inner_join(get_sentiments("afinn")) %>% 
  group_by(speech_doc_id) %>% summarize(sentiment = mean(value)) %>%
  mutate(method = "AFINN")

# bing sentiment
gw_bing_sentiment <- tidy_gw_sotu %>% inner_join(get_sentiments("bing")) %>%
  count(speech_doc_id, sentiment) %>% pivot_wider(names_from = sentiment, values_from = n) %>%
  mutate(sentiment = positive - negative, method = 'bing')

# Combining Afinn and bing
bind_rows(gw_afinn_sentiment, gw_bing_sentiment) %>%
  ggplot(aes(speech_doc_id, sentiment, fill = method)) + geom_col(show.legend = FALSE) +
  facet_wrap(~method, scales = "free_y")

# The chart shows the sentiment of the 8 George Washington speeches based on the
# AFFIN and bing dictionaries. Based on the results, the most positive speech was 
# his 1st based on AFFIN and 8th based on bing. His 6th speech was the most negative
# based on both AFFIN and bing. 



##### Question 2 #####
# Tokenizing text
tidy_sotu <- sotu %>% unnest_tokens(word, text)
# Removing stop words
tidy_sotu <- tidy_sotu %>% anti_join(stop_words)

# ncr sentiment
nrc_sentiment <- tidy_sotu %>% inner_join(get_sentiments("nrc")) %>% 
  count(speech_doc_id, sentiment) %>%
  pivot_wider(names_from = sentiment, values_from = n) %>%
  mutate(sentiment = positive - negative) 

#Visualizing all the speeches and their sentiment
ggplot(nrc_sentiment, aes(speech_doc_id, sentiment)) + geom_point()

# Returning the most positive speech
most_positive <- nrc_sentiment$speech_doc_id[which.max(nrc_sentiment$sentiment)]
sotu %>% filter(speech_doc_id == most_positive)
# The most positive speech from all the presidents was by William McKinley on 12/3/1900 

# Returning the most negative speech
most_negative <- nrc_sentiment$speech_doc_id[which.min(nrc_sentiment$sentiment)]
sotu %>% filter(speech_doc_id == most_negative)
# The most negative speech from all the presidents was by James Madison 1814-09-20



##### Question 3 #####
# Creating a custom stop word list
custom_stop_words <- bind_rows(tibble(word = c('government', 'united', 'states',
                                               'congress', 'country', 'american',
                                               'citizens', 'public', 'people'),
                                      lexicon = c('custom')), stop_words)
# word count for each president's speech 
tidy_sotu_custom <- sotu %>% unnest_tokens(word, text)
tidy_sotu_custom <- tidy_sotu_custom %>% anti_join(custom_stop_words)
word_count <- tidy_sotu_custom %>% count(speech_doc_id, word)
# term document matrix
sotu_dtm <- word_count %>% cast_dtm(speech_doc_id, word, n)
#LDA
sotu_lda <- LDA(sotu_dtm, k = 5, control = list(seed = 1337))
sotu_topic <- tidy(sotu_lda, matrix = "beta")
# Visualizing the top 10 terms in each group
top_terms <- sotu_topic %>% group_by(topic) %>% 
  slice_max(beta, n = 10) %>% ungroup()
top_terms %>% mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) + geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") + scale_y_reordered()
# Based on the visualization, each group could be labeled as:
# 1- "Legality". This is because most words deal with law and jurisdiction
# 2- "Influence and Dominance". This is because most words deal with things such as war vs peace, force and power
# 3- "Mexican War". This is because most words deal with mexico, wartime and power
# 4- "General Bureaucracy". This is because most words deal with general government terms like report and department
# 5- "Finances". This is because most words deal with money, commodities, and resources



###### Question 4 #####
sotu_al <- sotu %>% filter(president == "Abraham Lincoln")
tidy_sotu_al <- sotu_al %>% unnest_tokens(word, text)
tidy_sotu_al <- tidy_sotu_al %>% anti_join(custom_stop_words)
# 500 word word cloud
tidy_sotu_al %>% count(word) %>% with(wordcloud(word, n, max.words = 500))
# 100 word word cloud
tidy_sotu_al %>% count(word) %>% with(wordcloud(word, n, max.words = 100))
# 10 word word cloud
tidy_sotu_al %>% count(word) %>% with(wordcloud(word, n, max.words = 10))

