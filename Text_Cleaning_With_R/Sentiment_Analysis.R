# unigram approach i.e. single words' sentiment score is calculated which is summed up to give sentence context.
# Does not consider negation which alters the context
# the sum of positive and negative sentiment words can lead to a 0 overall score.
# these lexicons may not work on old texts such as thou or thy 

library(tidytext)
install.packages("textdata")

sentiments
get_sentiments("afinn") # gets specific sentiment lexicons
get_sentiments("bing")
get_sentiments("nrc")


library(janeaustenr)  # dataset of janeausten's books
library(dplyr)
library(stringr)

?austen_books
tidy_books <- austen_books() %>%
  group_by(book) %>%
  mutate(linenumber = row_number(),
         chapter = cumsum(str_detect(text, regex("^chapter [\\divxlc]",
                                                 ignore_case = TRUE)))) %>%
  ungroup() %>%
  unnest_tokens(word, text)

# get all words in nrc lexicon associated with joy
nrcjoy <- get_sentiments("nrc") %>%
  filter(sentiment == "joy")

tidy_books %>%
  filter(book == "Emma") %>%
  inner_join(nrcjoy) %>%  # joins all words in "Emma" that are in nrcjoy
  count(word, sort = TRUE) 

library(tidyr)

# Dividing text sections into 80 lines (small sections do not convey much context)
# and large sections can wipe out the narrative of the story
janeaustensentiment <- tidy_books %>%
  inner_join(get_sentiments("bing")) %>% # joins tidy books and the positive negative sentiments in bing lexicon 
  count(book, index = linenumber %/% 80, sentiment) %>% # count of total positive and negative sentiment in a section of 80 lines 
  spread(sentiment, n, fill = 0) %>%  # spread is spreading the sentiment column into postive and negative  
  mutate(sentiment = positive - negative) # net sentiment of the section
janeaustensentiment

library(ggplot2)

ggplot(janeaustensentiment, aes(index, sentiment, fill = book)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~book, ncol = 2, scales = "free_x") # key_variable i.e. ~book is used to plot facet_wrap
# facet-wrap allows multiple ploting of graphs but you need data in key-value pair 

pride_prejudice <- tidy_books %>%
  filter(book == "Pride & Prejudice")
pride_prejudice

afinn <- pride_prejudice %>%
  inner_join(get_sentiments("afinn")) %>%
  group_by(index = linenumber %/% 80) %>%
  summarise(sentiment = sum(value)) %>%
  mutate(method = "AFINN")
afinn

bing_and_nrc <- bind_rows(  # bind two rows(data1,data2,id)
  pride_prejudice %>%
    inner_join(get_sentiments("bing")) %>%
    mutate(method = "Bing et al."),
  pride_prejudice %>%
    inner_join(get_sentiments("nrc") %>%
                 filter(sentiment %in% c("positive", # since nrc has other categories such as joy/sadness we are filtering for + and - sentiment 
                                         "negative"))) %>%
    mutate(method = "NRC")) %>%
count(method, index = linenumber %/% 80, sentiment) %>% # count number of positive and negative sentiments per 80 lines section based on the method i.e. nrc or bing 
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative)
bing_and_nrc

bind_rows(afinn,
          bing_and_nrc) %>%
  ggplot(aes(index, sentiment, fill = method)) + # fill is providing the color here i.e. 3
  geom_col(show.legend = FALSE) +
  facet_wrap(~method, ncol = 1, scales = "free_y")

# why is nrc more biased to positive sentiment than bing or afinn

get_sentiments("nrc") %>%
  filter(sentiment %in% c("positive",
                          "negative")) %>%
  count(sentiment)

get_sentiments("bing") %>%
  count(sentiment)
# the ratio of positive to negative words in bing is higher than in nrc lexicon

# counting which words contribute more to the final sentiment

bing_word_counts <- tidy_books %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>% # word and sentiment came from bing lexicon
  ungroup()
bing_word_counts

bing_word_counts %>%
  group_by(sentiment) %>%
  top_n(10) %>%
  ungroup() %>%
  mutate(word = reorder(word, n)) %>% # reorder word column based on n(ocuurence of each word)
  ggplot(aes(word, n, fill = sentiment)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~sentiment, scales = "free_y") +
  labs(y = "Contribution to sentiment",
       x = NULL) +
  coord_flip()

# words such as miss are being classified as negative although it's used for title
# these words can be added in stop words

custom_stop_words <- bind_rows(data_frame(word = c("miss"),
                                          lexicon = c("custom")),
                               stop_words)
custom_stop_words

install.packages("wordcloud")
library(wordcloud)

tidy_books %>%
  anti_join(stop_words) %>% # returns all the rows in tidy_books where stop_words do not match
  count(word) %>%
  with(wordcloud(word, n, max.words = 100)) # top 100 words

install.packages("reshape2")
library(reshape2)

# acast converts data frame into a matrix
# formulas are defines as dependent variable ~ independent variable
# value.var is name of column which stores values
#?acast

tidy_books %>%
  inner_join(get_sentiments("bing")) %>%
  count(word, sentiment, sort = TRUE) %>%
  acast(word ~ sentiment, value.var = "n", fill = 0) %>% 
  comparison.cloud(colors = c("gray20", "gray80"), # compares the frequencies of words across docs 
                   max.words = 100)

# Looking at the text with sentence length

# to convert text into another encoding than UTF-8 
#iconv(text, to= "latin1") do it before unnesting

PandP_sentences <- data_frame(text = prideprejudice) %>%
  unnest_tokens(sentence, text, token = "sentences")

PandP_sentences$sentence[2]

austen_chapters <- austen_books() %>%
  group_by(book) %>%
  unnest_tokens(chapter, text, token = "regex",
                pattern = "Chapter|CHAPTER [\\dIVXLC]") %>%
  ungroup()

# gives all the chapters
austen_chapters %>%
  group_by(book) %>%
  summarise(chapters = n())

# get negative sentiments from bing lexicon
bingnegative <- get_sentiments("bing") %>%
  filter(sentiment == "negative")

wordcounts <- tidy_books %>%
  group_by(book, chapter) %>%
  summarize(words = n())

#? summarise

tidy_books %>%
  semi_join(bingnegative) %>% # returns all rows in tidy_books with a match in bingnegative
  group_by(book, chapter) %>% # group_by is followed by smmarize to produce output
  summarize(negativewords = n()) %>% # sums up all the negative words in each book's each chapter
  left_join(wordcounts, by = c("book", "chapter")) %>% # joins all rows from x i.e. summarize output
  mutate(ratio = negativewords/words) %>%
  filter(chapter != 0) %>%
  top_n(1) %>%
  ungroup()

# these are the chapters with most sad words

