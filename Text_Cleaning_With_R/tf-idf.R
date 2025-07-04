library(dplyr)
library(janeaustenr)
library(tidytext)

book_words <- austen_books() %>%
  unnest_tokens(word, text) %>%
  count(book, word, sort = TRUE) %>%
  ungroup()
book_words # word count of the words in each novel  

total_words <- book_words %>%
  group_by(book) %>%
  summarize(total = sum(n)) # creates a new column with total 
total_words # total number of words in each novel
 
book_words <- left_join(book_words, total_words)
book_words

# words such as "the", "of" "to" do not contribute much information but are in high usage throughout all novels
# term frequency graph
library(ggplot2)
ggplot(book_words, aes(n/total, fill = book)) +
  geom_histogram(show.legend = FALSE) +
  xlim(NA, 0.0009) +
  facet_wrap(~book, ncol = 2, scales = "free_y")

# the tails indicate the most common used words such as "the" or "of" 

freq_by_rank <- book_words %>%
  group_by(book) %>%
  mutate(rank = row_number(), # rank is the row number which is based on the "word" count 
         `term frequency` = n/total)
freq_by_rank

freq_by_rank %>%
  ggplot(aes(rank, `term frequency`, color = book)) +
  geom_line(size = 1.1, alpha = 0.8, show.legend = FALSE) + # alpha describes the transparency of the line
  scale_x_log10() +
  scale_y_log10()
# higher frequency corresponds to lower rank

#?lm
# viewing the middle section i.e. from rank 10 to 500 of our rank-term frequency graph
rank_subset <- freq_by_rank %>%
  filter(rank < 500,
         rank > 10)
lm(log10(`term frequency`) ~ log10(rank), data = rank_subset) # fitting a linear model
# intercept of our model(middle ranks from 10 to 500) is -0.62 and gradient = -1.11

#?geom_abline
freq_by_rank %>%
  ggplot(aes(rank, `term frequency`, color = book)) +
  geom_abline(intercept = -0.62, slope = -1.1, color = "gray50", linetype = 2) + # adds reference line for annotation purpose
  geom_line(size = 1.1, alpha = 0.8, show.legend = FALSE) +
  scale_x_log10() +
  scale_y_log10()
# deviation at higher ranks is common but at lower end defines that the author has used most common words less frequently than in many collections of language.

book_words <- book_words %>%
  bind_tf_idf(word, book, n) # defines the tf_idf scores (term,doc,counts)
book_words

book_words %>%
  select(-total) %>% # without total column
  arrange(desc(tf_idf)) # arrange in descending order of tf_idf column
# most of the rare ocurring words are basically character names which are quite important 

#?factor
#?rev

# plot of top 15 most important words
book_words %>%
  arrange(desc(tf_idf)) %>%
  mutate(word = factor(word, levels = rev(unique(word)))) %>%
  group_by(book) %>%
  top_n(15) %>%
  ungroup %>%
  ggplot(aes(word, tf_idf, fill = book)) +
  geom_col(show.legend = FALSE) +
  labs(x = NULL, y = "tf-idf") +
  facet_wrap(~book, ncol = 2, scales = "free") +
  coord_flip()

# same analysis but with physics text
install.packages("gutenbergr")
library(gutenbergr)
physics <- gutenberg_download(c(37729, 14725, 13476, 5001),
                              meta_fields = "author")
physics_words <- physics %>%
  unnest_tokens(word, text) %>%
  count(author, word, sort = TRUE) %>%
  ungroup()

physics_words

plot_physics <- physics_words %>%
  bind_tf_idf(word, author, n) %>%
  arrange(desc(tf_idf)) %>%
  mutate(word = factor(word, levels = rev(unique(word)))) %>%
  mutate(author = factor(author, levels = c("Galilei, Galileo",
                                            "Huygens, Christiaan",
                                            "Tesla, Nikola",
                                            "Einstein, Albert")))
plot_physics %>%
  group_by(author) %>%
  top_n(15, tf_idf) %>%
  ungroup() %>%
  mutate(word = reorder(word, tf_idf)) %>%
  ggplot(aes(word, tf_idf, fill = author)) +
  geom_col(show.legend = FALSE) +
  labs(x = NULL, y = "tf-idf") +
  facet_wrap(~author, ncol = 2, scales = "free") +
  coord_flip()

library(stringr)
physics %>%
  filter(str_detect(text, "eq\\.")) %>%
  select(text)

physics %>%
  filter(str_detect(text, "K1")) %>%
  select(text)

filter(str_detect(text, "AK")) %>%
  select(text)

mystopwords <- data_frame(word = c("eq", "co", "rc", "ac", "ak", "bn",
                                   "fig", "file", "cg", "cb", "cm"))
physics_words <- anti_join(physics_words, mystopwords, by = "word")
plot_physics <- physics_words %>%
  bind_tf_idf(word, author, n) %>%
  arrange(desc(tf_idf)) %>%
  mutate(word = factor(word, levels = rev(unique(word)))) %>%
  group_by(author) %>%
  top_n(15, tf_idf) %>%
  ungroup %>%
  mutate(author = factor(author, levels = c("Galilei, Galileo",
                                            "Huygens, Christiaan",
                                            "Tesla, Nikola",
                                            "Einstein, Albert")))
ggplot(plot_physics, aes(word, tf_idf, fill = author)) +
  geom_col(show.legend = FALSE) +
  labs(x = NULL, y = "tf-idf") +
  facet_wrap(~author, ncol = 2, scales = "free") +
  coord_flip()

