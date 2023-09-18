# Tokenization
# creating some random text
text <- c("Because I could not stop for Death -","He kindly stopped for me -","The Carriage held but just Ourselves -","and Immortality")
text

# install and load package "dplyr"
library(dplyr)

# Create a dataframe from the text
text_df <- data_frame(line = 1:4, text = text)
text_df

# install and load tidytext
install.packages("tidytext")
library(tidytext)

# pass test_df as input to unnest_tokens with words as the output column and text as input from text_df
text_df %>%
  unnest_tokens(word, text)

# Now we will use texts from Jane Austen's books

library(janeaustenr) # download the dataset
?janeaustenr
library(stringr) # load stringr
?austen_books

original_books <- austen_books() %>% # pass dataframe of books into group by
  group_by(book) %>%   # grouped on the basis of book(title) of each novel
mutate(linenumber = row_number(), # add column linenumber 
       chapter = cumsum(str_detect(text, regex("^chapter [\\divxlc]",  # add chapters column  
                                               ignore_case = TRUE)))) %>%
  ungroup()
original_books

# tokenized books with tidy data format (each row has a single word) (a word can be any text i.e. word, sentence, n-gram..)
# tidy data format makes working with tidytools easier 
tidy_books <- original_books %>%
  unnest_tokens(word, text)
tidy_books

# Remove stopwords
data(stop_words)
tidy_books <- tidy_books %>%
  anti_join(stop_words)

# Frequency of the words
tidy_books %>%
  count(word, sort = TRUE)

# Plot
library(ggplot2)

tidy_books %>%
  count(word, sort = TRUE) %>%
  filter(n > 600) %>% # words with count value > 600
  mutate(word = reorder(word, n)) %>% # reorder the word column in tidy-books based on their count
  ggplot(aes(word, n)) + # x,y data
  geom_col() + # bar chart
  xlab(NULL) +
  coord_flip() # flip x and y's labels' descrition 



