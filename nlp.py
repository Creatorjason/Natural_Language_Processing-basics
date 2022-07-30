import nltk
import matplotlib
from nltk.corpus import stopwords, state_union
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.stem import PorterStemmer # stemming class


exam_words = "This is the beginnig of wisdom"

tok_word = word_tokenize(exam_words)
stop_words = set(stop_words.words("English"))# gets all stopwords in english language
filtered_list = []

for w in tok_word:
    if w not in stop_words:
     filtered_list.append(w)


print(filtered_list)



ps = PorterStemmer()
ps.stem(tok_word)

train_text = state_union.raw('HISWILL.txt')
sample_text = state_union.raw('Glory.txt')
# nltk recognises this text as a corpus
#pass the trained text into the punktsentencetokenizer class

custom_text_tokenizer = PunktSentenceTokenizer(train_text)# a custom sentence is trained with a sample text
tokenized = custom_text_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged)
            namedEnt.draw()
            print(tagged)
    except Exception as e:
        print(str(e))

process_content() 

#################################################################################


