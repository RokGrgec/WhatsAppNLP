import pandas as pd
import re
import numpy as np
import time
from datetime import datetime as dt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import emoji
from collections import Counter
import string
import copy
import sys
from nltk.stem import WordNetLemmatizer
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import os
import pygsheets
from google.oauth2 import service_account 
import json
from data_preprocessing import *
from gensim.models import LdaModel
from sklearn.datasets import fetch_20newsgroups
from nltk.stem.porter import *
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')

croatian_regex_spliter = '(\d+\/\d+\/\d+)(,)(\s)(\d+:\d+)(\w+)(\s)(-)(\s\w+)*(:)'

punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~(...)+-^[0-9]'''

croatian_stopwords = ["bumo", "duz","hoce", "hocemo","hocete", "hoces", "hocu", "jos","kojima","koju","kroz","li","me","mene","meni",
 "mi","mimo","moj","moja","moje","mu","na","nad","nakon","nam","nama","nas","naš","naša", "nasa", "naše",
 "nase", "našeg", "naseg", "ne","nego","neka",
 "neki","nekog","neku","nema","netko","neće", "nece", "nećemo", "necemo", "nećete", "necete", "nećeš", "neces", "neću", "necu", 
 "nešto","ni","nije","nikoga","nikoje","nikoju",
 "nisam","nisi","nismo","niste","nisu","njega","njegov","njegova","njegovo","njemu","njezin","njezina","njezino","njih",
 "njihov","njihova","njihovo","njim","njima","njoj","nju","no","o","od","odmah","on","ona","oni","ono","ova","pa","pak","po",
 "pod","pored","prije","s","sa","sam","samo","se","sebe","sebi","si","smo","ste","su","sve","svi","svog","svoj","svoja","svoje",
 "svom","ta","tada","taj","tako","te","tebe","tebi","ti","to","toj","tome","tu","tvoj","tvoja","tvoje","u","uz","vam",
 "vama","vas","vaš", "vaša", "vasa", "vaše", "vase", "već", "vec", "vi","vrlo","za","zar","će" "ce", 
 "ćemo", "cemo", "ćete", "cete", "ćeš", "ces","ću","cu","što","sto", "kaj", "zas", "u", "ok", "kk", "k", "okay", "hehe", "haha", "hahaha",
                     "i", "pa", "te", "ni", "niti", "a", "ali", "nego", "vec", "no", "je", "da", "ne", "sam", "si", "smo",
                     "ste", "su", "u", "ja", "lol", "lmao", "wow", "srsly", "ih", "dns", "jbg", "vjv", "mby", "media", "omitted", "ugl", "okej", "okey"]

english_stopwords = ["a","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although",
"always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero","omg"]

emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\U00002500-\U00002BEF"  # chinese char
    u"\U00002702-\U000027B0"
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    u"\U0001f926-\U0001f937"
    u"\U00010000-\U0010ffff"
    u"\u2640-\u2642" 
    u"\u2600-\u2B55"
    u"\u200d"
    u"\u23cf"
    u"\u23e9"
    u"\u231a"
    u"\ufe0f"  # dingbats
    u"\u3030"
                  "]+", re.UNICODE)


# global var for length of list of words sent too google sheets to translate
num_of_sent_words = 0

# function for filtering more words with len(word) < 3
def less_than_3_len(arr):
    temp_li = list()
    for item in arr:
        if len(item) > 3:
            temp_li.append(item)
            
    return temp_li

# function for removing integers
def remove_numbers(arr):
    without_int = [word for word in arr if not re.search(r'\d',word)]
    
    return without_int

# removing laugh patterns ('%haha%')
def remove_laugh_words(arr):
    words_laugh_cleaned = list()
    for item in arr:
        result = re.sub("hah\w+", "", item)
        words_laugh_cleaned.append(result)
        
    return words_laugh_cleaned
        
# tokenizing words per line (spliting words from characters eg. good looking => 'good', ' ' , 'looking')
def tokenizing_list(arr):
    tokenized_array = list() 
    for item in arr:
        tokenized_array.append(word_tokenize(item))
    
    return tokenized_array
    

# removing all punctuaction chars from list
def remove_punctuaction(arr):
    words_punctuaction_cleaned = list()
    for item in arr:
        if item not in punctuations:
            words_punctuaction_cleaned.append(item)
            
    return words_punctuaction_cleaned
        
# removing croatian stopwords
def remove_cro_stopwords(arr):
    without_cro_stop_w = [word for word in arr if not word in croatian_stopwords]
    
    return without_cro_stop_w

#removing english stopwords
def remove_eng_stopwords(arr):
    without_eng_stop_w = [word for word in arr if not word in english_stopwords]
    
    return without_eng_stop_w

        
# modifying chat text data and stripping it of useless attributes for further NLP processing 
def prepare_words_for_translate(txt_file):
    temp = list()
    with open(txt_file, encoding='utf-8') as my_file:
        for line in my_file:
            # striping croatian message pattern
            datetime_strip = re.sub(croatian_regex_spliter, "", line)
            # removing emojis
            emoji_strip = emoji_pattern.sub(r'', datetime_strip)
            # splitting strings into single word string
            newline_strip = emoji_strip.strip()
            # lowercasing each string in array
            lower_case_word = newline_strip.lower()
            
            # adding cleaned words to array 
            temp.append(lower_case_word)
    
    # closign file
    my_file.close()
    # removing words with length less than 3 
    working_arr = less_than_3_len(temp)
    # removing numbers
    temp = remove_numbers(working_arr)
    
    # removing laugh patterns (%haha%)
    working_arr = remove_laugh_words(temp)
    
    # tokenizing words per line (spliting words from characters eg. good looking => 'good', ' ' , 'looking')
    tokenized_array = tokenizing_list(working_arr)
    
    # remmoving empty strings from list
    merged_list = [] 
    for item in tokenized_array:
        merged_list += item
    
    # removing punctuactions
    working_arr = remove_punctuaction(merged_list)
    
    # removing croatian stopwords
    temp = remove_cro_stopwords(working_arr)
    # removing english stopwords
    working_arr = remove_eng_stopwords(temp)
    
    # returning cleaned array
    return working_arr
    
    
def prepare_translated_words_for_model(translated_words_arr):
    translated_words_to_clean = translated_words_arr
    stripped_arr = []
    for item in translated_words_to_clean:
        new_line_strip = item.strip() # spliting multiple word strings into single strings
        lower_case_word = new_line_strip.lower() # lower-casing words
        stripped_arr.append(lower_case_word)

    # init word lemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    leminized_arr = []
    
    # lemnmatizing words (making => make, painted => paint, going => go)
    for word in stripped_arr:
        leminized_arr.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        
    # removing words with length less than 3 
    working_arr = less_than_3_len(leminized_arr)

    # tokenizing array
    tokenized_array = tokenizing_list(working_arr)

    # removing empty strings from list
    merged_list = []
    for item in tokenized_array:
        merged_list += item
     
    # removing punctuaction chars
    working_arr = remove_punctuaction(merged_list)

    # removing croatian stopwords
    temp = remove_cro_stopwords(working_arr)
    # removing english stopwords
    working_arr = remove_eng_stopwords(temp)
    
    return working_arr


# functions for preprocessing train model corpus
stemmer = SnowballStemmer("english")
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result

def translate_words_to_eng(words_to_translate):
    # service_account stores token uri required for uusing Python automation on google sheets 
    client = pygsheets.authorize(service_account_file='service_account.json')

    # url to sheet
    spreadsheet_url = "https://docs.google.com/spreadsheets/d/11SJ9bVleK4i8KezqnJhK7UvH0pIvb03CDpMmumbmgjs/edit#gid=468911975"
    test = spreadsheet_url.split('/d/')
    # parsing id required to open work sheet
    id_ = test[1:][0].split('/edit')[0]
    
    # reading words to file
    text_file = open("Output.txt", "w", encoding='utf-8')
    for item in words_to_translate:
        text_file.write(item)
        text_file.write('\n')
    text_file.close()

    # converting .txt to csv for google sheets
    os.rename(r"Output.txt", r"Words.csv")

    # opening google sheets
    sheet = client.open_by_key(id_)
    wks = sheet.worksheet_by_title('Output')

    # reading csv with words for translate
    df = pd.read_csv('Words.csv', encoding='UTF-8')
    os.remove('Words.csv')

    # sending words to google sheet for tranbslate
    wks.set_dataframe(df, start=(1,1))
    
    ############################
    #    Timer needs to be     #
    #        added here        #
    ############################
    # making a delay of 7 minutes for words to be translated
    for i in range(0,420):
        time.sleep(1)
    
    # retrieving translated words 
    values_list = wks.get_col(2)

    # list of words ready for model processing
    translated_cleaned_words = prepare_translated_words_for_model(values_list[:len(df)]) # len(df) is used because it marks word count sent 
                                                                         # too google sheets
    return translated_cleaned_words


############################
#            TO            #
#            DO            #
############################
def train_20newsgroup_lda_model(num_of_topics):
    # init of 20newsgroup corpus
    newsgroups_train = fetch_20newsgroups(subset='train', shuffle = True)
    
    processed_docs = []
    # preprocessing newsgroup data
    for doc in newsgroups_train.data:
        processed_docs.append(preprocess(doc))

    # Creating a dictionary from 'processed_docs' containing the number of times a word appears in the training set using gensim.corpora.Dictionary and call it 'dictionary'
    dictionary = gensim.corpora.Dictionary(processed_docs)
    
    # removing words appearing less than 15 times
    # removing words appearing in more than 10% of all documents
    # keeping first 100,000
    dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)
    
    # converting data to bag of words list
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    # training LDA model
    lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = num_of_topics, 
                                   id2word = dictionary,                                    
                                   passes = 10,
                                   workers = 2)
    # returnign trained model
    return lda_model


def analyize_topics(bag_of_words, lda_model):
     # init of 20newsgroup corpus
    newsgroups_train = fetch_20newsgroups(subset='train', shuffle = True)
    
    processed_docs = []
    # preprocessing newsgroup data
    for doc in newsgroups_train.data:
        processed_docs.append(preprocess(doc))

    # Creating a dictionary from 'processed_docs' containing the number of times a word appears in the training set using gensim.corpora.Dictionary and call it 'dictionary'
    dictionary = gensim.corpora.Dictionary(processed_docs)
    
    # creating bow vector of passed words to analyze
    bow_vector = dictionary.doc2bow(bag_of_words)
    
    # printing topic recognizer result
    for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
        print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
        
        
def analyize_chat_topics(file, num_of_topics):
    # preprocessing words from file
    cro_words_to_translate = prepare_words_for_translate(file)
    
    # translating words to english
    translated_words = translate_words_to_eng(cro_words_to_translate)
    
    # fetching newsgroup train data
    newsgroups_train = fetch_20newsgroups(subset='train', shuffle = True)
    
    # cleaning newsgroup train data
    processed_docs = []
    # preprocessing newsgroup data
    for doc in newsgroups_train.data:
        processed_docs.append(preprocess(doc))

    # Creating a dictionary from 'processed_docs' containing the number of times a word appears in the training set using gensim.corpora.Dictionary and call it 'dictionary'
    dictionary = gensim.corpora.Dictionary(processed_docs)
    
    # removing words appearing less than 15 times
    # removing words appearing in more than 10% of all documents
    # keeping first 100,000
    dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)
    
    # converting data to bag of words list
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    # training LDA model
    lda_model =  gensim.models.LdaMulticore(bow_corpus, num_topics = num_of_topics, id2word = dictionary, passes = 10, workers = 2)
        # creating bow vector of passed words to analyze
    bow_vector = dictionary.doc2bow(translated_words)
    
    # printing topic recognizer result
    for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
        print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
