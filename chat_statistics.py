import datetime
import re
import emoji
import emojis
from collections import defaultdict
from wordcloud import WordCloud
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt    
import pandas as pd
from data_preprocessing import prepare_words_for_translate

date_patt = re.compile(r'(\d+\/\d+\/\d+)\,')
time_patt = re.compile(r'\s(\d+\:\d+\s)\-')
name_patt = re.compile(r'\-\s([a-zA-Z0-9]+\s?[a-zA-Z0-9]+\s?[a-zA-Z0-9]+\s?)\:\s')

def read_file(file):
    '''Reads Whatsapp text file into a list of strings'''
    x = open(file,'r', encoding = 'utf-8') #Opens the text file into variable x but the variable cannot be explored yet
    y = x.read() #By now it becomes a huge chunk of string that we need to separate line by line
    content = y.splitlines() #The splitline method converts the chunk of string into a list of strings
    return content

def split_line(line):
    try:
        date = date_patt.search(line).group(1)
        time = time_patt.search(line).group(1)
        name = name_patt.search(line).group(1)
        msg = line.split(":")[2].lstrip().rstrip()
    except Exception:
        msg_data = {}
        # some data is missing so... ignore that message
    else:
        format_ = "%m/%d/%y"
        msg_data = {
            "date":datetime.datetime.strptime(date, format_),
            "time":time,
            "name":name,
            "message":msg
        }
    return msg_data

def startsWithDateTime(s):
    pattern = '^([0-2][0-9]|(3)[0-1])(\/)(((0)[0-9])|((1)[0-2]))(\/)(\d{2}|\d{4}), ([0-9][0-9]):([0-9][0-9]) -'
    result = re.match(pattern, s)
    if result:
        return True
    return False

def startsWithAuthor(s):
    patterns = [
        '([\w]+):',                        # First Name
        '([\w]+[\s]+[\w]+):',              # First Name + Last Name
        '([\w]+[\s]+[\w]+[\s]+[\w]+):'     # First Name + Middle Name + Last Name
    ]
    pattern = '^' + '|'.join(patterns)
    result = re.match(pattern, s)
    if result:
        return True
    return False

def getDataPoint(line):
    # line = 18/06/17, 22:47 - Loki: Why do you have 2 numbers, Banner?
    
    splitLine = line.split(' - ') # splitLine = ['18/06/17, 22:47', 'Loki: Why do you have 2 numbers, Banner?']
    
    dateTime = splitLine[0] # dateTime = '18/06/17, 22:47'
    
    date, time = dateTime.split(', ') # date = '18/06/17'; time = '22:47'
    
    message = ' '.join(splitLine[1:]) # message = 'Loki: Why do you have 2 numbers, Banner?'
    
    if startsWithAuthor(message): # True
        splitMessage = message.split(': ') # splitMessage = ['Loki', 'Why do you have 2 numbers, Banner?']
        author = splitMessage[0] # author = 'Loki'
        message = ' '.join(splitMessage[1:]) # message = 'Why do you have 2 numbers, Banner?'
    else:
        author = None
    return date, time, author, message

def extract_emojis(columnname, my_df):
    # Credit 
    emojis=[]
    for string in my_df[columnname]:
        my_str = str(string)
        for each in my_str:
            if each in emoji.UNICODE_EMOJI:
                emojis.append(each)
    return emojis

def generate_wordcloud(list_, mask=False, mask_path=""):
    #convert list to string
    temp = list()
    for item in list_:
        if len(item) > 3:
            temp.append(item)
    
    unique_string=(" ").join(temp)
    if mask:
        mask = np.array(Image.open(mask_path))
        wordcloud = WordCloud(width = 2000,mask=mask, height = 300, background_color="white").generate(unique_string)
    else:
        wordcloud = WordCloud(width = 2000, height = 500, background_color="white").generate(unique_string)
    
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()    
    plt.close()
    plt.savefig("word_cloud.png")
    return

def word_freq(vocab):
    words = defaultdict(int)
    for word in vocab:
        words[word] += 1
    words = {k: v for k, v in sorted(words.items(), key=lambda item: item[1], reverse=True)}
    return words

def pretty_print(dict_, n):
    for key, value in dict_.items():
        if n>=0:
            print(key, "--->", value)
            n-=1
            
def messages_per_user(file):
    chat = read_file(file)
    data = list()

    # reading file
    for line in chat:
        data.append(split_line(line))
    data = [entry for entry in data if entry]
    
    # creating dataframe
    df = pd.DataFrame(data)
    df = df.sort_values(by="date")
    
    # sorting messages per unique sender
    senders = df.name.unique()
    
    # printing messages per sender
    df.groupby(by="name")["name"].count()
    
    return df.groupby(by="name")["name"].count()
    
    
def media_per_user(file):
    chat = read_file(file)
    data = list()

    # reading file
    for line in chat:
        data.append(split_line(line))
    data = [entry for entry in data if entry]
    
    # creating dataframe
    df = pd.DataFrame(data)
    df = df.sort_values(by="date")
    
    # sorting messages per unique sender
    senders = df.name.unique()
    
    # media per sender
    return df[df['message'].str.contains("<Media omitted>")].groupby(by="name")["name"].count()
    
# returns df
def create_df(read_chat):
    data = list()

    # reading file
    for line in read_chat:
        data.append(split_line(line))
    data = [entry for entry in data if entry]
    
    # creating dataframe
    df = pd.DataFrame(data)
    df = df.sort_values(by="date")
    return df
    
def draw_messages_over_time_plt(file):

    chat = read_file(file)
    
    data = list()

    # reading file
    for line in chat:
        data.append(split_line(line))
    data = [entry for entry in data if entry]
    
    # creating dataframe
    df = pd.DataFrame(data)
    df = df.sort_values(by="date")
    
    # sorting by unique name
    senders = df.name.unique()
    
    # grouping by name
    df.groupby(by="name")["name"]
    df[df['message'].str.contains("<Media omitted>")].groupby(by="name")["name"]
    
    # grouping by messages over time
    messages_count_over_time = df.groupby(by=["date", "name"]).name.agg("count").to_frame("count_").reset_index().sort_values(by="date")
    
    freq = 3
    fig, ax = plt.subplots(figsize=(30,10))
    for sender in senders:
        tmp_df = messages_count_over_time[messages_count_over_time['name'] == sender]
    
        # Spot max count_ date
        max_data = tmp_df.loc[tmp_df['count_'].idxmax()]
        print(max_data.date, max_data.count_)
        ax.annotate('Date: {}\nCount: {}'.format(max_data.date, max_data.count_), xy=(max_data.date, max_data.count_))

        # printing plot of messages over time
        ax.plot(tmp_df.date, tmp_df.count_, label = sender)
        plt.xticks(rotation=90)
        ax.legend()
        ax.grid(True)
        plt.title("Messages sent over time")
        plt.xticks(tmp_df.date[::freq])
        plt.savefig("messages_over_time.png")
    
# takes file, returns emoji report
def emojis_per_sender(file):
    chat = read_file(file)
    df = create_df(chat)
    
    # sorting by unique sender name
    senders = df.name.unique()
    
    # extracting unique emojis
    emojis_per_sender = {}
    for sender in senders:
        emojis_count = defaultdict(int)
        messages = df[df['name'] == sender][['name','message']]
        emojis = extract_emojis("message", messages)
    
        # counting emojis
        for e in emojis:
            emojis_count[e] += 1    

        emojis_per_sender[sender] = emojis_count
        
    num_of_top_emojis = 20
    
    # printing report
    for sender, emojis in emojis_per_sender.items():
        print("Top {} emojis sent by {}".format(num_of_top_emojis, sender))
        sorted_dict = {k: v for k, v in sorted(emojis.items(), key=lambda item: item[1], reverse=True)}
        pretty_print(sorted_dict,num_of_top_emojis)
    
# takes words list and number of top words to display, returns report
def most_used_words(file):
    prep_words = prepare_words_for_translate(file)
    words = list()
    for word in prep_words:
        if len(word) > 5:
            words.append(word)
    # top words written
    num_of_top_words = 20
    word_frequencies = word_freq(words)
    print("Top {} croatian words written".format(num_of_top_words))
    pretty_print(word_frequencies,num_of_top_words)

def most_used_words_wordcloud(file):
    prep_words = prepare_words_for_translate(file)
    words = list()
    for word in prep_words:
        if len(word) > 5:
            words.append(word)

    mask = 'WordCloud\word_cloud1.png'
    # displaying top words used
    print('Croatian WordCloud')
    generate_wordcloud(words, True, mask)
    
# takes file, returns report
def freq_of_messages_per_day(file):
    file = open(r'chat_with_matea.txt',mode='r',encoding="utf8")
    data = file.read()
    file.close()

    hour_pattern = '(\d+):\d+\s+-\s+\w+\s?\w+?\s?\w+\s?\w+:\s'
    hours = re.findall(hour_pattern,data)
    time = pd.DataFrame({'hours':hours})
    busy_hours = time['hours'].value_counts()
    busy_hours.sort_index(inplace=True)
    plt.axes([1,1,1,0.98])
    plt.grid(True)
    busy_hours.plot.bar()
    plt.title("Frequency of messages per day")
    plt.xlabel('Hour')
    plt.ylabel('No. of Messages')
    plt.xticks(rotation=0)
    plt.show()
    plt.savefig("freq_of_messages_per_day.png")

# takes file path, return df    
def prepare_df_for_datetime_analysis(file):
    parsedData = []
    with open(file, encoding="utf-8") as fp:
        fp.readline() 

        messageBuffer = [] # Buffer to capture intermediate output for multi-line messages
        date, time, author = None, None, None # Intermediate variables to keep track of the current message being processed

        while True:
            line = fp.readline() 
            if not line: # Stop reading further if end of file has been reached
                break
            line = line.strip() 
            if startsWithDateTime(line): # If a line starts with a Date Time pattern, then this indicates the beginning of a new message
                if len(messageBuffer) > 0: # Check if the message buffer contains characters from previous iterations
                    parsedData.append([date, time, author, ' '.join(messageBuffer)]) # Save the tokens from the previous message in parsedData
                messageBuffer.clear() # Clear the message buffer so that it can be used for the next message
                date, time, author, message = getDataPoint(line) # Identify and extract tokens from the line
                messageBuffer.append(message) # Append message to buffer
            else:
                messageBuffer.append(line) # If a line doesn't start with a Date Time pattern, then it is part of a multi-line message. So, just append to buffer

    df = pd.DataFrame(parsedData, columns=['Date', 'Time', 'Author', 'Message'])
    
    return df

# takes file, returns media per user plt
def draw_media_per_user_plt(file):
    # creates df
    df = prepare_df_for_datetime_analysis(file)
    
    # sotrs media files sent by unique user
    media_messages_df = df[df['Message'] == '<Media omitted>']
    
    # counts media files by user
    author_media_messages_value_counts = media_messages_df['Author'].value_counts()
    top_10_author_media_messages_value_counts = author_media_messages_value_counts.head(10)
    top_10_author_media_messages_value_counts.plot.barh()
    plt.title("Number of media messages sent by user")
    plt.xlabel('Number of media messages')
    plt.ylabel('Authors')
    plt.savefig("media_files_per_user.png")
    
# takes dataframe, prints letter and word count
def calculate_word_and_letter_count(file): 
    # creates df
    df = prepare_df_for_datetime_analysis(file)
    
    null_authors_df = df[df['Author'].isnull()]
    
    # Drops all rows of the data frame containing messages from null authors
    messages_df = df.drop(null_authors_df.index)
    
    # sotrs media files sent by unique user
    media_messages_df = df[df['Message'] == '<Media omitted>']
    # Drops all rows of the data frame containing media messages
    messages_df = messages_df.drop(media_messages_df.index)
    
    messages_df['Letter_Count'] = messages_df['Message'].apply(lambda s : len(s))
    messages_df['Word_Count'] = messages_df['Message'].apply(lambda s : len(s.split(' ')))
    
    # prints letter and word count
    print('Letter count: {}, Word count: {}'.format(messages_df['Letter_Count'].sum(), messages_df['Word_Count'].sum()))

# takes file, draws words per author plt
def draw_words_per_author_plt(file):
    # creates df
    df = prepare_df_for_datetime_analysis(file)
    
    null_authors_df = df[df['Author'].isnull()]
    # Drops all rows of the data frame containing messages from null authors
    messages_df = df.drop(null_authors_df.index)

    messages_df['Letter_Count'] = messages_df['Message'].apply(lambda s : len(s))
    messages_df['Word_Count'] = messages_df['Message'].apply(lambda s : len(s.split(' ')))
    
    total_word_count_grouped_by_author = messages_df[['Author', 'Word_Count']].groupby('Author').sum()
    sorted_total_word_count_grouped_by_author = total_word_count_grouped_by_author.sort_values('Word_Count', ascending=False)
    top_10_sorted_total_word_count_grouped_by_author = sorted_total_word_count_grouped_by_author.head(10)
    top_10_sorted_total_word_count_grouped_by_author.plot.barh()
    plt.title("Number of words per author")
    plt.xlabel('Number of Words')
    plt.ylabel('Authors')
    plt.savefig("words_per_author.png")

# draws average words per message plt
def draw_avg_words_per_message(file):
    # creates df
    df = prepare_df_for_datetime_analysis(file)
    
    null_authors_df = df[df['Author'].isnull()]
    # Drops all rows of the data frame containing messages from null authors
    messages_df = df.drop(null_authors_df.index)
    
    # sotrs media files sent by unique user
    media_messages_df = df[df['Message'] == '<Media omitted>']

    # Drops all rows of the data frame containing media messages
    messages_df = messages_df.drop(media_messages_df.index)
    messages_df['Letter_Count'] = messages_df['Message'].apply(lambda s : len(s))
    messages_df['Word_Count'] = messages_df['Message'].apply(lambda s : len(s.split(' ')))
    
    plt.figure(figsize=(15, 2)) # To ensure that the bar plot fits in the output cell of a Jupyter notebook
    word_count_value_counts = messages_df['Word_Count'].value_counts()
    top_40_word_count_value_counts = word_count_value_counts.head(40)
    top_40_word_count_value_counts.plot.bar()
    plt.title("Frequency of words per message")
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.savefig("freq_of_words_per_message.png")

# draws number of letters per author plt
def draw_num_of_letters_per_author(file):
    # creates df
    df = prepare_df_for_datetime_analysis(file)
    
    null_authors_df = df[df['Author'].isnull()]
    # Drops all rows of the data frame containing messages from null authors
    messages_df = df.drop(null_authors_df.index)
    
    # sotrs media files sent by unique user
    media_messages_df = df[df['Message'] == '<Media omitted>']

    # Drops all rows of the data frame containing media messages
    messages_df = messages_df.drop(media_messages_df.index)
    messages_df['Letter_Count'] = messages_df['Message'].apply(lambda s : len(s))
    messages_df['Word_Count'] = messages_df['Message'].apply(lambda s : len(s.split(' ')))
    
    total_letter_count_grouped_by_author = messages_df[['Author', 'Letter_Count']].groupby('Author').sum()
    sorted_total_letter_count_grouped_by_author = total_letter_count_grouped_by_author.sort_values('Letter_Count', ascending=False)
    top_10_sorted_total_letter_count_grouped_by_author = sorted_total_letter_count_grouped_by_author.head(10)
    top_10_sorted_total_letter_count_grouped_by_author.plot.barh()
    plt.title("Number of letters per author")
    plt.xlabel('Number of Letters')
    plt.ylabel('Authors')
    plt.savefig("number_of_letters_per_author.png")
        
# draws average number of letters per message plt
def draw_num_of_letters_per_message(file):
    # creates df
    df = prepare_df_for_datetime_analysis(file)
    
    null_authors_df = df[df['Author'].isnull()]
    # Drops all rows of the data frame containing messages from null authors
    messages_df = df.drop(null_authors_df.index)
    
    # sotrs media files sent by unique user
    media_messages_df = df[df['Message'] == '<Media omitted>']

    # Drops all rows of the data frame containing media messages
    messages_df = messages_df.drop(media_messages_df.index)
    messages_df['Letter_Count'] = messages_df['Message'].apply(lambda s : len(s))
    messages_df['Word_Count'] = messages_df['Message'].apply(lambda s : len(s.split(' ')))
    
    plt.figure(figsize=(15, 2))
    letter_count_value_counts = messages_df['Letter_Count'].value_counts()
    top_40_letter_count_value_counts = letter_count_value_counts.head(40)
    top_40_letter_count_value_counts.plot.bar()
    plt.title("Number of letters per message")
    plt.xlabel('Letter count')
    plt.ylabel('Frequency')
    plt.savefig("number_of_letters_per_message.png")
    
# draws top dates with most words per day plt
def top_chatting_dates(file):
    # creates df
    df = prepare_df_for_datetime_analysis(file)
    
    null_authors_df = df[df['Author'].isnull()]
    # Drops all rows of the data frame containing messages from null authors
    messages_df = df.drop(null_authors_df.index)
    
    # sotrs media files sent by unique user
    media_messages_df = df[df['Message'] == '<Media omitted>']

    # Drops all rows of the data frame containing media messages
    messages_df = messages_df.drop(media_messages_df.index)
    messages_df['Letter_Count'] = messages_df['Message'].apply(lambda s : len(s))
    messages_df['Word_Count'] = messages_df['Message'].apply(lambda s : len(s.split(' ')))
    messages_df['Date'].value_counts().head(10).plot.barh()
    plt.title("Top dates with most messages that day")
    plt.xlabel('Number of Messages')
    plt.ylabel('Date')
    plt.savefig("top_chatting_dates.png")