#This is the code for the most basic NLP model the bag of word but done from scratch
#First prepare the data 
#Performing a Split of the data into a training and a Test set
#Then we will load and clean the reviews

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
import string
from collections import Counter
from os import listdir

#load doc into memory :

def load_doc(filename):
    file = open(filename,"r")
    text = file.read()
    file.close()

    return text


# save list to file
def save_list(lines, filename):
 # convert lines to a single blob of text
 data = '\n'.join(lines)
 # open file
 file = open(filename, 'w')
 # write text
 file.write(data)
 # close file
 file.close()
 



def clean_doc (doc):
    tokens = doc.split()
    #Remove the punctuation from each token
    table = str.maketrans('','',string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    
    #Remove remaining token that are not alphabetic
    tokens= [word for word in tokens if word.isalpha()]

    #Filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]

    #filters out short tokens
    tokens = [word for word in tokens if len(word) > 1 ]

    return tokens

#Load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
    #load doc
    doc = load_doc(filename)
    #clean doc
    tokens = clean_doc(doc)

    #updates counts
    vocab.update(tokens)


# load all docs in a directory
def process_docs(directory, vocab):
    lines = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load and clean the doc
        line = doc_to_line(path, vocab)
        # add to list
        lines.append(line)
    return lines

def doc_to_line(filename,vocab):
    #load the doc : 
    doc = load_doc(filename)

    #clean the doc and convert to tokens :
    tokens = clean_doc(doc)

    #Fitler by vocab :
    tokens = [w for w in tokens if w in vocab]
    return ''.join(tokens)




#define vocab    
vocab = Counter()

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load all training reviews
positive_lines = process_docs('txt_sentoken/pos', vocab)
negative_lines = process_docs('txt_sentoken/neg', vocab)


print(len(vocab))
# summarize what we have
print(len(positive_lines), len(negative_lines))