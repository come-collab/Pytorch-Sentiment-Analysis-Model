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


#Load all the docs in a directory : 
def process_docs(directory, vocab):
    #walk through all files in the folder: 
    for filename in listdir(directory):
        #Skip any reviews in test set
        if filename.startswith('cv9'):
            continue
        #create a full path of the file to open : 
        path = directory + '/' + filename
        #add doc to vocab 
        add_doc_to_vocab(path,vocab)

#define vocab    
vocab = Counter()

#add all docs to vocab
process_docs('txt_sentoken/pos',vocab)
process_docs('txt_sentoken/neg',vocab)
#print the size of the vocab
print(len(vocab))
#Print the top word of the vocab : 
print(vocab.most_common(50))
# save tokens to a vocabulary file
# save_list(tokens,'vocab.txt')