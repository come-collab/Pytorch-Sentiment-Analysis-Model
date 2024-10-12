#First prepare the data 
#Performing a Split of the data into a training and a Test set
#Then we will load and clean the reviews

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
import string

#load doc into memory :

def load_doc(filename):
    file = open(filename,"r")
    text = file.read()
    file.close()

    return text


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


#Load the document : 
filename = ('txt_sentoken/pos/cv000_29590.txt')
text = load_doc(filename)
tokens = clean_doc(text)
print(tokens)