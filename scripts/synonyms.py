from nltk.corpus import wordnet
from nltk.corpus import stopwords
import re
import requests
class Syn():
    def __init__(self):
        #TODO
        self.stop_words_list = list(set(stopwords.words('english')))
        return
    
    def GetSyn(self,word,k):
        #TODO
        wordnet_list = []
        synonyms_list = []
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                wordnet_list.append(l.name())
        wordnet_list = list(set(wordnet_list))
        for w in wordnet_list:
            if (w not in self.stop_words_list and re.match('^[a-zA-Z]+$',w)) and w!=word :
                synonyms_list.append(w)
        return synonyms_list[:min(len(synonyms_list),k)]

    def ApiGetSyn(self,word,k):
        synonyms_list = []
        url = 'https://api.datamuse.com/words?ml='
        r = requests.get(url+word)
        syn = []
        for w in r.json():
            syn.append(w['word'])
        for w in syn:
            if (w not in self.stop_words_list and re.match('^[a-zA-Z]+$',w)) and w!=word :
                synonyms_list.append(w)
        return synonyms_list[:min(len(synonyms_list),k)]


if __name__ == "__main__":
    syn = Syn()
    print(syn.GetSyn('good',5))
    print(syn.ApiGetSyn('good',5))
