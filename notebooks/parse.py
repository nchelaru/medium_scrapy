import pandas as pd
import collections
import nltk
import inflection as inf
import spacy
from spacy_langdetect import LanguageDetector

final_df = pd.read_csv("/Users/nancy/Documents/Github/medium_scrapy/processed/2019_processed_titles.csv")

nlp = spacy.load('en_core_web_sm')

nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)

counts = collections.Counter()

token_text = []

for sent in final_df["names"]:
    doc = nlp(sent)
    if doc._.language['language'] == 'en':
        word_list = []
        for token in doc:
            if token.is_alpha:
                word_list.append(inf.singularize(token.text.lower()))
        counts.update(nltk.bigrams(word_list))

## Get dataframe of bigram counts
bigram_counts = pd.DataFrame.from_dict(counts, orient='index').reset_index()

bigram_counts.columns = ["Bigrams", 'Count']

bigram_counts[['Term1', 'Term2']] = pd.DataFrame(bigram_counts['Bigrams'].tolist(), index=bigram_counts.index)

bigram_counts = bigram_counts[['Term1', 'Term2', 'Count']]

bigram_counts.columns = ['word1', 'word2', 'n']

bigram_counts.to_csv('/Users/nancy/Documents/Github/medium_scrapy/processed/2019_titles_bigrams_nov21.csv', index=None)

