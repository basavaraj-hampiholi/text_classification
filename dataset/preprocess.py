from classes import *
import argparse
import pandas as pd
import string
from torchtext.vocab import GloVe, vocab
from torchtext.data.utils import get_tokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')


def remove_punctuation(text):
    PUNCT_TO_REMOVE = string.punctuation
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


def remove_stopwords(text):
	STOPWORDS = set(stopwords.words('english'))
	return " ".join([word for word in str(text).split() if word not in STOPWORDS])


def lemmatize_words(text):
	lemmatizer = WordNetLemmatizer()
	wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
	pos_tagged_text = nltk.pos_tag(text.split())
	return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])


def encode_label(cls_name):
    for name, lbl in class_dict.items():
        if name==cls_name:
            return lbl


def encode_label(cls_name, lables_dict):
    for name, lbl in lables_dict.items():
        if name==cls_name:
            return lbl

def get_labels(df_train, label_dict):
    for i,rows in df_train.iterrows():
        rows['labels'] = encode_label(rows['labels'], label_dict)
    return df_train

def encode_vectors(text):
    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer(text)
    return tokens

def tokenize(df):
    tokenized_texts = []
    idx = 2
    word2vec={}
    word2vec['<pad>'] = 0
    word2vec['<unk>'] = 1
    max_len=0

    for i,rows in df.iterrows():
        tokenized_sentence = encode_vectors(rows['text_lemmatized'])
        tokenized_texts.append(tokenized_sentence)
        for token in tokenized_sentence:
            if token not in word2vec:
                word2vec[token] = idx
                idx += 1
        max_len = max(max_len, len(tokenized_sentence))
    return tokenized_texts, word2vec, max_len

def encode_index(tokenized_texts, word2idx, max_len):

    input_ids = []
    for tokenized_sent in tokenized_texts:
        # Pad sentences to max_len
        tokenized_sent += ['<pad>'] * (max_len - len(tokenized_sent))

        # Encode tokens to input_ids
        input_id = [word2idx.get(token) for token in tokenized_sent]
        input_ids.append(input_id)
        
    return np.array(input_ids)

def text_preprocessing(dframe):
	dframe['text_brackets']=dframe['text'].str.replace(r"\([^()]*\)","")
	dframe['text_brackets']=dframe['text_brackets'].str.replace(r"\d{1}|\d{2}|\d{3}|\d{4}","")
	dframe["text_lower"] = dframe["text_brackets"].str.lower()
	dframe["text_wo_punct"] = dframe["text_lower"].apply(lambda text: remove_punctuation(text))
	dframe["text_wo_stop"] = dframe["text_wo_punct"].apply(lambda text: remove_stopwords(text))
	dframe["text_lemmatized"] = dframe["text_wo_stop"].apply(lambda text: lemmatize_words(text))

	return dframe

def prepare(train_df, valid_df, text_df, lbl_dict):
	df_train_lemm = text_preprocessing(train_df)
	df_train_lemm = get_labels(df_train_lemm, lbl_dict)

	df_valid_lemm = text_preprocessing(valid_df)
	df_valid_lemm = get_labels(df_valid_lemm, lbl_dict)

	df_test_lemm = text_preprocessing(text_df)
	df_test_lemm = get_labels(df_test_lemm, lbl_dict)

	df_both = df_train_lemm.append(df_valid_lemm)
	df_all = df_both.append(df_test_lemm)

	return df_all

def read_csvs(train_path, valid_path, text_path):
	trn_frame = pd.read_csv(train_path)
	val_frame = pd.read_csv(valid_path)
	tst_frame = pd.read_csv(text_path)

	trn_frame['labels'] = trn_frame['l1']+'_'+trn_frame['l2']+'_'+trn_frame['l3']
	val_frame['labels'] = val_frame['l1']+'_'+val_frame['l2']+'_'+val_frame['l3']
	tst_frame['labels'] = tst_frame['l1']+'_'+tst_frame['l2']+'_'+tst_frame['l3']

	return trn_frame,val_frame,tst_frame

def main():

	parser = argparse.ArgumentParser(description='DBPedia Sentence Classification: Data Preparation')
	parser.add_argument('--train_path', default='DBPEDIA_train.csv', help='read train csv file')
	parser.add_argument('--val_path', default='DBPEDIA_val.csv', help='read validation csv file')
	parser.add_argument('--test_path', default='DBPEDIA_test.csv', help='read test csv file')
	parser.add_argument('--out_path', default='/home/vision/BMW_task/hmtc/npy/', help='output save path')

	args = parser.parse_args()

	df_train, df_valid, df_test = read_csvs(args.train_path, args.val_path, args.test_path)
	cat_df = prepare(df_train, df_valid, df_test, class_dict)
	text_tokens, text_word2vec, max_len = tokenize(cat_df)
	input_ids = encode_index(text_tokens, text_word2vec, max_len)

	train_inputs = input_ids[:len(df_train)]
	valid_inputs = input_ids[len(df_train):len(df_train)+len(df_valid)]
	test_inputs = input_ids[len(df_train)+len(df_valid):]

	np.save(args.out_path+'train_prepared.npy', train_inputs, allow_pickle=True)
	np.save(args.out_path+'valid_prepared.npy', valid_inputs, allow_pickle=True)
	np.save(args.out_path+'test_prepared.npy', test_inputs, allow_pickle=True)

if __name__ == '__main__':
   main()
