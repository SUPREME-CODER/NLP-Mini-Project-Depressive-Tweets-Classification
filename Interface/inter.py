import streamlit as st
import pickle as pk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


st.sidebar.image('depression.jpg', width = 300)
st.sidebar.title('BAD')
st.sidebar.subheader('Battle Against Depression')
st.sidebar.subheader('Created by:-')
st.sidebar.markdown('Abdeali Arsiwala.')
st.sidebar.markdown('Deep Doshi.')
st.sidebar.markdown('Sanket Sahasrabudhe.')
st.sidebar.markdown('Sujoy Torvi.')

st.write('<br><br>', unsafe_allow_html = True)
st.title('Are you depressed?')

st.write('<br>', unsafe_allow_html = True)
user_input = st.text_input("Input your tweet", '')


def clean(message): #doc is a string of text
    message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    sw = stopwords.words('english')
    words = [word for word in words if word not in sw]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

nb = pk.load(open('models/naivebayes.sav', 'rb'))
dt = pk.load(open('models/dtree.sav', 'rb'))
svm = pk.load(open('models/svm.sav', 'rb'))
lg = pk.load(open('models/logreg.sav', 'rb'))
vect = pk.load(open('models/vectorizer.sav', 'rb'))

st.write('<br>', unsafe_allow_html = True)
option = st.selectbox('Select a Model', ('Naive Bayes', 'Decision Tree', 'Logistic Regression', 'SVM'))

def nbpredict(tweet):
	global nb
	df = pd.DataFrame()
	df['msg'] = [tweet]
	test = vect.transform(df['msg'])
	pred = nb.predict(test)

	if pred[0] == 0:
		return 'Positive'
	else:
		return 'Negative'
	return pred

def dtreepredict(tweet):
	global dt
	df = pd.DataFrame()
	df['msg'] = [tweet]
	test = vect.transform(df['msg'])
	pred = dt.predict(test)

	if pred[0] == 0:
		return 'Positive'
	else:
		return 'Negative'
	return pred

def svmpredict(tweet):
	global svm
	df = pd.DataFrame()
	df['msg'] = [tweet]
	test = vect.transform(df['msg'])
	pred = svm.predict(test)

	if pred[0] == 0:
		return 'Positive'
	else:
		return 'Negative'
	return pred

def lgpredict(tweet):
	global lg
	df = pd.DataFrame()
	df['msg'] = [tweet]
	test = vect.transform(df['msg'])
	pred = lg.predict(test)

	if pred[0] == 0:
		return 'Positive'
	else:
		return 'Negative'
	return pred

if option == 'Naive Bayes':
	result = nbpredict(user_input)
elif option == 'Decision Tree':
	result = dtreepredict(user_input)
elif option == 'Logistic Regression':
	result = lgpredict(user_input)
else:
	result = svmpredict(user_input)

st.write('<br>', unsafe_allow_html = True)
if user_input != '':
	if result == 'Positive':
		st.success('The tweet is ' + result)
	else:
		st.error('The tweet is ' + result)

# st.write(user_input)