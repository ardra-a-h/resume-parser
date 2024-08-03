import streamlit as st
from pypdf import PdfReader  
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords as stp

import re
from sklearn import model_selection

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from mlxtend.classifier import StackingClassifier
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.title("Resume Parser and Ranker")
#extract names
jd=st.text_input("Enter your Job Description")
def extract_names(t):
    t=nltk.word_tokenize(t)
    tag=nltk.pos_tag(t)
    tree=nltk.chunk.ne_chunk(tag)
    for i in tree:
        if hasattr(i,"label"):
            pername=""
            #for j in range(len(i)):
            for j in range(1):
                pername=pername+" "+(i[j][0])
            break
    return pername


#get skills



def extract_skills(input_text):
    file1=open('skills.txt','r')
    SKILLS_DB=file1.read().split(',\n')
    stop_words = set(nltk.corpus.stopwords.words('english'))
    word_tokens = nltk.tokenize.word_tokenize(input_text)

    # remove the stop words
    filtered_tokens = [w for w in word_tokens if w not in stop_words]

    # remove the punctuation
    filtered_tokens = [w for w in word_tokens if w.isalpha()]

    # generate bigrams and trigrams (such as artificial intelligence)
    bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))

    # we create a set to keep the results in.
    found_skills = set()

    # we search for each token in our skills database
    for token in filtered_tokens:
        if token.lower() in SKILLS_DB:
            found_skills.add(token)

    # we search for each bigram and trigram in our skills database
    for ngram in bigrams_trigrams:
        if ngram.lower() in SKILLS_DB:
            found_skills.add(ngram)

    return found_skills

def getskillset(text):
    skills = extract_skills(text)
    skillset=""
    for i in skills:
        skillset+=i+" "
    return skillset

uploaded_files = st.file_uploader("Upload File",accept_multiple_files=True)

def readfile(f):    
    if f is not None:
        reader = PdfReader(f)
        ft=""
        for page in reader.pages:
            ft=ft+" "+page.extract_text()
        return ft

try:
    if uploaded_files is not None:
        alldata=[]
        for uploaded_file in uploaded_files:
            filedata = readfile(uploaded_file)
            alldata.append(filedata)

except:
    pass


def classify(resumetext):
    #classifier
    resumeDataSet = pd.read_csv('UpdatedResumeDataSet.csv' ,encoding='utf-8')
    resumeDataSet['cleaned_resume'] = ''
    def cleanResume(resumeText):
        resumeText = re.sub(r'http\S+\s*', ' ', resumeText)  # remove URLs
        resumeText = re.sub(r'RT|cc', ' ', resumeText)  # remove RT and cc
        resumeText = re.sub(r'#\S+', '', resumeText)  # remove hashtags
        resumeText = re.sub(r'@\S+', '  ', resumeText)  # remove mentions
        resumeText = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
        resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
        resumeText = re.sub(r'\s+', ' ', resumeText)  # remove extra whitespace
        return resumeText
        
    resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x).lower())
    resumeDataSet['categorynames'] =resumeDataSet['Category']
    var_mod = ['Category']
    le = LabelEncoder()
    for i in var_mod:
        resumeDataSet[i] = le.fit_transform(resumeDataSet[i])

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(resumeDataSet['cleaned_resume'], resumeDataSet['Category'], test_size=0.33, random_state=125)
    resumeDataSet['vectorized'] = ''
    requiredText = resumeDataSet['cleaned_resume'].values
    requiredTarget = resumeDataSet['Category'].values

    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words='english',
        max_features=1500)
    word_vectorizer.fit(requiredText)
    WordFeatures = word_vectorizer.transform(requiredText)

    sampleresume_v=word_vectorizer.transform(resumetext)
    print ("Feature completed .....")

    X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=0, test_size=0.2)
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf3 = GaussianNB()
    lr = LogisticRegression()
    sclf = StackingClassifier(classifiers=[clf1, clf3], 
                            meta_classifier=lr)
    sclf.fit(X_train.toarray(),y_train)
    p=sclf.predict(sampleresume_v.toarray())
    for i in resumeDataSet['Category']:
        if i==p:
            predictedcategory=resumeDataSet['categorynames'][i]
            break
    return predictedcategory

def ranker(df,job_description):
    lemmatizer = WordNetLemmatizer()
    analyzer = CountVectorizer().build_analyzer()
    def get_wordnet_pos(word):    
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)
    def stemmed_words(doc):
        return (lemmatizer.lemmatize(w,get_wordnet_pos(w)) for w in analyzer(doc) if w not in set(stp.words('english')))        
            
    all_resume_text = []
    for i in df.iloc[:,1:2].values:
        s = ''
        for j in list(i):
            if len(s) == 0:
                s = str(j)
            else:
                s = s + ' , ' + str(j)
        all_resume_text.append(s)
    def get_tf_idf_cosine_similarity(job_desc,all_resumes):    
        tf_idf_vect = TfidfVectorizer(analyzer=stemmed_words)
        tf_idf_desc_vector = tf_idf_vect.fit_transform([job_desc])
        tf_idf_resume_vector = tf_idf_vect.transform(all_resumes)
        cosine_similarity_list = []
        for i in range(len(tf_idf_resume_vector.todense())):
            cosine_similarity_list.append(cosine_similarity(tf_idf_desc_vector,tf_idf_resume_vector[i])[0][0])
        return cosine_similarity_list    
    cos_sim_list = get_tf_idf_cosine_similarity(job_description,all_resume_text)
    zipped_resume_rating = zip(cos_sim_list,dfip.name,[x for x in range(len(df))])
    sorted_resume_rating_list = sorted(zipped_resume_rating, key = lambda x: x[0], reverse=True)
    resume_score = [(round(x*100,2))for x in cos_sim_list]
    rankdf=pd.concat([dfip.name,dfip.Job_Desc,pd.DataFrame(resume_score,columns=['resume_score(%)'])],axis=1).sort_values(by=['resume_score(%)'],ascending=False).head(10)
    
    return rankdf

    


if st.button("Show Ranking"):
    data=[]
    no_of_resumes=len(alldata)
    for i in range(no_of_resumes):
        cname=extract_names(alldata[i])
        cskills=getskillset(alldata[i])
        data.append([cname,cskills,"",alldata[i]])
        
    dfip = pd.DataFrame(data, columns=['name', 'Skills','Job_Desc','ResumeDetails'])
    for i in range(len(data)):
        rt=dfip['ResumeDetails'][i]
        dfip['Job_Desc'][i]=classify([rt])
    
    #jd=customer segmentation, aws, testng, git,java, c/c++, perl, python, machine learning
    nd=ranker(dfip,getskillset(jd))
    st.write(nd.head())
    #st.write((nd.iloc[:,0:2]).head())
    