import warnings
import RAKE
import nltk
# import sys
import numpy as np
import pandas as pd
from scipy.cluster.vq import whiten
from numpy.linalg import inv

warnings.filterwarnings('ignore')


def query_symptoms(symptoms):

    Qsym = pd.DataFrame()
    sym = pd.read_csv('./data/sym_3.csv')
    for s in symptoms:
        Qsym = Qsym.append(sym[sym['symptom'] == s])
    return Qsym


def predict_disease(Qsym, Qsym2, US_df, dia, V, reductie=30):
    for xx in range(len(Qsym)):
        waa = Qsym.iloc[xx, :].name
        Qsy = pd.DataFrame((pd.DataFrame(US_df.loc[waa - 1, :reductie - 1]).T))
        Qsym2 = Qsym2.append(Qsy)
    del Qsym2['eye']

    Qtemp = Qsym2.sum() * 2

    dise = (dia[dia['_id'] == 0])
    similQd = np.dot(Qtemp, V[0:reductie, :]) / np.dot(np.abs(Qtemp), np.abs(V[0:reductie, :])) * 100
    for xyz in range(len(V)):
        if similQd[xyz] > 20:
            disname = dia.iloc[[xyz - 1]]
            disname.ix[:, 'index'] = similQd[xyz]
            dise = dise.append(disname)
    return Qsym2, dise


def predict_related_symptoms(Qsym2, dia, US, V, sym):
    Qtemp = Qsym2.sum() * 2
    syme = dia[dia['_id'] == 0]
    similQs = np.dot(Qtemp, US.T) / np.dot(np.abs(Qtemp), np.abs(US.T)) * 100
    for xyz in range(len(V)):
        if similQs[xyz] > 20:
            symname = sym.iloc[[xyz - 1]]
            symname.ix[:, 'index'] = similQs[xyz]
            syme = syme.append(symname)
    return syme


def parse(sentence):
    sw = ['summaryunsigneddisreport', 'yregistration', 'amed', 'date', 'patient', 'mm', 'st', 'amdischarge', 'doctor', 'hospital',
          'surgery', 'pain', 'problem', 'discharge', 'admission', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
          'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
          'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
          'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'be',
          'a', 'an', 'the', 'and', 'but', 'if', 'or', 'bad', 'because', 'as', 'of', 'at', 'by', 'for', 'about', 'against',
          'between', 'into', 'through', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
          'under', 'further', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'more',
          'most', 'other', 'some', 'such', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'don', "don't",
          'should', "should've", 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
          'ma', 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'won', "won't", 'wouldn', "wouldn't"
          'aboard', 'about', 'above', 'across', 'along', 'an', 'and', 'another', 'any', 'around', 'as', 'at', 'below', 'behind', 'below', 'beneath', 'beside', 'beyond', 'certain', 'down', 'during', 'each', 'following', 'for', 'from', 'inside', 'into', 'its',
          'like', 'minus', 'my', 'near', 'next', 'opposite', 'outside', 'out', 'over', 'plus', 'round', 'so', 'some', 'than', 'through',
          'toward', 'underneath', 'unlike', 'yet', 'under', 'unsigneded', 'coupled', 'also', 'seem']

    # English stopwords
    sw += nltk.corpus.stopwords.words('english')

    r = RAKE.Rake(sw)
    keywords = r.run(sentence)
    keywords = [x[0] for x in keywords]
    symptoms = []
    for i in range(len(keywords)):
        keywords[i] = keywords[i][0].upper() + keywords[i][1:]
    sym = pd.read_csv('./data/sym_3.csv')
    for k in keywords:
        if k in sym['symptom'].tolist():
            symptoms.append(k)
    return symptoms

# Medical NLP stopwords


def predict(symptoms):

    Qsym = query_symptoms(symptoms)

    syd = pd.read_csv('./data/sym_dis_matrix.csv')
    dia = pd.read_csv('./data/dia_3.csv')
    sym = pd.read_csv('./data/sym_3.csv')

    A = syd
    A = A.append(A)
    B = A.drop(['eye'], axis=1)
    C = B
    # You can merge the symptoms and the diagnosis
    # A=pd.merge(sym,syd,how='inner',right_on='eye',left_on='chief_complaint_id')
    # B=pd.merge(A,dia,how='inner',left_on='_id',right_on='id')
    C.index.name = 'eye'
    C_df = pd.DataFrame(C)
    dia.reset_index(level=0, inplace=True)
    #print("There are", len(dia)-1, 'diseases')
    #print("There are", len(sym)-1, 'symptoms')
    # this is the core function, SVD searching for singularity

    C = whiten(C_df)
    U, s, V = np.linalg.svd(C, full_matrices=False)
    S = np.diag(s)

    # reduce the columns and see the influence
    reductie = 30
    #print(s[:reductie].sum(), 'explained variability of total ',s.sum(),s[:reductie].sum()/s.sum()*100," % variation explained")
    # reduce the number of columns and see the influence on the ranking
    S = S[0:reductie, 0:reductie]
    iS = inv(S)
    US = np.dot(U[:, 0:reductie], iS)

    # A fill up with US matrix
    US_df = pd.DataFrame(data=US)

    Qsym2 = pd.DataFrame({'eye': []})

    Qsym2, dise = predict_disease(Qsym, Qsym2, US_df, dia, V)
    dise = dise.sort_values(('index'), ascending=0)
    # print(dise.sort_values(('index'),ascending=0))
    res = ""
    res += str(dise.iloc[1]['diagnose']) + ", "
    res += str(dise.iloc[2]['diagnose']) + ", "
    res += str(dise.iloc[3]['diagnose']) + ". "

    syme = predict_related_symptoms(Qsym2, dia, US, V, sym)
    syme = (syme.sort_values(('index'), ascending=0))
    symps = []
    # res += "\nThe symptoms you may have are :-\n"
    # res += str(syme.iloc[0]['symptom']) + " with a probability of " + str(np.round(syme.iloc[0]['index'], 2)) + "%\n"
    # res += str(syme.iloc[1]['symptom']) + " with a probability of " + str(np.round(syme.iloc[1]['index'], 2)) + "%\n"
    # res += str(syme.iloc[3]['symptom']) + " with a probability of " + str(np.round(syme.iloc[3]['index'], 2)) + "%\n"
    for i in [0, 1, 3]:
        symps.append(syme.iloc[i]['symptom'])
    return res, symps


# sentence = "I am down with a bad fever and headache. Like, also, I have dizziness"
# print(sentence, "\n")
# symptoms = parse(sentence)
# if symptoms == []:
#     print("You are healthy!")
#     sys.exit()
# print("Symptoms - ", symptoms, "\n")
# symps, res = predict(symptoms)
# print(res)
# print(symps)
