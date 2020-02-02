# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:25:43 2019
@author: Taufik Sutanto
taufik@tau-data.id
https://tau-data.id

~~Perjanjian Penggunaan Materi & Codes (PPMC) - License:~~
* Modul Python dan gambar-gambar (images) yang digunakan adalah milik dari berbagai sumber sebagaimana yang telah dicantumkan dalam masing-masing license modul, caption atau watermark.
* Materi & Codes diluar point (1) (i.e. "taudata.py" ini & semua slide ".ipynb)) yang digunakan di pelatihan ini dapat digunakan untuk keperluan akademis dan kegiatan non-komersil lainnya.
* Untuk keperluan diluar point (2), maka dibutuhkan izin tertulis dari Taufik Edy Sutanto (selanjutnya disebut sebagai pengarang).
* Materi & Codes tidak boleh dipublikasikan tanpa izin dari pengarang.
* Materi & codes diberikan "as-is", tanpa warranty. Pengarang tidak bertanggung jawab atas penggunaannya diluar kegiatan resmi yang dilaksanakan pengarang.
* Dengan menggunakan materi dan codes ini berarti pengguna telah menyetujui PPMC ini.
"""

from nltk.tokenize import TweetTokenizer; Tokenizer = TweetTokenizer(reduce_len=True)
from tqdm import tqdm, trange
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from bs4 import BeautifulSoup as bs
from sklearn.decomposition import LatentDirichletAllocation as LDA
from bz2 import BZ2File as bz2
from scipy import special
from collections import Counter
import re, networkx as nx, matplotlib.pyplot as plt, operator, numpy as np,community, os, csv#, spacy
#from spacy.lang.id import Indonesian
from twython import Twython, TwythonRateLimitError
import json, pandas as pd, itertools, nltk, time
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer;ps = PorterStemmer()
from itertools import chain
from html import unescape
from nltk import sent_tokenize
from unidecode import unidecode
from bs4.element import Comment
import warnings; warnings.simplefilter('ignore')

def loadCorpus(file='', sep=':', dictionary = True):
    file = open(file, 'r', encoding="utf-8", errors='replace')
    F = file.readlines()
    file.close()
    if dictionary:
        fix = {}
        for f in F:
            k, v = f.split(sep)
            k, v = k.strip(), v.strip()
            fix[k] = v
    else:
        fix = set( (w.strip().replace('\ufeff','') for w in F) )
    return fix

def cleanCorpus(file=''):
    f = open(file, 'r', encoding="utf-8", errors='replace')
    F = f.readlines()
    f.close()
    fix = set( (w.strip() for w in F) )
    fix = list(fix)
    fix.sort()
    with open(file, 'w') as f:
        for kata in fix:
            try:
                f.write(kata+'\n')
            except:
                pass
    print(fix[:5], '\n', fix[-5:])
    return True

def LoadStopWords(lang='en'):
    L = lang.lower().strip()
    if L == 'en' or L == 'english' or L == 'inggris':
        from spacy.lang.en import English as lemmatizer
        #lemmatizer = spacy.lang.en.English
        lemmatizer = lemmatizer()
        #lemmatizer = spacy.load('en')
        stops =  set([t.strip() for t in LoadDocuments(file = 'data/stopwords_en.txt')[0]])
    elif L == 'id' or L == 'indonesia' or L=='indonesian':
        from spacy.lang.id import Indonesian
        #lemmatizer = spacy.lang.id.Indonesian
        lemmatizer = Indonesian()
        stops = set([t.strip() for t in LoadDocuments(file = 'data/stopwords_id.txt')[0]])
    else:
        print('Warning, language not recognized. Empty StopWords Given')
        stops = set(); lemmatizer = None
    return stops, lemmatizer    

def adaAngka(s):
    return any(i.isdigit() for i in s)

def fixTags(t):
    getHashtags = re.compile(r"#(\w+)")
    pisahtags = re.compile(r'[A-Z][^A-Z]*')
    tagS = re.findall(getHashtags, t)
    for tag in tagS:
        if len(tag)>0:
            tg = tag[0].upper()+tag[1:]
            proper_words = []
            if adaAngka(tg):
                tag2 = re.split('(\d+)',tg)
                tag2 = [w for w in tag2 if len(w)>0]
                for w in tag2:
                    try:
                        _ = int(w)
                        proper_words.append(w)
                    except:
                        w = w[0].upper()+w[1:]
                        proper_words = proper_words+re.findall(pisahtags, w)
            else:
                proper_words = re.findall(pisahtags, tg)
            proper_words = ' '.join(proper_words)
            t = t.replace('#'+tag, proper_words)
    return t

def cleanText(T, fix={}, lemma=None, stops = set(), symbols_remove = True, min_charLen = 2, max_charLen = 15, fixTag= False, fixMix=True):
    pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    t = re.sub(pattern,' ',T) #remove urls if any
    pattern = re.compile(r'ftp[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    t = re.sub(pattern,' ',t) #remove urls if any
    t = unescape(t) # html entities fix
    if fixTag:
        t = fixTags(t) # fix abcDef
    t = t.lower().strip() # lowercase
    t = unidecode(t)
    t = ''.join(''.join(s)[:2] for _, s in itertools.groupby(t)) # remove repetition
    t = t.replace('\n', ' ').replace('\r', ' ')
    t = sent_tokenize(t) # sentence segmentation. String to list
    for i, K in enumerate(t):
        if symbols_remove:
            K = re.sub(r'[^.,_a-zA-Z0-9 -\.]',' ',K)
        if lemma:
            listKata = lemma(K)
        else:
            listKata = TextBlob(K).words
        cleanList = []
        for token in listKata:
            if lemma:
                if str(token.text) in fix.keys():
                    token = fix[str(token.text)]
                try:
                    token = token.lemma_
                except:
                    token = lemma(token)[0].lemma_
            else:
                if str(token) in fix.keys():
                    token = fix[str(token)]
            if stops:
                if len(token)>=min_charLen and token not in stops:
                    if fixMix and str(token.replace(' ','').replace('-','a')).isalpha():
                        cleanList.append(token)
                    elif not fixMix:
                        cleanList.append(token)
            else:
                if len(token)>=min_charLen:
                    if fixMix and str(token.replace(' ','').replace('-','a')).isalpha():
                        cleanList.append(token)
                    elif not fixMix:
                        cleanList.append(token)
        t[i] = ' '.join(cleanList)
    return ' '.join(t) # Return kalimat lagi


def crawlFiles(dPath,types=None): # dPath ='C:/Temp/', types = 'pdf'
    if types:
        return [str(dPath+'/'+f).replace('\\','/') for f in os.listdir(dPath) if f.endswith('.'+types)]
    else:
        return [str(dPath+'/'+f).replace('\\','/') for f in os.listdir(dPath)]

def LoadDocuments(dPath=None,types=None, file = None): # types = ['pdf','doc','docx','txt','bz2']
    Files, Docs = [], []
    if types:
        for tipe in types:
            Files += crawlFiles(dPath,tipe)
    if file:
        Files = [file]
    if not types and not file: # get all files regardless of their extensions
        Files += crawlFiles(dPath)
    for f in Files:
        if f[-3:].lower() in ['txt', 'dic','py', 'ipynb']:
            try:
                df=open(f,"r",encoding="utf-8", errors='replace')
                Docs.append(df.readlines());df.close()
            except:
                print('error reading{0}'.format(f))
        elif f[-3:].lower()=='csv':
            Docs.append(pd.read_csv(f))
        else:
            print('Unsupported format {0}'.format(f))
    if file:
        Docs = Docs[0]
    return Docs, Files

def loadPos_id(file = 'data/kata_dasar.txt'):
    kata_pos = {}
    df=open(file,"r",encoding="utf-8", errors='replace')
    data=df.readlines();df.close()
    for line in data:
        d = line.split()
        kata = d[0].strip()
        pos = d[-1].strip().replace("(",'').replace(')','')
        kata_pos[kata] = pos
    return kata_pos

def simpleClean(T):
    pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    t = re.sub(pattern,' ',T) #remove urls if any
    pattern = re.compile(r'ftp[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    t = re.sub(pattern,' ',t) #remove urls if any
    t = unescape(t) # html entities fix
    t = t.lower().strip() # lowercase
    t = unidecode(t)
    t = ''.join(''.join(s)[:2] for _, s in itertools.groupby(t)) # remove repetition
    t = t.replace('\n', ' ').replace('\r', ' ')
    return re.sub(r'[^.,_a-zA-Z0-9 -\.]',' ',t)

def print_Topics(model, feature_names, Top_Topics, n_top_words):
    for topic_idx, topic in enumerate(model.components_[:Top_Topics]):
        print("Topic #%d:" %(topic_idx+1))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

"""
def loadCorpus(file='', sep=':', dictionary = True):
    file = open(file, 'r', encoding="utf-8", errors='replace')
    F = file.readlines()
    file.close()
    if dictionary:
        fix = {}
        for f in F:
            k, v = f.split(sep)
            k, v = k.strip(), v.strip()
            fix[k] = v
    else:
        fix = set( (w.strip().replace('\ufeff','') for w in F) )
    return fix
"""
def twitter_html2csv(fData, fHasil):
    urlPattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    print('Loading Data: ', flush = True)
    Tweets, Username, waktu, replies, retweets, likes, Language, urlStatus =  [], [], [], [], [], [], [], []
    soup = bs(open(fData,encoding='utf-8', errors = 'ignore', mode='r'),'html.parser')
    data = soup.find_all('li', class_= 'stream-item')
    for i,t in tqdm(enumerate(data)):
        T = t.find_all('p',class_='TweetTextSize')[0] # Loading tweet
        Tweets.append(bs(str(T),'html.parser').text)
        U = t.find_all('span',class_='username')
        Username.append(bs(str(U[0]),'html.parser').text)
        T = t.find_all('a',class_='tweet-timestamp')[0]# Loading Time
        waktu.append(bs(str(T),'html.parser').text)
        RP = t.find_all('span',class_='ProfileTweet-actionCountForAria')[0]# Loading reply, retweet & Likes
        replies.append(int((bs(str(RP), "lxml").text.split()[0]).replace('.','').replace(',','')))
        RT = t.find_all('span',class_='ProfileTweet-actionCountForAria')[1]
        RT = int((bs(str(RT), "lxml").text.split()[0]).replace('.','').replace(',',''))
        retweets.append(RT)
        L  = t.find_all('span',class_='ProfileTweet-actionCountForAria')[2]
        likes.append(int((bs(str(L), "lxml").text.split()[0]).replace('.','').replace(',','')))
        try:# Loading Bahasa
            L = t.find_all('span',class_='tweet-language')
            Language.append(bs(str(L[0]), "lxml").text)
        except:
            Language.append('')
        url = str(t.find_all('small',class_='time')[0])
        try:
            url = re.findall(urlPattern,url)[0]
        except:
            try:
                mulai, akhir = url.find('href="/')+len('href="/'), url.find('" title=')
                url = 'https://twitter.com/' + url[mulai:akhir]
            except:
                url = ''
        urlStatus.append(url)
    print('Saving Data to "%s" ' %fHasil, flush = True)
    dfile = open(fHasil, 'w', encoding='utf-8', newline='')
    dfile.write('Time, Username, Tweet, Replies, Retweets, Likes, Language, urlStatus\n')
    with dfile:
        writer = csv.writer(dfile)
        for i,t in enumerate(Tweets):
            writer.writerow([waktu[i],Username[i],t,replies[i],retweets[i],likes[i],Language[i],urlStatus[i]])
    dfile.close()
    print('All Finished', flush = True)

def BoxPlot(Data, Labels=None, dots = True, figName = 'boxPlot.png', Title = '',dpi=600):
    xLabel = 'Akurasi'; yLabel = 'Model'
    #Labels = list(dData.keys())
    #Data = [np.array(dData[d]['scores']) for d in dData.keys()]
    N = len(Data); pos = np.arange(N) + 1
    fig = plt.figure(1, figsize=(9, 6))
    ax1 = fig.add_subplot(111)
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.set_title(Title); ax1.set_xlabel(xLabel, size=32); ax1.set_ylabel(yLabel)
    BP = ax1.boxplot(Data,positions=pos,labels=Labels,notch=True, vert=False)
    for box in BP['boxes']:
        box.set( color='#7570b3', linewidth=2)
    if dots:
        for i,dt in zip(pos,Data):
            y = dt
            x = np.random.normal(i, 0.03, size=len(y))
            ax1.plot(y, x, 'b.', alpha=0.30, markersize = 5) # Flip this for vertical BP
    fig.savefig(figName, bbox_inches='tight', pad_inches = 0, dpi=dpi)
    plt.show()

def twitter_connect(Ck, Cs, At, As):
    try:
        twitter = Twython(Ck, Cs, At, As)
        user = twitter.verify_credentials()
        print('Welcome "%s" you are now connected to twitter server' %user['name'])
        return twitter
    except:
        print("Connection failed, please check your API keys or connection")

def getTweets(twitter, topic, N = 100, lan = None):
    Tweets, MAX_ATTEMPTS, count, dBreak, next_max_id = [], 3, 0, False, 0
    i = 0
    while True:
        if(count>=N or dBreak):
            print('\nFinished importing %.0f' %count);break
        if(i == 0):
            if lan:
                results=twitter.search(q=topic, lang=lan, count=100, tweet_mode = 'extended')
            else:
                results=twitter.search(q=topic, count=100, tweet_mode = 'extended')

            Tweets.extend(results['statuses'])
            count += len(results['statuses'])
            if count>N:
                print("\rNbr of Tweets captured: {}".format(N), end="")
                Tweets = Tweets[:N]
                dBreak = True; break
            else:
                print("\rNbr of Tweets captured: {}".format(count), end="")

        else:
            try:
                if lan:
                    results=twitter.search(q=topic,include_entities='true',max_id=next_max_id, lang=lan, count=100, tweet_mode = 'extended')
                else:
                    results=twitter.search(q=topic,include_entities='true',max_id=next_max_id, count=100, tweet_mode = 'extended')

                Tweets.extend(results['statuses'])
                count += len(results['statuses'])
                if count>N:
                    print("\rNbr of Tweets captured: {}".format(N), end="")
                    Tweets = Tweets[:N]
                    dBreak = True; break
                else:
                    print("\rNbr of Tweets captured: {}".format(count), end="")

                try:
                    next_results_url_params=results['search_metadata']['next_results']
                    next_max_id=next_results_url_params.split('max_id=')[1].split('&')[0]
                except:
                    print('\nFinished, no more tweets available for query "%s"' %str(topic), flush = True)
                    dBreak = True; break

            except TwythonRateLimitError:
                print('\nRate Limit reached ... sleeping for 15 Minutes', flush = True)
                for itr in trange(15*60):
                    time.sleep(1)
            except:
                print('\nSomething is not right, retrying ... (attempt = {}/{})'.format(i+1,MAX_ATTEMPTS), flush = True)
        i += 1
    return Tweets

def saveTweets(Tweets,file='Tweets.json', plain = False): #in Json Format
    with open(file, 'w') as f:
        for T in Tweets:
            if plain:
                f.write(T+'\n')
            else:
                try:
                    f.write(json.dumps(T)+'\n')
                except:
                    pass
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def text_from_html(body):
    soup = bs(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)

def loadTweets(file='Tweets.json'):
    f=open(file,encoding='utf-8', errors ='ignore', mode='r');T=f.readlines();f.close()
    for i,t in enumerate(T):
        T[i] = json.loads(t.strip())
    return T

def visualize_wordEmbedding(model):
    from sklearn.manifold import TSNE
    vocab = list(model.wv.vocab)
    X = model[vocab]
    print('projecting to lower dimension for visualization, \nplease wait (Might take awhile for large data) ...')
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
    print('Drawing ...')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df['x'], df['y'])
    for word, pos in df.iterrows():
        ax.annotate(word, pos)
    plt.show()
    print('Done ... ')
"""
def LoadStopWords(lang='en'):
    L = lang.lower().strip()
    if L == 'en' or L == 'english' or L == 'inggris':
        from spacy.lang.en import English as lemmatizer
        #lemmatizer = spacy.lang.en.English
        lemmatizer = lemmatizer()
        #lemmatizer = spacy.load('en')
        stops =  set([t.strip() for t in LoadDocuments(file = 'data/stopwords_en.txt')[0]])
    elif L == 'id' or L == 'indonesia' or L=='indonesian':
        from spacy.lang.id import Indonesian
        #lemmatizer = spacy.lang.id.Indonesian
        lemmatizer = Indonesian()
        stops = set([t.strip() for t in LoadDocuments(file = 'data/stopwords_id.txt')[0]])
    else:
        print('Warning, language not recognized. Empty StopWords Given')
        stops = set(); lemmatizer = None
    return stops, lemmatizer    

def cleanText(T, fix={}, lemma=None, stops = set(), symbols_remove = True, min_charLen = 2, max_charLen = 15, fixTag= False, fixMix=True):
    # lang & stopS only 2 options : 'en' atau 'id'
    # symbols ASCII atau alnum
    #T = 'kupu-kupu warna biru, abu-abu.'
    pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    t = re.sub(pattern,' ',T) #remove urls if any
    pattern = re.compile(r'ftp[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    t = re.sub(pattern,' ',t) #remove urls if any
    t = unescape(t) # html entities fix
    if fixTag:
        t = fixTags(t) # fix abcDef
    t = t.lower().strip() # lowercase
    t = unidecode(t)
    t = ''.join(''.join(s)[:2] for _, s in itertools.groupby(t)) # remove repetition
    t = t.replace('\n', ' ').replace('\r', ' ')
    t = sent_tokenize(t) # sentence segmentation. String to list
    for i, K in enumerate(t):
        if symbols_remove:
            K = re.sub(r'[^.,_a-zA-Z0-9 -\.]',' ',K)
        if lemma:
            listKata = lemma(K)
        else:
            listKata = TextBlob(K).words
        cleanList = []
        for token in listKata:
            if lemma:
                if str(token.text) in fix.keys():
                    token = fix[str(token.text)]
                try:
                    token = token.lemma_
                except:
                    token = lemma(token)[0].lemma_
            else:
                if str(token) in fix.keys():
                    token = fix[str(token)]
            if stops:
                if len(token)>=min_charLen and token not in stops:
                    if fixMix and str(token.replace(' ','').replace('-','a')).isalpha():
                        cleanList.append(token)
                    elif not fixMix:
                        cleanList.append(token)
            else:
                if len(token)>=min_charLen:
                    if fixMix and str(token.replace(' ','').replace('-','a')).isalpha():
                        cleanList.append(token)
                    elif not fixMix:
                        cleanList.append(token)
        t[i] = ' '.join(cleanList)
    return ' '.join(t) # Return kalimat lagi


def crawlFiles(dPath,types=None): # dPath ='C:/Temp/', types = 'pdf'
    if types:
        return [dPath+f for f in os.listdir(dPath) if f.endswith('.'+types)]
    else:
        return [dPath+f for f in os.listdir(dPath)]

def LoadDocuments(dPath=None,types=None, file = None): # types = ['pdf','doc','docx','txt','bz2']
    Files, Docs = [], []
    if types:
        for tipe in types:
            Files += crawlFiles(dPath,tipe)
    if file:
        Files = [file]
    if not types and not file: # get all files regardless of their extensions
        Files += crawlFiles(dPath)
    for f in Files:
        if f[-3:].lower() in ['txt', 'dic','py', 'ipynb']:
            try:
                df=open(f,"r",encoding="utf-8", errors='replace')
                Docs.append(df.readlines());df.close()
            except:
                print('error reading{0}'.format(f))
        elif f[-3:].lower()=='bz2':
            try:
                Docs.append(readBz2(f))
            except:
                print('error reading{0}'.format(f))
        elif f[-3:].lower()=='csv':
            Docs.append(pd.read_csv(f))
        else:
            print('Unsupported format {0}'.format(f))
    if file:
        Docs = Docs[0]
    return Docs, Files
"""
def readBz2(file):
    with bz2(file, "r") as bzData:
        txt = []
        for line in bzData:
            try:
                txt.append(line.strip().decode('utf-8','replace'))
            except:
                pass
    return ' '.join(txt)

def countWords(Doc):
    Full_Tokens  = []
    for line in Doc:
        Full_Tokens.append(TextBlob(line.lower()).words)
    Full_Tokens = [kata for line in Full_Tokens for kata in line] # Flatten List
    Words = Counter(Full_Tokens)
    Words = [(kata,freq) for kata,freq in Words.items()]
    Words.sort(key=lambda tup: tup[1])
    Words = Words[::-1] # Reverse = Descending
    frekuensi = [f[1] for f in Words]
    kata = [f[0] for f in Words]
    return kata, frekuensi

def barChart(kata,frekuensi,N=100):
    y_pos = np.arange(len(kata[:N]))
    plt.bar(y_pos, frekuensi[:N], align='center', alpha=0.5)
    plt.xticks(y_pos, kata)
    plt.ylabel('Frekuensi')
    plt.title('Token')
    plt.show()


def Zipf(frekuensi, par=2, nBins = 100, yScale=0.05):
    count, bins, ignored = plt.hist(np.array(frekuensi), nBins, normed=True)
    plt.title("Zipf plot")
    x = np.arange(1., nBins)
    plt.xlabel("Frequency Rank of Token")
    y = x**(-par) / special.zetac(par)
    plt.ylabel("Absolute Frequency of Token")
    plt.plot(x, y*yScale/max(y), linewidth=2, color='r')
    plt.show()

def WordNet_id(f1 = 'data/wn-ind-def.tab', f2 = 'data/wn-msa-all.tab'):
    w1, wn_id = {}, {}
    df=open(f1,"r",encoding="utf-8", errors='replace')
    d1=df.readlines();df.close()
    df=open(f2,"r",encoding="utf-8", errors='replace')
    d2=df.readlines();df.close(); del df
    for line in d1:
        data = line.split('\t')
        w1[data[0].strip()] = data[-1].strip()
    for line in d2:
        data = line.split('\t')
        kata = data[-1].strip()
        kode = data[0].strip()
        if data[1].strip()=="I":
            if kode in w1.keys():
                if kata in wn_id:
                    wn_id[kata]['def'].append(w1[kode])
                    wn_id[kata]['pos'].append(kode[-1])
                else:
                    wn_id[kata] = {}
                    wn_id[kata]['def'] = [w1[kode]]
                    wn_id[kata]['pos'] = [kode[-1]]
            #else:
            #    wn_id[kata] = {}
            #    wn_id[kata]['def'] = ['']
            #    wn_id[kata]['pos'] = [kode[-1]]
    return wn_id
"""
def loadPos_id(file = 'data/kata_dasar.txt'):
    kata_pos = {}
    df=open(file,"r",encoding="utf-8", errors='replace')
    data=df.readlines();df.close()
    for line in data:
        d = line.split()
        kata = d[0].strip()
        pos = d[-1].strip().replace("(",'').replace(')','')
        kata_pos[kata] = pos
    return kata_pos
"""
def lesk_wsd(sentence, ambiguous_word, pos=None, stem=True, hyperhypo=True):
    # https://en.wikipedia.org/wiki/Lesk_algorithm
    # https://stackoverflow.com/questions/20896278/word-sense-disambiguation-algorithm-in-python
    max_overlaps = 0; lesk_sense = None
    context_sentence = sentence.split()
    for ss in wn.synsets(ambiguous_word):
        #break
        if pos and ss.pos is not pos: # If POS is specified.
            continue
        lesk_dictionary = []
        lesk_dictionary+= ss.definition().replace('(','').replace(')','').split() # Includes definition.
        lesk_dictionary+= ss.lemma_names() # Includes lemma_names.
        # Optional: includes lemma_names of hypernyms and hyponyms.
        if hyperhypo == True:
            lesk_dictionary+= list(chain(*[i.lemma_names() for i in ss.hypernyms()+ss.hyponyms()]))

        if stem == True: # Matching exact words causes sparsity, so lets match stems.
            lesk_dictionary = [ps.stem(i) for i in lesk_dictionary]
            context_sentence = [ps.stem(i) for i in context_sentence]

        overlaps = set(lesk_dictionary).intersection(context_sentence)

        if len(overlaps) > max_overlaps:
            lesk_sense = ss
            max_overlaps = len(overlaps)
    return lesk_sense.name()

def words(text): return re.findall(r'\w+', text.lower())

corpus = 'data/kata_dasar.txt'
WORDS = Counter(words(open(corpus).read()))

def P(word):
    "Probability of `word`."
    N=sum(WORDS.values())
    return WORDS[word] / N

def correction(word):
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def saveTweets_old(Tweets,file='Tweets.json', plain = False): #in "Json" Format or "txt" in plain type
    with open(file, 'w') as f:
        for T in Tweets:
            if plain:
                try:
                    f.write(T['nlp']+'\n')
                except:
                    f.write(T['fullTxt']+'\n')
            else:
                try:
                    f.write(json.dumps(T)+'\n')
                except:
                    pass

def loadTweets_old(file):
    f=open(file,encoding='utf-8', errors ='ignore', mode='r');T=f.readlines();f.close()
    for i,t in enumerate(T):
        T[i] = json.loads(t.strip())
    return T

def strip_non_ascii(string,symbols):
    ''' Returns the string without non ASCII characters''' #isascii = lambda s: len(s) == len(s.encode())
    stripped = (c for c in string if 0 < ord(c) < 127 and c not in symbols)
    return ''.join(stripped)
"""
def adaAngka(s):
    return any(i.isdigit() for i in s)

def fixTags(t):
    getHashtags = re.compile(r"#(\w+)")
    pisahtags = re.compile(r'[A-Z][^A-Z]*')
    tagS = re.findall(getHashtags, t)
    for tag in tagS:
        if len(tag)>0:
            tg = tag[0].upper()+tag[1:]
            proper_words = []
            if adaAngka(tg):
                tag2 = re.split('(\d+)',tg)
                tag2 = [w for w in tag2 if len(w)>0]
                for w in tag2:
                    try:
                        _ = int(w) # error if w not a number
                        proper_words.append(w)
                    except:
                        w = w[0].upper()+w[1:]
                        proper_words = proper_words+re.findall(pisahtags, w)
            else:
                proper_words = re.findall(pisahtags, tg)
            proper_words = ' '.join(proper_words)
            t = t.replace('#'+tag, proper_words)
    return t
"""

def cleanTweets(Tweets):
    factory = StopWordRemoverFactory(); stopwords = set(factory.get_stop_words()+['twitter','rt','pic','com','yg','ga','https'])
    factory = StemmerFactory(); stemmer = factory.create_stemmer()
    for i,tweet in enumerate(tqdm(Tweets)):
        txt = tweet['fullTxt'] # if you want to ignore retweets  ==> if not re.match(r'^RT.*', txt):
        txt = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',' ',txt)# clean urls
        txt = txt.lower() # Lowercase
        txt = Tokenizer.tokenize(txt)
        symbols = set(['@']) # Add more if you want
        txt = [strip_non_ascii(t,symbols) for t in txt] #remove all non ASCII characters
        txt = ' '.join([t for t in txt if len(t)>1])
        Tweets[i]['cleanTxt'] = txt # this is not a good Python practice, only for learning.
        txt = stemmer.stem(txt).split()
        Tweets[i]['nlp'] = ' '.join([t for t in txt if t not in stopwords])
    return Tweets

def translate(txt,language='en'): # txt is a TextBlob object
    try:
        return txt.translate(to=language)
    except:
        return txt

def sentiment_old(Tweets): #need a clean tweets
    print("Calculating Sentiment and Subjectivity Score: ... ")
    T = [translate(TextBlob(tweet['cleanTxt'])) for tweet in tqdm(Tweets)]
    Sen = [tweet.sentiment.polarity for tweet in tqdm(T)]
    Sub = [float(tweet.sentiment.subjectivity) for tweet in tqdm(T)]
    Se, Su = [], []
    for score_se, score_su in zip(Sen,Sub):
        if score_se>0.1:
            Se.append('pos')
        elif score_se<-0.05: #I prefer this
            Se.append('neg')
        else:
            Se.append('net')
        if score_su>0.5:
            Su.append('Subjektif')
        else:
            Su.append('Objektif')
    label_se = ['Positif','Negatif', 'Netral']
    score_se = [len([True for t in Se if t=='pos']),len([True for t in Se if t=='neg']),len([True for t in Se if t=='net'])]
    label_su = ['Subjektif','Objektif']
    score_su = [len([True for t in Su if t=='Subjektif']),len([True for t in Su if t=='Objektif'])]
    PieChart(score_se,label_se); PieChart(score_su,label_su)
    Sen = [(s,t['fullTxt']) for s,t in zip(Sen,Tweets)]
    Sen.sort(key=lambda tup: tup[0])
    Sub = [(s,t['fullTxt']) for s,t in zip(Sub,Tweets)]
    Sub.sort(key=lambda tup: tup[0])
    return (Sen, Sub)

def sentiment(D): #need a clean tweets
    print("Calculating Sentiment and Subjectivity Score: ... ")
    T = [translate(TextBlob(t)) for t in tqdm(D)]
    Sen = [tweet.sentiment.polarity for tweet in tqdm(T)]
    Sub = [float(tweet.sentiment.subjectivity) for tweet in tqdm(T)]
    Se, Su = [], []
    for score_se, score_su in zip(Sen,Sub):
        if score_se>0.0:
            Se.append('pos')
        elif score_se<0.0: #I prefer this
            Se.append('neg')
        else:
            Se.append('net')
        if score_su>0.5:
            Su.append('Subjektif')
        else:
            Su.append('Objektif')
    label_se = ['Positif','Negatif', 'Netral']
    score_se = [len([True for t in Se if t=='pos']),len([True for t in Se if t=='neg']),len([True for t in Se if t=='net'])]
    label_su = ['Subjektif','Objektif']
    score_su = [len([True for t in Su if t=='Subjektif']),len([True for t in Su if t=='Objektif'])]
    PieChart(score_se,label_se); PieChart(score_su,label_su)
    Sen = [(s,t) for s,t in zip(Sen,D)]
    Sen.sort(key=lambda tup: tup[0])
    Sub = [(s,t) for s,t in zip(Sub,D)]
    Sub.sort(key=lambda tup: tup[0])
    return (Sen, Sub)

def printSA(SA, N = 2, emo = 'positif'):
    Sen, Sub = SA
    e = emo.lower().strip()
    if e=='positif' or e=='positive':
        tweets = Sen[-N:]
    elif e=='negatif' or e=='negative':
        tweets = Sen[:N]
    elif e=='netral' or e=='neutral':
        net = [(abs(score),t) for score,t in Sen if abs(score)<0.01]
        net.sort(key=lambda tup: tup[0])
        tweets = net[:N]
    elif e=='subjektif' or e=='subjective':
        tweets = Sub[-N:]
    elif e=='objektif' or e=='objective':
        tweets = Sub[:N]
    else:
        print('Wrong function input parameter = "{0}"'.format(emo)); tweets=[]
    print('"{0}" Tweets = '.format(emo))
    for t in tweets:
        print(t)

def wordClouds(Tweets, file = 'wordCloud.png', plain = False, stopwords=None):
    if plain: # ordinary (large) Text file - String
        txt = Tweets
    else:
        txt = [t['full_text'] for t in Tweets]; txt = ' '.join(txt)
    wc = WordCloud(background_color="white")#, max_font_size=40
    wordcloud = wc.generate(txt)
    plt.figure(num=1, facecolor='w', edgecolor='k') #figsize=(4, 3), dpi=600, #wc.to_file('wordCloud.png')
    plt.imshow(wordcloud, cmap=plt.cm.jet, interpolation='nearest', aspect='auto'); plt.xticks(()); plt.yticks(())
    #plt.savefig('wordCloud.png',bbox_inches='tight', pad_inches = 0.1, dpi=300)
    plt.show()

def PieChart(score,labels):
    fig1 = plt.figure(); fig1.add_subplot(111)
    plt.pie(score, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal');plt.show()
    return None

def drawGraph(G, Label, layOut='spring'):
    fig3 = plt.figure(); fig3.add_subplot(111)
    if layOut.lower()=='spring':
        pos = nx.spring_layout(G)
    elif layOut.lower()=='circular':
        pos=nx.circular_layout(G)
    elif layOut.lower()=='random':
        pos = nx.random_layout(G)
    elif layOut.lower()=='shells':
        shells = [G.core_nodes,sorted(G.major_building_routers, key=lambda n: nx.degree(G.topo, n)) + G.distribution_routers + G.server_nodes,G.hosts + G.minor_building_routers]
        pos = nx.shell_layout(G, shells)
    elif layOut.lower()=='spectral':
        pos=nx.spectral_layout(G)
    else:
        print('Graph Type is not available.')
        return
    nx.draw_networkx_nodes(G,pos, alpha=0.2,node_color='blue',node_size=600)
    if Label:
        nx.draw_networkx_labels(G,pos)
    nx.draw_networkx_edges(G,pos,width=4)
    plt.show()

def Graph(Tweets, Label = False, layOut='spring', plain = True): # Need the Tweets Before cleaning
    print("Please wait, building Graph .... ")
    G=nx.Graph()
    if plain:
        users = Tweets[0]
        tweet = Tweets[1]
        for u,t in zip(users, tweet):
            if u not in G.nodes():
                G.add_node(u)
            mentionS =  re.findall("@([a-zA-Z0-9]{1,15})", t)
            for mention in mentionS:
                if "." not in mention: #skipping emails
                    usr = mention.replace("@",'').strip()
                    if usr not in G.nodes():
                        G.add_node(usr)
                    G.add_edge(u,usr)
    else:
        for tweet in tqdm(Tweets):
            if tweet['user']['screen_name'] not in G.nodes():
                G.add_node(tweet['user']['screen_name'])
            mentionS =  re.findall("@([a-zA-Z0-9]{1,15})", tweet['full_text'])
            for mention in mentionS:
                if "." not in mention: #skipping emails
                    usr = mention.replace("@",'').strip()
                    if usr not in G.nodes():
                        G.add_node(usr)
                    G.add_edge(tweet['user']['screen_name'],usr)
    Nn, Ne = G.number_of_nodes(), G.number_of_edges()
    drawGraph(G, Label, layOut)
    print('Finished. There are %d nodes and %d edges in the Graph.' %(Nn,Ne))
    return G

def Centrality(G, N=10, method='katz', outliers=False, Label = True, layOut='shells'):

    if method.lower()=='katz':
        phi = 1.618033988749895 # largest eigenvalue of adj matrix
        ranking = nx.katz_centrality_numpy(G,1/phi)
    elif method.lower() == 'degree':
        ranking = nx.degree_centrality(G)
    elif method.lower() == 'eigen':
        ranking = nx.eigenvector_centrality_numpy(G)
    elif method.lower() =='closeness':
        ranking = nx.closeness_centrality(G)
    elif method.lower() =='betweeness':
        ranking = nx.betweenness_centrality(G)
    elif method.lower() =='harmonic':
        ranking = nx.harmonic_centrality(G)
    elif method.lower() =='percolation':
        ranking = nx.percolation_centrality(G)
    else:
        print('Error, Unsupported Method.'); return None

    important_nodes = sorted(ranking.items(), key=operator.itemgetter(1))[::-1]#[0:Nimportant]
    data = np.array([n[1] for n in important_nodes])
    dnodes = [n[0] for n in important_nodes][:N]
    if outliers:
        m = 1 # 1 standard Deviation CI
        data = data[:N]
        out = len(data[abs(data - np.mean(data)) > m * np.std(data)]) # outlier within m stDev interval
        if out<N:
            dnodes = [n for n in dnodes[:out]]

    print('Influencial Users: {0}'.format(str(dnodes)))
    print('Influencial Users Scores: {0}'.format(str(data[:len(dnodes)])))
    Gt = G.subgraph(dnodes)
    return Gt

def Community(G):
    part = community.best_partition(G)
    values = [part.get(node) for node in G.nodes()]
    mod, k = community.modularity(part,G), len(set(part.values()))
    print("Number of Communities = %d\nNetwork modularity = %.2f" %(k,mod)) # https://en.wikipedia.org/wiki/Modularity_%28networks%29
    fig2 = plt.figure(); fig2.add_subplot(111)
    nx.draw_shell(G, cmap = plt.get_cmap('gist_ncar'), node_color = values, node_size=30, with_labels=False)
    plt.show
    return values

"""
def print_Topics(model, feature_names, Top_Topics, n_top_words):
    for topic_idx, topic in enumerate(model.components_[:Top_Topics]):
        print("Topic #%d:" %(topic_idx+1))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
"""
def getTopics_old(Tweets,n_topics=5, Top_Words=7):
    Txt = [t['nlp'] for t in Tweets] # cleaned: stopwords, stemming
    tf_vectorizer = CountVectorizer(strip_accents = 'unicode', token_pattern = r'\b[a-zA-Z]{3,}\b', max_df = 0.95, min_df = 2)
    dtm_tf = tf_vectorizer.fit_transform(Txt)
    tf_terms = tf_vectorizer.get_feature_names()
    lda_tf = LDA(n_components=n_topics, learning_method='online', random_state=0).fit(dtm_tf)
    vsm_topics = lda_tf.transform(dtm_tf); doc_topic =  [a.argmax()+1 for a in tqdm(vsm_topics)] # topic of docs
    print('In total there are {0} major topics, distributed as follows'.format(len(set(doc_topic))))
    fig4 = plt.figure(); fig4.add_subplot(111)
    plt.hist(np.array(doc_topic), alpha=0.5); plt.show()
    print('Printing top {0} Topics, with top {1} Words:'.format(n_topics, Top_Words))
    print_Topics(lda_tf, tf_terms, n_topics, Top_Words)
    return lda_tf, dtm_tf, tf_vectorizer

def getTopics(Txt,n_topics=5, Top_Words=7):
    tf_vectorizer = CountVectorizer(strip_accents = 'unicode', token_pattern = r'\b[a-zA-Z]{3,}\b', max_df = 0.95, min_df = 2)
    dtm_tf = tf_vectorizer.fit_transform(Txt)
    tf_terms = tf_vectorizer.get_feature_names()
    lda_tf = LDA(n_components=n_topics, learning_method='online', random_state=0).fit(dtm_tf)
    vsm_topics = lda_tf.transform(dtm_tf); doc_topic =  [a.argmax()+1 for a in tqdm(vsm_topics)] # topic of docs
    print('In total there are {0} major topics, distributed as follows'.format(len(set(doc_topic))))
    fig4 = plt.figure(); fig4.add_subplot(111)
    plt.hist(np.array(doc_topic), alpha=0.5); plt.show()
    print('Printing top {0} Topics, with top {1} Words:'.format(n_topics, Top_Words))
    print_Topics(lda_tf, tf_terms, n_topics, Top_Words)
    return lda_tf, dtm_tf, tf_vectorizer

def get_nMax(arr, n):
    indices = arr.ravel().argsort()[-n:]
    indices = (np.unravel_index(i, arr.shape) for i in indices)
    return [(arr[i], i) for i in indices]

def filter_for_tags(tagged, tags=['NN', 'JJ', 'NNP']):
    return [item for item in tagged if item[1] in tags]

def normalize(tagged):
    return [(item[0].replace('.', ''), item[1]) for item in tagged]

def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in itertools.ifilterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element

def lDistance(firstString, secondString):
    "Function to find the Levenshtein distance between two words/sentences - gotten from http://rosettacode.org/wiki/Levenshtein_distance#Python"
    if len(firstString) > len(secondString):
        firstString, secondString = secondString, firstString
    distances = range(len(firstString) + 1)
    for index2, char2 in enumerate(secondString):
        newDistances = [index2 + 1]
        for index1, char1 in enumerate(firstString):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1])))
        distances = newDistances
    return distances[-1]

def buildGraph(nodes):
    "nodes - list of hashables that represents the nodes of the graph"
    gr = nx.Graph() #initialize an undirected graph
    gr.add_nodes_from(nodes)
    nodePairs = list(itertools.combinations(nodes, 2))

    #add edges to the graph (weighted by Levenshtein distance)
    for pair in nodePairs:
        firstString = pair[0]
        secondString = pair[1]
        levDistance = lDistance(firstString, secondString)
        gr.add_edge(firstString, secondString, weight=levDistance)

    return gr

def kataKunci(text):
    #tokenize the text using nltk
    wordTokens = nltk.word_tokenize(text)
    #assign POS tags to the words in the text
    tagged = nltk.pos_tag(wordTokens)
    textlist = [x[0] for x in tagged]

    tagged = filter_for_tags(tagged)
    tagged = normalize(tagged)

    unique_word_set = unique_everseen([x[0] for x in tagged])
    word_set_list = list(unique_word_set)

   #this will be used to determine adjacent words in order to construct keyphrases with two words

    graph = buildGraph(word_set_list)
    #pageRank - initial value of 1.0, error tolerance of 0,0001,
    calculated_page_rank = nx.pagerank(graph, weight='weight')
    #most important words in ascending order of importance
    keyphrases = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)
    #the number of keyphrases returned will be relative to the size of the text (a third of the number of vertices)
    aThird = len(word_set_list) / 3
    keyphrases = keyphrases[0:aThird+1]

    #take keyphrases with multiple words into consideration as done in the paper - if two words are adjacent in the text and are selected as keywords, join them
    #together
    modifiedKeyphrases = set([])
    dealtWith = set([]) #keeps track of individual keywords that have been joined to form a keyphrase
    i = 0
    j = 1
    while j < len(textlist):
        firstWord = textlist[i]
        secondWord = textlist[j]
        if firstWord in keyphrases and secondWord in keyphrases:
            keyphrase = firstWord + ' ' + secondWord
            modifiedKeyphrases.add(keyphrase)
            dealtWith.add(firstWord)
            dealtWith.add(secondWord)
        else:
            if firstWord in keyphrases and firstWord not in dealtWith:
                modifiedKeyphrases.add(firstWord)

            #if this is the last word in the text, and it is a keyword,
            #it definitely has no chance of being a keyphrase at this point
            if j == len(textlist)-1 and secondWord in keyphrases and secondWord not in dealtWith:
                modifiedKeyphrases.add(secondWord)

        i = i + 1
        j = j + 1

    return modifiedKeyphrases

def Rangkum(text,M):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentenceTokens = sent_detector.tokenize(text.strip())
    graph = buildGraph(sentenceTokens)
    calculated_page_rank = nx.pagerank(graph, weight='weight')
    #most important sentences in ascending order of importance
    sentences = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)
    #return a 100 word summary
    summary = ' '.join(sentences[:M])
    summaryWords = summary.split()
    summaryWords = summaryWords[0:101]
    summary = ' '.join(summaryWords)
    return summary