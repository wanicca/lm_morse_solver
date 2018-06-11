
import torch
import torch.nn as nn
from torch.autograd import Variable
# from helpers import *
# from model import *

chars = ",.0123456789?abcdefghijklmnopqrstuvwxyz-"
codes = """--..-- .-.-.- ----- .---- ..--- ...-- ....- ..... -.... --... ---..
      ----. ..--.. .- -... -.-. -.. . ..-. --. .... .. .--- -.- .-.. --
      -. --- .--. --.- .-. ... - ..- ...- .-- -..- -.-- --.. -....-"""
keys = dict(zip(chars, codes.split()))

text = """.-.-....-.-...--.-...-....--...-.-...-.--.------..-...-..-.-.---...-..-..---..-......--..-.--.-...-.--......-.........-..-.----.-.....-....--.-.-.--.-..---..-......-...-..-.--.-.----......-.--.-----..-------.-.-..---.-.-.--..-.-...............--...--....--..-....-.-----.....-...-------.-......-.........-..-..--.-....-...--....-.--.-.....--..-.....--..-.---.--...-.-.-..-.-.....---.-.-.-.----....-..-.....--..----......-...-.--.-...--.....--.....-.......-....---..-..--...-------.--....---..---.....-.-.-....-.-...--..-....---..--.--...-.-.-..-.-.....---.-.-.-.----....-..-.....--..----."""
text2 = '...---.........--.-...-.-.----......-....-...-.--...-.......-.---.---.--..-.-...-....--..-...-...-...........-.-.---..-.-..-....-.---.-.-.........-----...--.----.--.........-...-....-.....-..-.-.--....-.....--.-.--.--.-.....-..-.--...-...--.....-......-.-......--..-....-.....-.-..---..-.--.--.-..-.--..--...-.....-.---....--.-......-..-..-----..-.-..-..-.....-----.-.-....-.....-.-.---..--....-...---..-.--..--..-.--...-.--..--.-...--...-.-.-.-..-.--.-.....-...-............--.....-......--..-...-..-..-.--.-.--......-..-..--...-.....-.--..-.....--...-.-.-.-.--.....-..--.-.-..-..---....-...-.--......-.'
text3 = '..-.-.-...-.......--.-..---------.-------..-----....-.-...-.'
temptext = text

def word_tensor(word):
    wordid = word_to_id[word]
    return torch.tensor([wordid])

#返回匹配结果，候选词
def match(text):
    matched = []
    for k,v in words_to_morse.items():
        if text.startswith(v):
            matched.append((k,len(v)))
    return matched

def search(decoder,text,beam=1):
    hidden = decoder.init_hidden()
    
    matched = match(text)

    #初始化beamlist,不考虑beam大小，而是把第一步所有可能的字符都考虑进来
    beamlist = []
    beamlist_cache = []
    for pr,start in matched:
        beamlist.append(((pr,),word_tensor(pr).unsqueeze(0),start,hidden,0))
        # print(word_tensor(pr))
        #list中存放的元组为 预测出的文本，前一个字符的向量，剩下的文本在电码中的位置，隐藏层，总概率
    
    search_end = 0
    with torch.no_grad():
        while(search_end < beam):
            search_end = 0
            while True:
                for i in range(len(beamlist)):
                    print(i,'>',end='')
                    print_result(beamlist[i][0],beamlist[i][4],maxlength = 10)
                inp = input('>')
                if inp == '':
                    break
                elif inp=='x':
                    if(len(beamlist_cache)==0):
                        print('No cache.')
                    else:
                        beamlist = beamlist_cache.pop()
                else:
                    if(len(beamlist_cache)>5):beamlist_cache.pop(0)
                    beamlist_cache.append(beamlist)
                    beamlist = [beamlist[int(inp)]]
                    break

            temp_list = []
            for pretext,lastword,start,hidden,score in beamlist:
                if start<len(text) :   
                    matched = match(text[start:])
                    if len(matched)==0: 
                        continue
                else :
                    temp_list.append((pretext,lastword,start,hidden,score))
                    search_end += 1
                    continue
                output, hidden = decoder(lastword,hidden)
                output_dist = output.data.view(-1).exp()
                output_dist = output_dist.div(output_dist.sum())
                output_dist = torch.log(output_dist)

                for m_word,m_len in matched:
                    # if(m_word=='sister'):
                    #     m_word = m_word
                    m_word_id = word_to_id[m_word]
                    temp_list.append(((*pretext,m_word),word_tensor(m_word).unsqueeze(0),start+m_len,hidden,score+float(output_dist[m_word_id])))
            # temp_list.sort(key=lambda k:k[4]/(len(k[0])-1),reverse=True)
            temp_list.sort(key=lambda k:k[4]/(len(k[0])-1)+k[2]/len(k[0]),reverse=True)
            # temp_list.sort(key=lambda k:k[4]*(k[2]/len(text)),reverse=True)
            # temp_list.sort(key=lambda k:k[4]*(1+0.5*k[2]/len(k[0])),reverse=True)
            # temp_list.sort(key=lambda k:k[4]*(1+k[2]/len(text))+k[2]/len(k[0]),reverse=True)
            if search_end == len(temp_list):
                break
            beamlist = []
            if len(temp_list)==0:
                print('Nothing matched! Please go back.')
            # maxscore = temp_list[0][4]
            for i in range(len(temp_list)):
                if(i>=beam): break
                # if(maxscore-temp_list[i][4]>0.2):break
                beamlist.append(temp_list[i])
            
    return [(a,e/(len(a)-1)) for a,b,c,d,e in beamlist]

def print_result(guess,score,maxlength = 0):
    print(score,':',end=' ')
    if maxlength > 0 and len(guess)>maxlength:
        guess = guess[-maxlength:]
    for w in guess:
        print(w,end=' ')
    print()


# decoder = torch.load('model.pt')
decoder = torch.load('lm_model.pt')
with open('word_to_id','r') as f:
    word_to_id = eval(f.read())
with open('id_to_word','r') as f:
    id_to_word = eval(f.read())


words_to_morse = {}
for word in word_to_id.keys():
    morse = ""
    # if not word.isalpha():
        # continue
    for c in word:
        morse = morse + keys[c.lower()]
    words_to_morse[word]=morse

# print(decoder.batch_size,' ',decoder.num_steps)

r = search(decoder,text,beam=30)
# print(r)
for s,sc in r:
    print_result(s,sc)