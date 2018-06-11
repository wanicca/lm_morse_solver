
import torch
import torch.nn as nn
from torch.autograd import Variable
from helpers import *
from model import *

chars = ",.0123456789?abcdefghijklmnopqrstuvwxyz-"
codes = """--..-- .-.-.- ----- .---- ..--- ...-- ....- ..... -.... --... ---..
      ----. ..--.. .- -... -.-. -.. . ..-. --. .... .. .--- -.- .-.. --
      -. --- .--. --.- .-. ... - ..- ...- .-- -..- -.-- --.. -....-"""
keys = dict(zip(chars, codes.split()))

text = """.-.-....-.-...--.-...-....--...-.-...-.--.------..-...-..-.-.---...-..-..---..-......--..-.--.-...-.--......-.........-..-.----.-.....-....--.-.-.--.-..---..-......-...-..-.--.-.----......-.--.-----..-------.-.-..---.-.-.--..-.-...............--...--....--..-....-.-----.....-...-------.-......-.........-..-..--.-....-...--....-.--.-.....--..-.....--..-.---.--...-.-.-..-.-.....---.-.-.-.----....-..-.....--..----......-...-.--.-...--.....--.....-.......-....---..-..--...-------.--....---..---.....-.-.-....-.-...--..-....---..--.--...-.-.-..-.-.....---.-.-.-.----....-..-.....--..----."""
temptext = text
# translated = ""
# while(temptext != ""):
#     matched = []
#     for k,v in keys.items():
#         if temptext.startswith(v):
#             matched.append(k)
#     for i in range(len(matched)):
#         print(("{}){}".format(i,matched[i])))
#     choose = eval(input(translated+'>'))
#     temptext = temptext[len(keys[matched[choose]]):]
#     translated = translated + matched[choose]


# def search(text,head=""):
#     matched = []
#     for k,v in keys.items():
#         if text.startswith(v):
#             matched.extend(search(text[len(v):],head+k))
#     if(matched == []):
#         matched = [head]
#     return matched

#返回匹配结果和每一项匹配后剩余部分开头的位置
def match(text):
    matched = []
    for k,v in keys.items():
        if text.startswith(v):
            matched.append((k,len(v)))
    return matched

def search(decoder,text,beam=1):
    hidden = decoder.init_hidden(1)
    
    matched = match(text)

    #初始化beam_list,不考虑beam大小，而是把第一步所有可能的字符都考虑进来
    beam_list = []
    for pr,start in matched:
        beam_list.append((pr,char_tensor(pr).unsqueeze(0),start,hidden,0))
        #list中存放的元组为 预测出的文本，前一个字符的向量，剩下的文本在电码中的位置，隐藏层，总概率
    
    search_end = 0
    with torch.no_grad():
        while(search_end < beam):#搜索中止条件，有beam个的分支都找到了结尾
            search_end = 0
            temp_list = []
            for pretext,lastchar,start,hidden,score in beam_list:
                if start<len(text) :   
                    matched = match(text[start:])
                    if len(matched)==0: 
                        continue
                else :
                    temp_list.append((pretext,lastchar,start,hidden,score))
                    search_end += 1
                    continue
                output, hidden = decoder(lastchar,hidden)
                output_dist = output.data.view(-1).exp()
                output_dist = output_dist.div(output_dist.sum()) #算了一下softmax结果
                output_dist = torch.log(output_dist) #取对数，愚蠢的我一开始忘了这么做，不过加上效果还是很差
                for m_ch,m_len in matched:
                    m_ch_id = all_characters.index(m_ch)
                    #下面几句是想把小写和大写字母看做同一个，并取其中大的概率值计分，不过没什么卵用
                    if(m_ch.isalpha() and m_ch.islower() and output_dist[m_ch_id+26]>output_dist[m_ch_id]):
                        m_ch_id += 26
                        m_ch = m_ch.upper()
                    temp_list.append((pretext+m_ch,char_tensor(m_ch).unsqueeze(0),start+m_len,hidden,score+float(output_dist[m_ch_id])))
            #排序，注意要除一下匹配出的字符长度，算出平均到每个词上的概率
            temp_list.sort(key=lambda k:k[4]/(len(k[0])-1),reverse=True)
            if search_end == len(temp_list):#为了应付最终给出的结果数不足beam个的情况，直接跳出
                break
            beam_list = []
            for i in range(len(temp_list)):
                if(i>=beam): break
                beam_list.append(temp_list[i])
            
    return [(a,e/(len(a)-1)) for a,b,c,d,e in beam_list]

if __name__ == '__main__':
    # decoder = torch.load('model.pt')
    decoder = torch.load('model.pt')
    r = search(decoder,text,beam=10)
    print(r)