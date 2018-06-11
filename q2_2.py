import q2_1
import nltk.corpus

words = nltk.corpus.words.words()

words_to_morse = {}
for word in words:
    morse = ""
    for c in word:
        morse = morse + q2_1.keys[c.lower()]
    words_to_morse[word]=morse

words_in = []
for word,morse in words_to_morse.items():
    if morse in q2_1.text:
        words_in.append(word)



#查看结尾候选
# words_end = []
# for word,morse in words_to_morse.items():
#     if q2_1.text[:-len(words_to_morse['toe'])].endswith(morse):
#         words_end.append(word)



#查看开头候选
# words_start = []
# for word,morse in words_to_morse.items():
#     if q2_1.text.startswith(morse):
#         words_start.append(word)

#
# translated = ""
# temptext = q2_1.text
# while(temptext != ""):
#     # matched = []
#     words_end = []
#     for word,morse in words_to_morse.items():
#         if temptext.endswith(morse):
#             words_end.append(word)
#     for i in range(len(words_end)):
#         print(("{}){}".format(i,words_end[i])))
#     choose = eval(input(translated+'<'))
#     temptext = temptext[:-len(words_to_morse[words_end[choose]])]
#     translated = words_end[choose] + ' ' + translated 


translated = ""
temptext = q2_1.text
while(temptext != ""):
    # matched = []
    words_start = []
    for word,morse in words_to_morse.items():
        if temptext.startswith(morse):
            words_start.append(word)
    for i in range(len(words_start)):
        print(("{}){}".format(i,words_start[i])))
    choose = eval(input(translated+'>'))
    temptext = temptext[len(words_to_morse[words_start[choose]]):]
    translated = translated +' '+ words_start[choose]
