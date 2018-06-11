# lm_morse_solver
A solution based on language model to decipher morse cyphertext without spaces , inspired by an online question of DeeCamp.



It is original to solve a question posted by [DeeCamp (a public welfare camp for students interested in AI)](https://challenger.ai/deecamp_2018_reg) in their online examination. But I think the solution given here is easy to be applied in other similar problems. I have not seen others used the neural language model to improve the search when deciphering morse without spaces. Here is the question:



>一个粗心的发报员在发送莫尔斯电码（Morse Code）的时候，忘记在发送字母和单词之间停顿，结果收报系统收到的是下面这样的一个没有分隔符的点（.）划（-）的序列（请忽略换行符）。
```
.-.-....-.-...--.-...-....--...-.-...-.--.------..-...-..-.-.---...-..-..---..-..
....--..-.--.-...-.--......-.........-..-.----.-.....-....--.-.-.--.-..---..-....
..-...-..-.--.-.----......-.--.-----..-------.-.-..---.-.-.--..-.-...............
--...--....--..-....-.-----.....-...-------.-......-.........-..-..--.-....-...--
....-.--.-.....--..-.....--..-.---.--...-.-.-..-.-.....---.-.-.-.----....-..-....
.--..----......-...-.--.-...--.....--.....-.......-....---..-..--...-------.--...
.---..---.....-.-.-....-.-...--..-....---..--.--...-.-.-..-.-.....---.-.-.-.----.
...-..-.....--..----.
```
>已知这份报文的原始内容是一部著名英文小说的片段，请问，这部小说的作者是：

>(A) H. G. Wells
>(B) J. K. Rowling
>(C) Isaac Asimov
>(D) Lewis Carroll
>(E) Jack London
>(F) Stephen King
>(G) J. R. R. Tolkien
>(H) Edgar Rice Burroughs



To solve the problem, I try using beam search with the help of neural language model.  Thanks to [char-rnn.pytorch](https://github.com/spro/char-rnn.pytorch) and [pytorch-language-model](https://github.com/deeplearningathome/pytorch-language-model.git) , I trained a char-level model and a word-level model , and it seemed the word-level performed better. 


##Using

Here are just the script to solve the problem, so the codes are not well organized and just for study.  You can run python q2_1 to q2_5. they are different solutions that I tried to solve the problem.

- q2_1 : using char-level model to solve the problem, but it failed.
- q2_2 : a interactive program to solve the problem by detecting words which are matched in the prefix of what is left in the ciphertext and displaying them to you so that you can choose the next one.
- q2_4 : a word-level model to solve the problem, it work well
- q2_5 : a interactive program to solve the problem based on q2_4, and you can input the number of the most possible sequence you think to influence the search program ,or just press enter to let the program act as q2_4 , or input 'x' to go back the recent intervention point.

The result of q2_4:

>Alice was beginning to get very tired of sitting by her sister on the bank and of having nothing to do once or twice she had peeped into the book her sister was reading but it had no pictures or conversations in it and what is the use of a book thought Alice without pictures or conversation 
>

##More Information

I write a note for this question on [Zhihu](https://zhuanlan.zhihu.com/p/37971592).
