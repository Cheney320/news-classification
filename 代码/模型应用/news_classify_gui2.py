from tkinter import *
import tkinter.scrolledtext as tkscroll
import tkinter.filedialog as tfl
from tkinter import ttk
import jieba
from jieba import analyse
import pickle
from lxml import etree
import random
from selenium import webdriver
import re
from Stacking import StackingModels
import warnings
warnings.filterwarnings('ignore')

dic = {'体育':0, '军事':1, '国际':2, '娱乐':3, '时尚':4, '汽车':5, '科技':6, '财经':7}
labels = {v:k for k,v in dic.items()}


# 爬取网易新闻数据
# dict = {'国际':'http://news.163.com/world/',
#         '军事':'http://war.163.com/',
#         '体育':'http://sports.163.com/',
#         '科技':'http://tech.163.com/',
#         '财经':'http://money.163.com/',
#         '时尚':'http://lady.163.com/',
#         '娱乐':'http://ent.163.com/',
#         '汽车':'https://auto.163.com/'}

# 爬取今日头条新闻
dict = {'国际':'https://www.toutiao.com/ch/news_world/',
        '军事':'https://www.toutiao.com/ch/news_military/',
        '体育':'https://www.toutiao.com/ch/news_sports/',
        '科技':'https://www.toutiao.com/ch/news_tech/',
        '财经':'https://www.toutiao.com/ch/news_finance/',
        '时尚':'https://www.toutiao.com/ch/news_fashion/',
        '娱乐':'https://www.toutiao.com/ch/news_entertainment/',
        '汽车':'https://www.toutiao.com/ch/news_car/'}

t = Tk()
t.title('新闻主题分类器')
t.geometry("800x600")

# 创建带滚动条的文本框，用于输入新闻文本
ts = tkscroll.ScrolledText(t, width=65, height=20)
# 创建文本框，用于展示提取的关键词
text1 = Text(t, width=40, height=5)
# 创建文本框，用于输出新闻预测结果
text2 = Text(t, width=40, height=5)
# 创建下拉框
cmb = ttk.Combobox(t)
cmb['value'] = ('国际','军事','体育','科技','财经','娱乐','时尚','汽车')
cmb.current(0)

label1 = Label(t, text='新闻文本：', font=15)
label2 = Label(t, text='关键词：', font=15)
label3 = Label(t, text='主题：', font=15)
label4 = Label(t, text='选择要爬取的新闻类别：', font=15)

# 控件布局
label1.place(x=10, y=10)
ts.place(x=100, y=10, width=600, height=200)
label2.place(x=10, y=240)
text1.place(x=100, y=240, width=600, height=30)
label3.place(x=10, y=300)
text2.place(x=100, y=300, width=600, height=150)
label4.place(x=100, y=480)
cmb.place(x=400, y=480)

# 打开本地文件插入到新闻文本框
def open_file(e):
    filename = tfl.askopenfilename()
    with open(filename, "r", encoding='utf-8') as f:
        string = f.read()
    ts.delete(1.0, END)  # 删除新闻文本框中内容，再插入新的内容
    ts.insert(0.0, string)

# 将新闻文本框中的内容写入到本地文件
def save_file(e):
    filename = tfl.asksaveasfilename()
    string = ts.get(1.0, END)
    with open(filename, "w", encoding='utf-8') as f:
        f.write(string)

# 分词
def split_word(content):
    segment = []
    try:
        segs = jieba.lcut(content)
        for seg in segs:
            if len(seg) > 1 and seg != '\r\n':
                segment.append(seg)
    except Exception as e:
        print(e)
    return segment

# 去停用词
def drop_stopwords(segment):
    with open('../../data/stopwords.txt', encoding='utf-8') as f:
        stopwords = []
        for line in f:
            stopwords.append(line.strip('\n'))
    clean_segment = [word for word in segment if word not in stopwords]
    return clean_segment

# 关键词抽取
def extract_keyword(e):
    text1.delete(1.0, END)  # 删除关键词文本框内的内容
    content = ts.get(1.0, END)  # 获取新闻文本框中的文本
    segment = split_word(content)
    clean_segment = drop_stopwords(segment)
    keywords = analyse.extract_tags(' '.join(clean_segment), topK=3, allowPOS=('ns','nr','nt','nz','vn'))  # 提取5个特征词
    # keywords = analyse.textrank(' '.join(clean_segment), topK=3, allowPOS=('ns','n'))
    text1.insert(0.0, ' '.join(keywords))

# 文本分类
def predict(e):
    text2.delete(1.0, END)  # 删除主题框中的内容
    content = ts.get(1.0, END)  # 获取新闻文本框中的文本
    segment = split_word(content)
    clean_segment = drop_stopwords(segment)
    with open('models/Stacking.pkl', 'rb') as f1, open('models/vect.pkl', 'rb') as f2:
        model = pickle.load(f1)  # 加载预训练好的模型
        vect = pickle.load(f2)  # 加载tf-idf模型
    X = vect.transform([' '.join(clean_segment)]) # 转换为tf-idf矩阵
    label, prob = model.predict(X)
    # prob = model.predict_proba(X).tolist()[0]
#     classes = model.meta_model.classes_.tolist()
    str = ''
    for i,j in zip(dic, prob[0]):
        str += "{}:{:6f}(可信度)\n".format(i, j)
    str = str + '预测主题：' + labels[label[0]]
    text2.insert(0.0, str)

def get_data(e):
    news_t = cmb.get()  # 获取下拉框选择值
    first_url = dict[news_t]
    driver =  webdriver.PhantomJS()
    driver.get(url=first_url)
    selector = etree.HTML(driver.page_source)
    second_urls = selector.xpath('//a[@class="link title"]/@href')
    second_url = random.choice(second_urls)  # 随机选择一个url进入爬取文本内容
    id = re.search('\d+', second_url).group()
    second_url = 'https://www.toutiao.com/a'+id
    driver.get(url=second_url)
    selector = etree.HTML(driver.page_source)
    article_content = ''.join(selector.xpath('//div[@class="article-content"]//p//text()'))
    ts.delete(1.0, END)  # 删除新闻文本框中内容，再插入新的内容
    ts.insert(0.0, article_content)

button1 = Button(t, text="打开文件")
button2 = Button(t, text="保存文件")
button3 = Button(t, text="关键词抽取")
button4 = Button(t, text="文本分类")
button5 = Button(t, text="数据爬取")

button1.place(x=150, y=520)
button2.place(x=250, y=520)
button3.place(x=350, y=520)
button4.place(x=450, y=520)
button5.place(x=550, y=520)

button1.bind("<Button-1>", open_file)
button2.bind("<Button-1>", save_file)
button3.bind("<Button-1>", extract_keyword)
button4.bind("<Button-1>", predict)
button5.bind("<Button-1>", get_data)

t.mainloop()





