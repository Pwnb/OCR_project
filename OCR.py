#!/usr/bin/env python
# coding: utf-8

# # Environment Setup

# ## Once

# In[1]:


from PIL import Image
import pytesseract

# Change output format
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
#InteractiveShell.ast_node_interactivity = "last_expr"

# Import packages
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.cluster import KMeans
import json
import requests
#!pip install folium
#import folium
import random
import imageio
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
import matplotlib.patches as patches
import matplotlib.colors as colors
import math
from io import BytesIO
#from tabulate import tabulate
import time
#from chinese_calendar import is_workday, is_holiday
#from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#from xgboost import plot_importance
from PIL import Image, ImageEnhance
import cv2
import re
# Other settings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns',None)
#pd.set_option('display.max_rows', None)
#sns.set(font='SimHei')

# -*- coding: utf-8 -*- 
#import cx_Oracle
import os
#os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
#from collections import Counter
# df.to_pickle('./Cache/cache_df_nd_2019.pkl')
# pd.read_pickle('samples')

def see(df):
    display(df.head(2))
    print(df.shape)

def see_null(df):
    col_list = df.columns
    print('空值情况：')
    for i in col_list:
        null = df[i].isnull().sum()
        print(i+': '+str(null))
        


# ## Control

# In[2]:


#InteractiveShell.ast_node_interactivity = "last_expr"
InteractiveShell.ast_node_interactivity = "all"
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#np.set_printoptions(suppress=False)
#os.chdir('/project/datadir/wnb')
#weather = pd.read_csv('./Data/weather.csv', encoding = 'utf-16', delimiter = '\t')
np.set_printoptions(suppress=True)  # 取消科学计数法输出


# # Use TR package

# In[ ]:


# coding: utf-8
import tr
import sys, cv2, time, os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

#print("recognize", tr.recognize("imgs/line.png"))
print("recognize", tr.recognize("E:/work/code/check_web/data/icon/icon_chima4.png"))

# img_path = "imgs/id_card.jpeg"
img_path = "E:/work/code/ocr/新增-商品手册/784381.jpg"
# img_path = "imgs/name_card.jpg"
# img_path = "E:/work/code/check_web/data/icon/temp_tishi41.png"

img_pil = Image.open(img_path)
try:
    if hasattr(img_pil, '_getexif'):
        # from PIL import ExifTags
        # for orientation in ExifTags.TAGS.keys():
        #     if ExifTags.TAGS[orientation] == 'Orientation':
        #         break
        orientation = 274
        exif = dict(img_pil._getexif().items())
        if exif[orientation] == 3:
            img_pil = img_pil.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img_pil = img_pil.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img_pil = img_pil.rotate(90, expand=True)
except:
    pass

MAX_SIZE = 1600
if img_pil.height > MAX_SIZE or img_pil.width > MAX_SIZE:
    print('adasdasd')
    scale = max(img_pil.height / MAX_SIZE, img_pil.width / MAX_SIZE)

    new_width = int(img_pil.width / scale + 0.5)
    new_height = int(img_pil.height / scale + 0.5)
    img_pil = img_pil.resize((new_width, new_height), Image.ANTIALIAS)

color_pil = img_pil.convert("RGB")
gray_pil = img_pil.convert("L")

img_draw = ImageDraw.Draw(color_pil)
colors = ['red', 'green', 'blue', "purple"]

t = time.time()
n = 1
for _ in range(n):
    tr.detect(gray_pil, flag=tr.FLAG_RECT)
print("time", (time.time() - t) / n)

results = tr.run(gray_pil, flag=tr.FLAG_ROTATED_RECT)

for i, rect in enumerate(results):
    cx, cy, w, h, a = tuple(rect[0])
    print(i, "\t", rect[1], rect[2])
    box = cv2.boxPoints(((cx, cy), (w, h), a))
    box = np.int0(np.round(box))

    for p1, p2 in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        img_draw.line(xy=(box[p1][0], box[p1][1], box[p2][0], box[p2][1]), fill=colors[i % len(colors)], width=2)

color_pil.show()


# In[ ]:





# # Baidu API

# In[ ]:


# 获取token

# encoding:utf-8
import requests 

# client_id 为官网获取的AK， client_secret 为官网获取的SK
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=yourstring&client_secret=yourstring'
response = requests.get(host)
if response:
    print(response.json())


# ## result

# In[15]:


# encoding:utf-8

import requests
import base64

# 文件地址
path = '/Users/bryan/Documents/MetersBonwe/Data/pics_sample'
file_names = os.listdir(path)
file_names.remove('.DS_Store')
file_index = 1
file = path+'/'+file_names[file_index]

def ocr(file):
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general"
    # 二进制方式打开图片文件
    f = open(file, 'rb')
    img = base64.b64encode(f.read())

    params = {"image":img}
    access_token = '24.7b4457906aaa2722d3deae375c0bdf01.2592000.1615451791.282335-23654983'
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    if response:
        print (response.json())
    ocr_result = response.json()
    
    return ocr_result

ocr_result_list = []
for i in file_names:
    file = path+'/'+i
    ocr_result = ocr(file)
    a = dict(file = i, ocr_result = ocr_result)
    ocr_result_list.append(a)
ocr_result_list


# ## Result optimization - correct position (Optional)

# In[9]:


result_np = np.array(ocr_result_list)
np.save('samples_ocr_result.npy', result_np)

test=np.load('samples_ocr_result.npy')
test=test.tolist()

image = Image.open(file)
image = image.convert("RGB")

print('图:')
display(image)


# In[43]:


words = [i['words'] for i in ocr_result['words_result']]
key_words = ['批次','卖点','面料','版型','货架号','尺寸范围','尺码范围','设计','版单编号','厚薄','弹力','手感']

def scan_pos(words):
    output = []
    for i in key_words:
        find_result = words.find(i)
        if find_result!=-1:
            output.append(find_result)
    return output

def arrange_words(words):
    key_words = ['批次','卖点','面料','版型','货架号','尺寸范围','尺码范围','设计','版单编号','厚薄','弹力','手感']
    form_words = []
    unform_words = []
    for word in words:    
        m = re.compile('^.{2,4}-\d{2}').findall(word)
        if len(m)!=0:
            for i in m:
                form_words.append(i)
            continue
        m = re.match(r'\d{6}', word)
        if m is not None:
            form_words.append(word)
            continue

        pos_list = scan_pos(word)

        if len(pos_list)==0:
            unform_words.append(word)
        if len(pos_list)==1:
            if pos_list[0]==0:
                form_words.append(word)
            if pos_list[0]!=0:
                unform_words.append(word[0:pos_list[0]])
                form_words.append(word[pos_list[0]:])
        if len(pos_list)>=2:
            if pos_list[0]==0:
                for i in range(len(pos_list)-1):
                    form_words.append(word[pos_list[i]:pos_list[i+1]])
                form_words.append(word[pos_list[-1]:])
            if pos_list[0]!=0:
                unform_words.append(word[0:pos_list[0]])
                for i in range(len(pos_list)-1):
                    form_words.append(word[pos_list[i]:pos_list[i+1]])
                form_words.append(word[pos_list[-1]:])
    return [words,form_words, unform_words]
#arrange_words(words)
# words
# form_words
# unform_words


# In[42]:


scan_pos('尺码范围:165/88A-185/104B')


# In[37]:


image = Image.open(file)
image = image.convert("RGB")

print('图:')
display(image)


# In[23]:


ocr_result_list


# In[44]:


output_list = []

for i in ocr_result_list:
    words = [j['words'] for j in i['ocr_result']['words_result']]
    words = arrange_words(words)
    a = dict(file = i['file'], ocr_result = words)
    output_list.append(a)

output_list


# In[45]:


output_list = []

for i in ocr_result_list:
    words = [j['words'] for j in i['ocr_result']['words_result']]
    #words = arrange_words(words)
    a = dict(file = i['file'], ocr_result = words)
    output_list.append(a)

output_list


# In[98]:


#output_list = []
result = pd.DataFrame(columns=['pic_name','ocr_result'])

for i in ocr_result_list:
    words = [j['words'] for j in i['ocr_result']['words_result']]
    #words = arrange_words(words)
    result = result.append(pd.DataFrame({'pic_name':i['file'], 'ocr_result':words}),ignore_index=True)

result['pic_name'][result.duplicated(subset=['pic_name'])] = np.nan
result.to_excel('../Output/pic_samples_ocr.xlsx')


# In[ ]:


result


# In[92]:


result['pic_name'][0]=1


# In[ ]:





# In[ ]:





# # Google tesseract

# ## One picture OCR with 4 kinds (config)

# In[324]:


path = '/Users/bryan/Documents/MetersBonwe/Data/pics_sample'
file_names = os.listdir(path)
file_index = 1
file = path+'/'+file_names[file_index]
image = Image.open(file)
image = image.convert("RGB")

print('图:')
display(image)

for i in [1,3,6,12]:
    con = '--psm '+str(i)


    print('OCR结果:\n')
    text = pytesseract.image_to_string(image, lang='chi_sim', config=con)
    print(text)
    print('\n\n')


# ## try to cut the picture and better the result

# ### scan picture v1

# In[179]:


# 算分割点
def v_scan(file):
    img = cv2.imread(file)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,w,c = rgb.shape

    white = True
    white_list = []
    for j in range(h):
        fail=False
        for i in range(w):
            if img[j,i,:].tolist() != [255,255,255]:
                #print('Failed')
                if white:
                    white = False
                    white_list.append(j)
                fail=True
                break  
        if not fail:
            if not white:
                white = True
                white_list.append(j)
    cond = (np.array(white_list[1:]) - np.array(white_list[:-1]))>5
    div_ht = np.array(white_list[:-1])[cond]
    
    return div_ht

path = '/Users/bryan/Documents/MetersBonwe/Data/pics_sample'
file_names = os.listdir(path)
file_index = 2
file = path+'/'+file_names[file_index]
#file = '/Users/bryan/Documents/MetersBonwe/Data/ocr_test_7.png'


image = Image.open(file)
image = image.convert("RGB")
display(image)
v_scan(file)


# ### scan picture v2

# In[231]:


# 算分割点
def v_scan(file): #v2
    img = cv2.imread(file)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,w,c = rgb.shape

    current_status = 'white'
    ch_list = []
    for i in range(h):
        for j in range(w):
            if img[i,j,:].tolist() != [255,255,255]: #遇到黑色
                if current_status == 'white': #此时是白
                    #print(str(i)+'遇到黑色 现在是白色')
                    ch_list.append(i)
                    current_status = 'black'
                    break
                if current_status == 'black': #此时是黑
                    #print(str(i)+'遇到黑色 现在是黑色')                    
                    break
        else:    
            if current_status == 'white': #此时是白
                #print(str(i)+'遇到白色 现在是白色')
                continue
            if current_status == 'black': #此时是黑
                #print(str(i)+'遇到白色 现在是黑色')
                ch_list.append(i)
                current_status = 'white'
                continue
    
    cond = (np.array(ch_list[1:]) - np.array(ch_list[:-1]))>5
    div_ht = np.array(ch_list[:-1])[cond]
    
    return div_ht

path = '/Users/bryan/Documents/MetersBonwe/Data/pics_sample'
file_names = os.listdir(path)
file_index = 2
file = path+'/'+file_names[file_index]
#file = '/Users/bryan/Documents/MetersBonwe/Data/ocr_test_7.png'


image = Image.open(file)
image = image.convert("RGB")
display(image)
v_scan(file)


# ### h_scan v2

# In[ ]:


# 横扫描
def h_scan(image): #v2
    img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)  

    #img = cv2.imread(file)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,w,c = rgb.shape

    current_status = 'white'
    ch_list = []
    for j in range(w):
        for i in range(h):
            if img[i,j,:].tolist() != [255,255,255]: #遇到黑色
                if current_status == 'white': #此时是白
                    #print(str(i)+'遇到黑色 现在是白色')
                    ch_list.append(j)
                    current_status = 'black'
                    break
                if current_status == 'black': #此时是黑
                    #print(str(i)+'遇到黑色 现在是黑色')                    
                    break
        else:    
            if current_status == 'white': #此时是白
                #print(str(i)+'遇到白色 现在是白色')
                continue
            if current_status == 'black': #此时是黑
                #print(str(i)+'遇到白色 现在是黑色')
                ch_list.append(j)
                current_status = 'white'
                continue
    #print(ch_list)
    cond = (np.array(ch_list[1:]) - np.array(ch_list[:-1]))
    
    t = np.array(ch_list[:-1])[cond]
    b = t[int(len(t)/2)]
    return cond
#     if len(t)<=2:
#         return 0
#     else:
#         return int(b/2)

path = '/Users/bryan/Documents/MetersBonwe/Data/pics_sample'
file_names = os.listdir(path)
file_index = 2
#file = path+'/'+file_names[file_index]
file = '/Users/bryan/Documents/MetersBonwe/Data/ocr_test_10.png'


image = Image.open(file)
image = image.convert("RGB")
#display(image)
image
h_scan(image)


# ## OCR after scan

# ### no left and right split

# In[158]:


path = '/Users/bryan/Documents/MetersBonwe/Data/pics_sample'
file_names = os.listdir(path)
file_index = 2
file = path+'/'+file_names[file_index]
image = Image.open(file)
image = image.convert("RGB")


# 算分割点
img = cv2.imread(file)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h,w,c = rgb.shape

# for i in range(w):
#     if img[:,i,:].tolist() == np.array([0]*3*h).reshape((h,3)).tolist():
#         print(i)
#         break

# for i in range(1):
#     for j in range(h):
#         img[j,i,:] = np.array([255,255,255])

white = True
white_list = []
for j in range(h):
    fail=False
    for i in range(w):
        if img[j,i,:].tolist() != [255,255,255]:
            #print('Failed')
            if white:
                white = False
                white_list.append(j)
            fail=True
            break  
    if not fail:
        if not white:
            white = True
            white_list.append(j)
cond = (np.array(white_list[1:]) - np.array(white_list[:-1]))>5
div_ht = np.array(white_list[:-1])[cond]

#分割图 
w,h = image.size
con = '--psm 6'
# print('图:')
# div_image = image.crop((0,0,w,div_ht[0])) #分割第一个
# text = pytesseract.image_to_string(div_image, lang='chi_sim', config=con)
# display(div_image)
# print('文:')
# print(text)

for i in range(0, len(div_ht)-1, 2): #分割中间
    div_image = image.crop((0,div_ht[i],w,div_ht[i+1]))

    print('图:')
    text = pytesseract.image_to_string(div_image, lang='chi_sim', config=con)
    display(div_image)
    print('文:')
    print(text)
    

    
print('图:')
div_image = image.crop((0,div_ht[-1],w,h)) #分割最后一个
text = pytesseract.image_to_string(div_image, lang='chi_sim', config=con)
display(div_image)
print('文:')
print(text)


# ### split with left and right

# In[162]:


path = '/Users/bryan/Documents/MetersBonwe/Data/pics_sample'
file_names = os.listdir(path)
file_index = 2
file = path+'/'+file_names[file_index]
image = Image.open(file)
image = image.convert("RGB")


# 算分割点
img = cv2.imread(file)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h,w,c = rgb.shape

# for i in range(w):
#     if img[:,i,:].tolist() == np.array([0]*3*h).reshape((h,3)).tolist():
#         print(i)
#         break

# for i in range(1):
#     for j in range(h):
#         img[j,i,:] = np.array([255,255,255])

white = True
white_list = []
for j in range(h):
    fail=False
    for i in range(w):
        if img[j,i,:].tolist() != [255,255,255]:
            #print('Failed')
            if white:
                white = False
                white_list.append(j)
            fail=True
            break  
    if not fail:
        if not white:
            white = True
            white_list.append(j)
cond = (np.array(white_list[1:]) - np.array(white_list[:-1]))>5
div_ht = np.array(white_list[:-1])[cond]

#分割图 
w,h = image.size
con = '--psm 6'
# print('图:')
# div_image = image.crop((0,0,w,div_ht[0])) #分割第一个
# text = pytesseract.image_to_string(div_image, lang='chi_sim', config=con)
# display(div_image)
# print('文:')
# print(text)

for i in range(0, len(div_ht)-1, 2): #分割中间
    div_image = image.crop((0,div_ht[i],w,div_ht[i+1]))
    w_2,h_2 = div_image.size
    
    div_image_left = div_image.crop((0,0,w_2/2,h_2))
    div_image_right = div_image.crop((w_2/2,0,w_2,h_2))

    print('图 - 左:')
    text = pytesseract.image_to_string(div_image_left, lang='chi_sim', config=con)
    display(div_image_left)
    print('文 - 左:')
    print(text)
    
    print('图 - 右:')
    text = pytesseract.image_to_string(div_image_right, lang='chi_sim', config=con)
    display(div_image_right)
    print('文 - 右:')
    print(text)
    
print('图:')
div_image = image.crop((0,div_ht[-1],w,h)) #分割最后一个
text = pytesseract.image_to_string(div_image, lang='chi_sim', config=con)
display(div_image)
print('文:')
print(text)


# ### split with left and right (store the result in a df)

# In[297]:


path = '/Users/bryan/Documents/MetersBonwe/Data/pics_sample'
file_names = os.listdir(path)
file_index = 4
file = path+'/'+file_names[file_index]
image = Image.open(file)
image = image.convert("RGB")


# 算分割点
img = cv2.imread(file)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h,w,c = rgb.shape

# for i in range(w):
#     if img[:,i,:].tolist() == np.array([0]*3*h).reshape((h,3)).tolist():
#         print(i)
#         break

# for i in range(1):
#     for j in range(h):
#         img[j,i,:] = np.array([255,255,255])

white = True
white_list = []
for j in range(h):
    fail=False
    for i in range(w):
        if img[j,i,:].tolist() != [255,255,255]:
            #print('Failed')
            if white:
                white = False
                white_list.append(j)
            fail=True
            break  
    if not fail:
        if not white:
            white = True
            white_list.append(j)
cond = (np.array(white_list[1:]) - np.array(white_list[:-1]))>5
div_ht = np.array(white_list[:-1])[cond]

#分割图 
w,h = image.size
con = '--psm 6'
# print('图:')
# div_image = image.crop((0,0,w,div_ht[0])) #分割第一个
# text = pytesseract.image_to_string(div_image, lang='chi_sim', config=con)
# display(div_image)
# print('文:')
# print(text)
result = pd.DataFrame(columns=['left','right'])
for i in range(0, len(div_ht)-1, 2): #分割中间
    div_image = image.crop((0,div_ht[i],w,div_ht[i+1]))
    w_2,h_2 = div_image.size
    
    div_image_left = div_image.crop((0,0,w_2/2,h_2))
    div_image_right = div_image.crop((w_2/2,0,w_2,h_2))

    text_left = pytesseract.image_to_string(div_image_left, lang='chi_sim', config=con).replace('\n','    ')
    text_right = pytesseract.image_to_string(div_image_right, lang='chi_sim', config=con).replace('\n','    ')
    result = result.append(pd.DataFrame({'left':[text_left], 'right':[text_right]}),ignore_index=True)

    
div_image = image.crop((0,div_ht[-1],w,h)) #分割最后一个
w_2,h_2 = div_image.size
    
div_image_left = div_image.crop((0,0,w_2/2,h_2))
div_image_right = div_image.crop((w_2/2,0,w_2,h_2))

text_left = pytesseract.image_to_string(div_image_left, lang='chi_sim', config=con)
text_right = pytesseract.image_to_string(div_image_right, lang='chi_sim', config=con)
result = result.append(pd.DataFrame({'left':[text_left], 'right':[text_right]}),ignore_index=True)

display(image)
result


# ### split left and right (with better scan algorithm)

# In[429]:


path = '/Users/bryan/Documents/MetersBonwe/Data/pics_sample'
file_names = os.listdir(path)
file_index = 2
file = path+'/'+file_names[file_index]
image = Image.open(file)
image = image.convert("RGB")


# 算分割点
img = cv2.imread(file)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h,w,c = rgb.shape


white = True
white_list = []
for j in range(h):
    fail=False
    for i in range(w):
        if img[j,i,:].tolist() != [255,255,255]:
            #print('Failed')
            if white:
                white = False
                white_list.append(j)
            fail=True
            break  
    if not fail:
        if not white:
            white = True
            white_list.append(j)
cond = (np.array(white_list[1:]) - np.array(white_list[:-1]))>5
div_ht = np.array(white_list[:-1])[cond]

#分割图 
w,h = image.size
con = '--psm 6'
# print('图:')
# div_image = image.crop((0,0,w,div_ht[0])) #分割第一个
# text = pytesseract.image_to_string(div_image, lang='chi_sim', config=con)
# display(div_image)
# print('文:')
# print(text)
result = pd.DataFrame(columns=['left','right'])
for i in range(0, len(div_ht)-1, 2): #分割中间
    div_image = image.crop((0,div_ht[i],w,div_ht[i+1]))
    half = h_scan(div_image)
    w_2,h_2 = div_image.size

    if half==0:
        text = pytesseract.image_to_string(div_image, lang='chi_sim', config=con).replace('\n','    ')
        result = result.append(pd.DataFrame({'left':[text], 'right':np.nan}),ignore_index=True)
    else:    
        div_image_left = div_image.crop((0,0,half,h_2))
        div_image_right = div_image.crop((half,0,w_2,h_2))
        
        text_left = pytesseract.image_to_string(div_image_left, lang='chi_sim', config=con).replace('\n','    ')
        text_right = pytesseract.image_to_string(div_image_right, lang='chi_sim', config=con).replace('\n','    ')
    
        print('图 - 左:')
        display(div_image_left)
        print('文 - 左:')
        print(text_left)

        print('图 - 右:')
        display(div_image_right)
        print('文 - 右:')
        print(text_right)
        
        result = result.append(pd.DataFrame({'left':[text_left], 'right':[text_right]}),ignore_index=True)

    
div_image = image.crop((0,div_ht[-1],w,h)) #分割最后一个
w_2,h_2 = div_image.size
    
div_image_left = div_image.crop((0,0,w_2/2,h_2))
div_image_right = div_image.crop((w_2/2,0,w_2,h_2))

text_left = pytesseract.image_to_string(div_image_left, lang='chi_sim', config=con)
text_right = pytesseract.image_to_string(div_image_right, lang='chi_sim', config=con)
result = result.append(pd.DataFrame({'left':[text_left], 'right':[text_right]}),ignore_index=True)

display(image)
display(result)


# ## Test a bunch of pictures

# In[97]:


def img_rcg(file):
    image = Image.open(file)
    image = image.convert("RGB")
    # 算分割点
    img = cv2.imread(file)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,w,c = rgb.shape

    white = True
    white_list = []
    for j in range(h):
        fail=False
        for i in range(w):
            if img[j,i,:].tolist() != [255,255,255]:
                #print('Failed')
                if white:
                    white = False
                    white_list.append(j)
                fail=True
                break  
        if not fail:
            if not white:
                white = True
                white_list.append(j)
    cond = (np.array(white_list[1:]) - np.array(white_list[:-1]))>5
    div_ht = np.array(white_list[:-1])[cond]
    white_list

    #分割图 
    w,h = image.size
    con = '--psm 6'
    # print('图:')
    # div_image = image.crop((0,0,w,div_ht[0])) #分割第一个
    # text = pytesseract.image_to_string(div_image, lang='chi_sim', config=con)
    # display(div_image)
    # print('文:')
    # print(text)

    for i in range(0, len(div_ht)-1, 2): #分割中间
        print('图:')
        div_image = image.crop((0,div_ht[i],w,div_ht[i+1])) 
        text = pytesseract.image_to_string(div_image, lang='chi_sim', config=con)
        display(div_image)
        print('文:')
        print(text)

    print('图:')
    div_image = image.crop((0,div_ht[-1],w,h)) #分割最后一个
    text = pytesseract.image_to_string(div_image, lang='chi_sim', config=con)
    display(div_image)
    print('文:')
    print(text)


# In[103]:


path = '/Users/bryan/Documents/MetersBonwe/Data/pics_sample'
file_names = os.listdir(path)
file_names.remove('.DS_Store')

for file_index in range(2,10,1):
    print('PICTURE '+str(file_index)+':')
    file = path+'/'+file_names[file_index]
    img_rcg(file)
    print('\n\n')

