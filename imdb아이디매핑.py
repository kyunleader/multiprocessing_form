# 수기매핑 개선안

# 이미지로 비교해보기
import pandas as pd
import poster_similarity as ps
import multiprocessing
import os
from tqdm import tqdm


# 사이트 이름 설정
site = 'vudu'
n_cha = '16'

# 데이터 불러오기

Desktop = os.path.dirname(os.getcwd())
file_path = Desktop + \
            f'/마이무비 기획/4.Free_streaming_Sites_crawled/result_Crawling/{n_cha}차_Update/2.code_mapping/'

if site == 'tubi':
    # tubi일 경우
    # original_data = pd.read_excel(file_path+f'new_codemapped_{site}.xlsx')
    original_data = pd.read_excel('/Users/mycelebs-it1/Desktop/tubi_매핑필요.xlsx')
    url_data = original_data
    url_data['image_url'] = url_data['image_url'].fillna(0)
    url_data = url_data[url_data['image_url'] != 0]  # image url 정보 없는건 제외
    url_data['imdb_id'] = url_data['imdb_id'].fillna(0)
    url_data = url_data[url_data['imdb_id'] == 0]  # imdb_id 있는건 제외
    excel_url_list = url_data['image_url']

elif site == 'darkmatter':
    original_data = pd.read_excel(file_path+f'new_codemapped_{site}.xlsx')
    url_data = original_data
    url_data['imdb_id'] = url_data['imdb_id'].fillna(0)
    url_data = url_data[url_data['imdb_id'] == 0]  # imdb_id 있는건 제외
    excel_url_list = []
else:
    # 다른사이트일 경우
    original_data = pd.read_excel(file_path+f'new_codemapped_{site}.xlsx')
    url_data = original_data
    url_data['image_url'] = url_data['image_url'].fillna(0)
    url_data = url_data[url_data['image_url'] != 0]  # image url 정보 없는건 제외
    url_data['imdb_id'] = url_data['imdb_id'].fillna(0)
    url_data = url_data[url_data['imdb_id'] == 0]  # imdb_id 있는건 제외
    excel_url_list = url_data['image_url']


# 엑셀 이미지 가져오기
excel_image_list = [ps.excel_poster_load(i) for i in tqdm(excel_url_list)]


# 검색결과 url 가져오기 (title 이용)
imdb_url = [ps.imdb_search_result_url(i, max=5) for i in tqdm(url_data['title'])]

# 이미지가 일치하는 곳의 url 가져오기
most_similarity_url = []
for i, k in tqdm(zip(imdb_url, excel_image_list), total=len(imdb_url)):

    try:
        score = []
        for j in i:
            try:
                score.append(ps.image_match_score(k, ps.get_image(j)))
            except:
                score.append(0)
        if max(score) < 5:
            most_similarity_url.append(0)
        else:
            most_similarity_url.append(i[score.index(max(score))])
    except:
        most_similarity_url.append(0)

# 데이터 병합
imdb_id = [str(i)[str(i).find('title/')+6:-1] for i in most_similarity_url]


# 감독이름으로 찾기

# 재시도 필요할 시
# from retrying import retry
# def wait(attempts, delay):
#     print(f'{attempts}회 재시도')
#     return delay
#
# @retry(stop_max_attempt_number=3, wait_func=wait)
def scrap(args):
    """
    Args:
        args: list format that includes startPoint and endPoint(ex:[1,3])
    Returns:
    """
    # arguments
    StartPoint, EndPoint = args[0], args[1]
    imdb_url_p = imdb_url[StartPoint:EndPoint]
    directors = list(url_data['director'])[StartPoint:EndPoint]
    director_imdb = ps.director_matching_dic(imdb_url_p, directors)
    return director_imdb

def split(num, n=8):
    a_list = []
    c = round(num/n)
    a_list.append((0,c))
    for i in range(1,n-1):
        a_list.append((c*i,c*(i+1)))
    a_list.append((c*(n-1),num+1))
    return a_list

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=8)
    result2 = pool.map(scrap, [i for i in split(len(imdb_url))])
    pool.close()
    pool.join()

#데이터 프레임화 시키기
result_df = pd.DataFrame({'index': [], 'director': [], 'imdb_id2': []})

ind_list = []; di_list = []; k_list = []
for dic in result2:
    for i, k in dic.items():
        di, ind = i.split('/')
        ind_list.append(ind)
        di_list.append(di)
        k_list.append(k)
result_df['index'] = ind_list
result_df['director'] = di_list
result_df['imdb_id2'] = k_list

result_si = pd.concat([url_data.reset_index(drop=True), result_df],axis=1)

if site != 'darkmatter':
    merge_data = pd.concat([url_data['image_url'].reset_index(drop=True),
                            pd.Series(most_similarity_url), pd.Series(imdb_id)], axis=1)

    test_data = pd.merge(original_data, merge_data, on='image_url', how='left')


    result_site = pd.merge(test_data, result_si[['image_url', 'imdb_id2']],
                            on='image_url', how='left')
    # 데이터 이름바꾸기
    result_site.rename(columns={0: 'check_url', 1: 'ids'}, inplace=True)

    # imdb 합치기
    imdb_full = []
    for k, i in result_site[['imdb_id', 'ids', 'imdb_id2']].iterrows():
        if not str(i['imdb_id']).find('tt'):
            imdb_full.append(i['imdb_id'])
        elif not str(i['ids']).find('tt'):
            imdb_full.append(i['ids'])
        elif not str(i['imdb_id2']).find('tt'):
            imdb_full.append(i['imdb_id2'])
        else:
            imdb_full.append("")
    result_site['final_imdb'] = imdb_full
else:
    result_site = pd.merge(original_data, result_si[['darkmatter_id', 'imdb_id2']],
                           on='darkmatter_id', how='left')
    imdb_full = []
    for k, i in result_site[['imdb_id', 'imdb_id2']].iterrows():
        if not str(i['imdb_id']).find('tt'):
            imdb_full.append(i['imdb_id'])
        elif not str(i['imdb_id2']).find('tt'):
            imdb_full.append(i['imdb_id2'])
        else:
            imdb_full.append("")
    result_site['final_imdb'] = imdb_full
    result_site.to_excel(f'kkh_codemapped_{site}.xlsx')

result_site.to_excel(f'kkh_codemapped_{site}.xlsx')


# 연습
'''
# 이미지 비교하기

import cv2
import matplotlib.pyplot as plt
from selenium import webdriver
import time
import requests
import pandas as pd
import openpyxl
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup as bs
import re
from tqdm import tqdm
import numpy as np
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
import urllib.request
# option = Options()
# option.add_argument('--headless')

# 드라이버 열기
# driver = webdriver.Chrome(ChromeDriverManager().install(), options=option)
# driver.implicitly_wait(10)

# 여러개 일때 @@@@@@@@@@@@@@@@@@@

# 엑셀 이미지 가져오기
url_data = pd.read_excel('/Users/mycelebs/Desktop/new_codemapped_popcorn.xlsx')
url_data['image_url'] = url_data['image_url'].fillna(0)
url_data = url_data[url_data['image_url'] != 0]  # image url 정보 없는건 제외
excel_url_list = url_data['image_url']

# a = excel_url_list[5] == 'nan'
# a = True
# url = ['http://imghost.device.screenmedia.net/55536_Guardians_214x306.jpg']
import image_url_to_image
a11 = image_url_to_image.get_file_data_list_with_thread(excel_url_list)
a22 = list(a11.values())
a33 = [image_url_to_image.get_img_np_arr(i) for i in a22]
img_excel = [image_url_to_image.get_img_data_arr(i) for i in a33]
# html_0 = requests.get(url).text
# soup_0 = bs(html_0, 'html.parser')
#
#
# image_ex = driver.find_element_by_css_selector('img')
#
# img = image_ex.get_attribute('src')
#
# urllib.request.urlretrieve(img,'img_excel.jpg')  # 가져오기
# img_excel = cv2.imread('img_excel.jpg')
plt.imshow(img_excel[0]),plt.show()


# imdb 검색결과 url들 가져오기
title = url_data['title'].reset_index(drop=True)  ########## 추후 괄호 제거 코드 추가 하기
url2_list = ['https://imdb.com/find?q='+ str(i) +'&s=tt&ref_=fn_al_tt_mr'.replace(' ','') for i in title]
html_list = [requests.get(url2).text for url2 in tqdm(url2_list)]
soup_list = [bs(html, 'html.parser') for html in tqdm(html_list)]

soup_list[7].select('table')[0].select('tr')
url2_list[7]

imdb_root_list = []
for j, i in enumerate(soup_list):
    try:
        imdb_root_list.append(i.select('table')[0].select('tr'))
    except:
        print(title[j] + ' 검색결과 없음')
        pass


imdb_ids_list = []
for i in tqdm(imdb_root_list):
    ids = []
    if len(i) > 30:
        i = i[0:30]
    for j in range(len(i)):
        ids.append(i[j].select('a')[0]['href'])
        print(i[j].select('a')[0]['href'])
    imdb_ids_list.append(ids)

len(imdb_ids_list[40])

############### 최대 30개 까지 id 나오도록 했음
### 그 이후로 id 마다 이미지 가져와서 매칭 -> 제일 점수 높은 것의 id 가져오기 or 매칭된 이미지 가져오기

imdb_ids_list[4]

imdb_id_url_list = ['https://imdb.com/'+str(i) for i in imdb_ids_list[4]]
htmls = [requests.get(i).text for i in tqdm(imdb_id_url_list)]
soups = [bs(i, 'html.parser') for i in htmls]
img_url_list = []
for i in soups:
    try:
        img_url_list.append(i.select('.poster')[0].select('img')[0]['src'])
    except:
        pass


a1 = image_url_to_image.get_file_data_list_with_thread(img_url_list)
a2 = list(a1.values())
a3 = [image_url_to_image.get_img_np_arr(i) for i in a2]
imdb_imgs = [image_url_to_image.get_img_data_arr(i) for i in a3]

plt.imshow(imdb_imgs[0]),plt.show()


good_point_list = []
for img_imdb in tqdm(imdb_imgs):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_excel, None)
    kp2, des2 = sift.detectAndCompute(img_imdb, None)
    bf = cv2.BFMatcher()
    matchs = bf.knnMatch(des1, des2, k=2)
    good = []
    for m,n in matchs:
        if m.distance < 0.3*n.distance:
            good.append([m])
    good_point_list.append(len(good))
max_value_list.append(good_point_list.index(max(good_point_list)))









imdb_root_list = [i.select('table')[0].select('tr') for i in tqdm(soup_list)]
len(soup_list[21].select('table')[0].select('tr'))
soup_list[21].select('table')[0].select('tr')
[1].select('a')[0]['href']

len(bs(requests.get(url2_list[21]).text, 'html.parser').select('table')[0].select('tr'))
bs(requests.get(url2_list[51]).text, 'html.parser').select('table')[0].select('tr')
soup_list[21]
# imdb_root_list = [soup_list[i].select('table')[0].select('tr') for i in range(len(soup_list))]


imdb_id = []
for i in tqdm(range(len(soup_list))):
    try:
        imdb_id.append(soup_list[i].select('table')[0].select('tr')[0].select('a')[0]['href'])
    except:
        imdb_id.append('none')
len(imdb_id)
imdb_id[0]
# if imdb_root_list > 20:
#     imdb_root_list = imdb_root_list[0:20]

max_value_list = []
for imdb_root in tqdm(imdb_root_list):

    imdb_id = []
    for i in tqdm(range(len(imdb_root_list))):
        try:
            id = imdb_root[i].select('a')[0]['href']
            imdb_id.append(id)
        except:
            pass




# imdb의 사진 가져오기
url3_list = ['https://imdb.com/'+str(i) for i in imdb_id]
htmls = [requests.get(i).text for i in tqdm(url3_list)]
soups = [bs(i, 'html.parser') for i in htmls]
img_url_list = []
for i in soups:
    try:
        img_url_list.append(i.select('.poster')[0].select('img')[0]['src'])
    except:
        pass

# url 이미지 image data로 가져오기
a1 = image_url_to_image.get_file_data_list_with_thread(img_url_list)
a2 = list(a1.values())
a3 = [image_url_to_image.get_img_np_arr(i) for i in a2]
imdb_imgs = [image_url_to_image.get_img_data_arr(i) for i in a3]
# plt.imshow(imdb_imgs[0]),plt.show()

# 이미지 비교하기
# imdb 이미지

good_point_list = []
for img_imdb in tqdm(imdb_imgs):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_excel, None)
    kp2, des2 = sift.detectAndCompute(img_imdb, None)
    bf = cv2.BFMatcher()
    matchs = bf.knnMatch(des1, des2, k=2)
    good = []
    for m,n in matchs:
        if m.distance < 0.3*n.distance:
            good.append([m])
    good_point_list.append(len(good))
max_value_list.append(good_point_list.index(max(good_point_list)))



    # img3 = cv2.drawMatchesKnn(img_excel, kp1, img_imdb, kp2, good, None, flags=2)
# plt.imshow(img3),plt.show()

# img_imdb = imdb_imgs[19]
# sift = cv2.SIFT_create()
# kp1, des1 = sift.detectAndCompute(img_excel, None)
# kp2, des2 = sift.detectAndCompute(img_imdb, None)
# bf = cv2.BFMatcher()
# matchs = bf.knnMatch(des1, des2, k=2)
# good = []
# for m,n in matchs:
#     if m.distance < 0.3*n.distance:
#         good.append([m])
# # good_point_list.append(len(good))
# img3 = cv2.drawMatchesKnn(img_excel, kp1, img_imdb, kp2, good, None, flags=2)
# plt.imshow(img3),plt.show()


def scrap(args):
    """
    Args:
        args: list format that includes startPoint and endPoint(ex:[1,3])
    Returns:
    """
    # arguments
    StartPoint, EndPoint = args[0], args[1]
    imdb_url_p = imdb_url[StartPoint:EndPoint]
    directors = list(url_data['director'])[StartPoint:EndPoint]
    director_imdb = ps.director_matching_dic(imdb_url_p, directors)
    return director_imdb

if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=8)
    result2 = pool.map(scrap,[(0,225),(225,450),
                             (450,675),(675,900),
                              (900,1125),(1125,1350),
                              (1350,1575),(1575,1800)])
    pool.close()
    pool.join()



#데이터 프레임화 시키기

result_df = pd.DataFrame({'index': [], 'director': [], 'imdb_id2': []})

ind_list = []
di_list = []
k_list = []
for dic in result2:
    for i, k in dic.items():
        di, ind = i.split('/')
        ind_list.append(ind)
        di_list.append(di)
        k_list.append(k)
result_df['index'] = ind_list
result_df['director'] = di_list
result_df['imdb_id2'] = k_list
result_df.to_csv('result_df.csv')

result_tubi = pd.concat([url_data.reset_index(drop=True), result_df],axis=1)


result_tubi2 = pd.merge(original_data, result_tubi[['image_url', 'imdb_id2']],
                        on='image_url', how='left')
result_tubi2.to_excel('result_tubi2.xlsx')
imdb_full = []

for k, i in result_tubi2[['imdb_id', '1', 'imdb_id2']][:20].iterrows():
    if not str(i['imdb_id']).find('tt'):
        imdb_full.append(i['imdb_id'])
    elif not str(i['1']).find('tt'):
        imdb_full.append(i['1'])
    elif not str(i['imdb_id2']).find('tt'):
        imdb_full.append(i['imdb_id2'])
    else:
        imdb_full.append("")'''




