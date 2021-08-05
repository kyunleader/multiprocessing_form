'''

db update 중, 수기 매핑을 하기전에 포스터를 비교하여 수기매핑의 수고를 덜어주기 위해 만들었습니다.

new_codemapped_(site).xlsx 에 있는 포스터(image_url)와
imdb에서 제목을 검색하여 나온 포스터를 비교하여
점수(공통점 개수)를 통해 매핑을 시도합니다.

data set 에 image_url 칼럼이 없으면 안됩니다.

'''

# import
import cv2
import requests
import time
import numpy as np
import matplotlib.pyplot as plt

from strsimpy.levenshtein import Levenshtein
from bs4 import BeautifulSoup as bs
from tqdm import tqdm
from fake_useragent import UserAgent
from traceback import print_exc
import json



# 엑셀(new_codemapped_(site).xlsx)에 있는 이미지 가져오기. 한 행 기준
def excel_poster_load(excel_image_url):
    '''
    한 행을 기준으로 함수를 만들었습니다.
    실제 사용은 for문을 이용하여 돌리는 것을 권장합니다.

    :param excel_image_url: 엑셀의 image_url 칼럼의 데이터
    :return: image 데이터
    plt.imshow(img),plt.show()를 사용하여 이미지가 잘 불러와 졌는지 확인 합니다.
    '''
    byte = requests.get(excel_image_url).content
    ar = np.frombuffer(byte, np.uint8)
    img = cv2.imdecode(ar, cv2.IMREAD_GRAYSCALE)
    return img


# 이미지 보기
def image_show(img):
    '''
    excel_poster_load나 get_image에서 나온 결과값을 프린트해서 보여주는 함수입니다.
    :param img: array 모양의 이미지
    :return: 없음 (image print)
    '''
    plt.imshow(img), plt.show()


# 엑셀에 있는 타이틀을 imdb에 검색하여 검색 결과의 url 가져오기
def imdb_search_result_url(title, max=30):
    '''
    영화의 제목을 기준으로 imdb에 검색을 한 후 해당 결과의 url들을 가져옵니다.

    :param title: 엑셀의 title 칼럼의 데이터
    :param max: 상위 최대 몇개의 결과를 비교할 것인이 정함. (너무 많으면 시간이 오래 걸림)
    :return: url들을 list형태로 가져 옵니다.
    '''
    title = str(title)
    if '(' in title:
        title = title[:title.find('(')]  # 타이틀의 괄호부분 제거
    title = title.strip()
    title = title.replace('  ', ' ')
    title = title.replace(':', ' ')
    title = title.replace(' ', '+')
    search_url = 'https://imdb.com/find?q=' + str(title) + '&s=tt&ref_=fn_al_tt_mr'
    try:
        html = requests.get(search_url).text
        soup = bs(html, 'html.parser')
        root = soup.select('table')[0].select('tr')

        if len(root) > max:
            root = root[:max]

        url_l = []
        for i in range(len(root)):
            if ('TV Series' in root[i].text) or ('TV Episode' in root[i].text) or ('TV Mini-Series' in root[i].text):
                continue
            url_l.append(root[i].select('a')[0]['href'])

        url_l = ['https://imdb.com/' + str(i) for i in url_l]
    except (requests.ConnectionError,
            requests.exceptions.ReadTimeout,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectTimeout) as e:
        print(e)
        url_l = []
        time.sleep(30)
    except:
        print(title, ' 검색결과 없음 ')
        url_l = []

    return url_l


# 해당 url 사이트의 포스터 가져오기
def get_image(url):
    try:
        html = requests.get(url).text
        soup = bs(html, 'html.parser')

        if soup.select('.poster'):  # 수정 필요
            image_url = soup.select('.poster')[0].select('img')[0]['src']
        else:
            image_url = soup.select('.ipc-image')[0]['src']

        byte = requests.get(image_url).content
        ar = np.frombuffer(byte, np.uint8)
        img = cv2.imdecode(ar, cv2.IMREAD_GRAYSCALE)
    except (requests.ConnectionError,
            requests.exceptions.ReadTimeout,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectTimeout) as e:
        print(e)
        img = None
        time.sleep(30)
    return img


# 포스터(이미지) 비교하기
def image_match_score(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matchs = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matchs:
        if m.distance < 0.45 * n.distance:  # 숫자 조절 하기
            good.append([m])
    return len(good)


# 포스터(이미지) 비교한 것 시각화 하기
def image_match_show(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matchs = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matchs:
        if m.distance < 0.45 * n.distance:  # 숫자 조절 하기
            good.append([m])

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    plt.imshow(img3), plt.show()
    return img3


# 감독 맞으면 맞는것으로 판단하기
def director_matching(imdb_url, directors):
    '''
    감독이름이 같으면 해당 url의 imdb_id를 가져온다
    :param imdb_url: main.py에서 작업한 imdb_url(이중 리스트)
    :param directors: url_data의 director만 list or url_data['director']로 가져와야 함
    :return: 매칭된것은 아이디로 나오고, 매칭이 안된한것은 빈칸으로 나옴
    '''
    levenshtein = Levenshtein()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'}

    imdb_id2 = []
    for sites, data_director in tqdm(zip(imdb_url, directors), total=len(imdb_url)):

        # time.sleep(2.5)
        if not sites:
            print('리스트 없음')
            imdb_id2.append('')
            continue

        matching = False

        for url in sites:
            if str(data_director) == 'nan':
                print('감독이름 없음')
                imdb_id2.append('')
                break
            try:
                try:
                    print('리퀘스트 준비')
                    html = requests.get(url, headers=headers).text
                    print('리퀘스트 됨')
                    soup = bs(html, 'html.parser')
                    imdb_director = soup.select('.credit_summary_item > a')[0].text
                    print(imdb_director)
                except IndexError:
                    try:
                        time.sleep(1)
                        html = requests.get(url, headers=headers).text
                        soup = bs(html, 'lxml')
                        imdb_director = soup.select('.credit_summary_item > a')[0].text
                    except IndexError:
                        time.sleep(1)
                        html = requests.get(url, headers=headers).text
                        soup = bs(html, 'lxml')
                        imdb_director = soup.select('.credit_summary_item > a')[0].text
            except:
                print('크롤링 실패')
                continue

            try:
                matching = levenshtein.distance(imdb_director, data_director) < 3
                if matching:
                    imdb_id2.append(str(url)[str(url).find('title/') + 6:-1])
                    print(f'감독이름 매칭 => {imdb_director} : {data_director}')
                    print('영화 url => ' + str(url))
                    break
                else:
                    print('매칭안됨')
            except:
                print_exc()
                print('다 안됨')

        if not matching:
            imdb_id2.append('')  # 5개 다 매칭이 안된경우

    return imdb_id2


# 추가해야할 것

# 사진 ajax 방법으로 업데이트
# 구글 크롤링으로 검수하는 방법도 추가


def director_matching_dic(imdb_url, directors):
    '''
    감독이름이 같으면 해당 url의 imdb_id를 가져온다
    :param imdb_url: main.py에서 작업한 imdb_url(이중 리스트)
    :param directors: url_data의 director만 list or url_data['director']로 가져와야 함
    :return: 매칭된것은 아이디로 나오고, 매칭이 안된한것은 빈칸으로 나옴
    '''
    levenshtein = Levenshtein()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'}

    imdb_id2 = {}
    for n, (sites, data_director) in tqdm(enumerate(zip(imdb_url, directors)), total=len(imdb_url)):

        time.sleep(1)
        if not sites:
            print('리스트 없음')
            imdb_id2[str(data_director) + f'/{n}'] = ""
            continue

        matching = False

        for url in sites:
            if str(data_director) == 'nan':
                print('감독이름 없음')
                imdb_id2[str(data_director) + f'/{n}'] = ""
                break
            try:
                html = requests.get(url, headers=headers).text
                soup = bs(html, 'html.parser')
                a2 = soup.select('body > script')[1].text
                middle_v = list(json.loads(a2)['props']['urqlState'].keys())[0]
                imdb_director = json.loads(a2)['props']['urqlState'][middle_v] \
                    ['data']['title']['directors'][0]['credits'][0] \
                    ['name']['nameText']['text']
                print(imdb_director)
            except:
                print('크롤링 실패')
                continue

            try:
                matching = levenshtein.distance(imdb_director, data_director) < 3
                if matching:
                    imdb_id2[str(data_director) + f'/{n}'] = str(url)[str(url).find('title/') + 6:-1]
                    print(f'감독이름 매칭 => {imdb_director} : {data_director}')
                    print('영화 url => ' + str(url))
                    break
                else:
                    print('매칭안됨')
            except:
                print_exc()
                print('다 안됨')


        if not matching:
            imdb_id2[str(data_director) + f'/{n}'] = ""  # 5개 다 매칭이 안된경우


    return imdb_id2


def director_matching_dic_retry(imdb_url, directors):
    '''
    retry 용
    감독이름이 같으면 해당 url의 imdb_id를 가져온다
    :param imdb_url: main.py에서 작업한 imdb_url(이중 리스트)
    :param directors: url_data의 director만 list or url_data['director']로 가져와야 함
    :return: 매칭된것은 아이디로 나오고, 매칭이 안된한것은 빈칸으로 나옴
    '''
    levenshtein = Levenshtein()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'}

    imdb_id2 = {}
    for n, (sites, data_director) in tqdm(enumerate(zip(imdb_url, directors)), total=len(imdb_url)):

        # time.sleep(2.5)
        if not sites:
            print('리스트 없음')
            imdb_id2[str(data_director) + f'/{n}'] = ""
            continue

        matching = False

        for url in sites:
            if str(data_director) == 'nan':
                print('감독이름 없음')
                imdb_id2[str(data_director) + f'/{n}'] = ""
                break
            try:
                html = requests.get(url, headers=headers).text
                soup = bs(html, 'html.parser')
                a2 = soup.select('body > script')[1].text
                middle_v = list(json.loads(a2)['props']['urqlState'].keys())[0]
                imdb_director = json.loads(a2)['props']['urqlState'][middle_v] \
                    ['data']['title']['directors'][0]['credits'][0] \
                    ['name']['nameText']['text']
                print(imdb_director)
            except:
                print('크롤링 실패')
                raise RuntimeError
                continue

            try:
                matching = levenshtein.distance(imdb_director, data_director) < 3
                if matching:
                    imdb_id2[str(data_director) + f'/{n}'] = str(url)[str(url).find('title/') + 6:-1]
                    print(f'감독이름 매칭 => {imdb_director} : {data_director}')
                    print('영화 url => ' + str(url))
                    break
                else:
                    print('매칭안됨')
                    raise RuntimeError
            except:
                print_exc()
                print('다 안됨')
                raise RuntimeError

        if not matching:
            imdb_id2[str(data_director) + f'/{n}'] = ""  # 5개 다 매칭이 안된경우
            raise RuntimeError

    return imdb_id2