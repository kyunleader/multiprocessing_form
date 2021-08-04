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
