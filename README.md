# multiprocessing_form

<ul>
멀티프로세싱을 바로바로 사용하기 위한 폼을 만듭니다. 현재는 기본적인 멀티프로세싱만 가능하며, 범용적으로 사용할 수 있도록 만들어 가는 과정입니다.
</ul>


<ul>
지금은 저만 사용하기 쉽게 만들었지만 코드 개선을 통해 잘 만들어 보도록 노력하는 중입니다. 
</ul>


## imdb매핑
 - 영화 코드를 매핑하는 코드의 일부분입니다. openCV를 이용한 포스터 매칭하는 코드도 같이 포함되어 있으며, MultiProcessing은 감독이름을 매칭하는 과정에서 쓰여집니다.



## 아주 간단한 멀티프로세싱

 - 간단한 멀티 프로세싱 입니다.

 ```python
import time
def count_num(number):
    for i in range(number):
        print(f'{i}번 입니다.')
        time.sleep(1)

import multiprocessing
if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=4)
    result2 = pool.map(count_num, [5,5,5,5])
    pool.close()
    pool.join()
    
  >>> 0번 입니다.
      0번 입니다.
      0번 입니다.
      0번 입니다.
      1번 입니다.
      1번 입니다.
      1번 입니다.
      1번 입니다.
      2번 입니다.
      2번 입니다.
      2번 입니다.
      2번 입니다.
      3번 입니다.
      3번 입니다.
      3번 입니다.
      3번 입니다.
      4번 입니다.
      4번 입니다.
      4번 입니다.
      4번 입니다.
 ```
