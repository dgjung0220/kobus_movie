import requests
import json
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from datetime import datetime

class KobusMovie:
    
    def __init__(self, filename):
        
        self.apikey = 'f5eef3421c602c6cb7ea224104795888'
        self.df = pd.read_excel(filename, engine='openpyxl')

        # delete useless columns
        self.df.drop('순번', axis=1, inplace=True)
        self.df.drop('수입사', axis=1, inplace=True)
        self.df.drop('영화유형', axis=1, inplace=True)
        self.df.drop('국적', axis=1, inplace=True)

        self.df.columns = ['movieName', 'director', 'company', 'distributor', 'openDt', 'type1', 
                            'num_screen', 'num_sales', 'num_seats', 'seoul_sales', 'seoul_seats', 
                            'genre', 'grade', 'type2']
        self.df.dropna(axis=0, inplace=True)
        self.df['openDt'] = self.df['openDt'].apply(lambda x: str(x).split(' ')[0])
        
    def searchMovieCode(self, movieName, directerNm):
        '''
        input: 영화 이름
        output: 영화 코드 
        '''
        
        res = requests.get(f"http://kobis.or.kr/kobisopenapi/webservice/rest/movie/searchMovieList.json?key={self.apikey}&movieNm={movieName}&directorNm={directerNm}")
        text = res.text
        d = json.loads(text)
    
        movieCd = 0
        for b in d['movieListResult']['movieList']:    
            if movieName == b['movieNm']:
                movieCd = b['movieCd']
                    
        return movieCd
    
    def searchMovieInfo(self, movieCd):
        '''
        input: 영화 코드 (movieCd)
        output
        주연 배우 수, 스탭 수, 상영 시간, 주연배우(3명까지)
        '''
                
        res = requests.get(f"http://www.kobis.or.kr/kobisopenapi/webservice/rest/movie/searchMovieInfo.json?key={self.apikey}&movieCd={movieCd}")
        text = res.text
        d = json.loads(text)
        num_actors = len(d['movieInfoResult']['movieInfo']['actors'])
        num_staffs = len(d['movieInfoResult']['movieInfo']['staffs'])
        running_time = d['movieInfoResult']['movieInfo']['showTm']    
        
        top3_actors = [i['peopleNm'] for i in d['movieInfoResult']['movieInfo']['actors']][:3]

        return num_actors, num_staffs, running_time, top3_actors

    def count_movies_before_data(self, directorNm, openStartDt):
        
        # input : 감독 이름 , 개봉일
        # output : 입력 개봉일 이전의 영화 개수, 입력 개봉일 이전 영화 관객 수 평균
        res = requests.get(f"http://www.kobis.or.kr/kobisopenapi/webservice/rest/movie/searchMovieList.json?key={self.apikey}&directorNm={directorNm}")
        text = res.text
        d = json.loads(text)
        
        
        raw_data = [movie_info['openDt'] for movie_info in d['movieListResult']['movieList']]
        open_dates = [datetime.strptime(rd, "%Y%m%d") for rd in raw_data if rd != ''] 
        is_before = [open_date < datetime.strptime(openStartDt, "%Y-%m-%d") for open_date in open_dates]
        
        movieNms = [movie_info['movieNm'] for movie_info in d['movieListResult']['movieList'] if movie_info['openDt'] != '']
        
        if sum(is_before) == 0:
            return 0, 0, 0
        else:
            movies_is_before = np.array(movieNms)[np.array(is_before)]
            num_seats_count = []
            for movieNm in movies_is_before:
                
                seats = self.df.loc[(self.df['movieName'] == movieNm) & (self.df['director'] == directorNm)]['num_seats']
                if len(seats) == 0:
                    num_seats_count.append(0)
                else:
                    num_seats_count.append(int(seats.item()))
            
            avg_seats = np.average(num_seats_count)
            
            return len(movies_is_before), np.average(num_seats_count), np.max(num_seats_count)
    
    def get_people_counts_from_actor(self, peopleNm, filmoNames, openStartDt):
        # input : 배우 이름, 영화 이름, 개봉일
        # output : 입력 개봉일 이전 영화들의 평균 관객수
        
        res = requests.get(f"http://kobis.or.kr/kobisopenapi/webservice/rest/people/searchPeopleList.json?key={self.apikey}&peopleNm={peopleNm}&filmoNames={filmoNames}")
        text = res.text
        d = json.loads(text)

        results = [people_info for people_info in d['peopleListResult']['peopleList'] if (people_info['repRoleNm'] == '배우') & (people_info['peopleNmEn'] != '')]
        if len(results) == 0:
            return 0
        results = [results[0]]
        filmo_names = results[0]["filmoNames"].split("|")
          
        # 영화 리스트 얻기
        prev_movies = [self.df.loc[self.df['movieName'] == filmo_name,:] for filmo_name in filmo_names if len(self.df.loc[self.df['movieName'] == filmo_name,:]) == 1]
        idx = np.array([(datetime.strptime(prev_movie['openDt'].item(), "%Y-%m-%d") <  datetime.strptime(openStartDt, "%Y-%m-%d")) for prev_movie in prev_movies ])
        
        if len(idx ) == 0:
            return 0
        prev_movies_before_date = np.array(prev_movies)[idx]
        cnt_people = [movie[0][8] for movie in prev_movies_before_date if movie[0][8] != 0]
        
        if len(cnt_people) != 0:
            avg_people = int(np.array(cnt_people).mean())
            return avg_people
        else:
            return 0    