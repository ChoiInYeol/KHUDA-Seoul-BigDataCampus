# public-late-night-pharmacies

## 2022 서울시 빅데이터 캠퍼스 멘토링 3기

- [ ] **경희대학교 소프트웨어융합학과 김수빈**
- [ ] **경희대학교 소프트웨어융합학과 박재훈**
- [ ] **경희대학교 소프트웨어융합학과 정다영**
- [ ] **경희대학교 소프트웨어융합학과 최인열**

---

### git clone

```git
git clone https://github.com/ChoiInYeol/KHUDA-Seoul-BigDataCampus.git
```

### git 강제 pull 방법

```git
git fetch --all
git reset --hard origin/main
git pull origin main
```

### Python 가상환경 설정

cd 명령어를 이용하여 로컬 저장소로 이동 후

```bash
python -m venv {가상환경 이름}
cd ./{설정한 가상환경 이름}/Scripts
activate
```

## Requirement install

```shell
pip install -r requirements.txt
pipwin install -r requirements2.txt
```

```plain text
pandas
numpy
matplotlib
seaborn
tqdm
statsmodels
parmap
sklearn
tensorflow
xgboost
datetime
ipykernel
bs4
pylint
jupyter
requests
pipwin
geopandas
libpysal
matplotlib_scalebar
ortools
spaghetti
```

## 참조

- [제주 공간분석 예제 1](colab.research.google.com/drive/1GCGwWWrgXB6RupXQC1ZUvjM7Oq8rKM9U?usp=sharing)
- [제주 공간분석 예제 2](github.com/Hwan-I/Dacon_Analyze_Jejuspace)
- [COMPAS 공간분석 활용](compas.lh.or.kr/gis?pageIndex=1&pageSize=10&searchText=&searchKey=both)
- [COMPAS 고양시 공공자전거 입지선정](github.com/wansook0316/GoyangCityOptimalBicycleStationSuggestion)
- [서울빅캠 최우수상 지하철 입지선정](https://github.com/panghyuk/BigdataCampus)
- [서울빅캠 국공립 어린이집 입지선정](https://github.com/heejinADP/Seoul_Bigdata_Competition)
- [서울빅캠 우수상 우리동네키움센터 입지선정](https://github.com/jyshin0926/BigDataCampusContest)
- [공공빅데이터인턴십 이동노동자 간이쉼터 입지선정](https://github.com/DonghyunAnn/Gbig-Hackathon)
- [서울시 열섬현상 완화를 위한 녹지 및 바람길 입지 선정](https://github.com/nseunghee97/heat_island_seoul)
- [통계데이터분석활용대회 전통시장 DT 활용 방안](https://github.com/jjonhwa/Policy-to-utilize-DT-in-traditional-markets)
- [스타벅스출점전략을 이용한 자영업자 출점입지 추천 모델](https://github.com/donghwan2/location_analysis/blob/master/Starbucks.ipynb)
- [대구빅데이터분석경진대회 대상 공공도서관 입지 선정](https://github.com/yoonhyeyoon/DaeguBigdataContest)

---

### 데이터셋

[재훈이를 위한 데이터셋 통합 정리](https://docs.google.com/spreadsheets/d/1vDCOE8dpheTkEFP9GXWKvH5jLpMGtzSVBk8jVdEzJ78/edit?usp=sharing_)

#### 다영

- [서울시 안전상비의약품 판매업소 위치](http://data.seoul.go.kr/dataList/OA-16483/S/1/datasetView.do)
- [서울시 응급실 위치](http://data.seoul.go.kr/dataList/OA-20338/S/1/datasetView.do)
- [서울시 약국 위치(일반 약국 포함)](http://data.seoul.go.kr/dataList/OA-20402/S/1/datasetView.do)
- [서울시 주거 밀집 구역(취약층 거주)](http://data.seoul.go.kr/dataList/10727/S/2/datasetView.do)

#### 수빈

- 고민중

#### 인열

- [전국 심야약국, 응급의료회 e-gen](https://www.e-gen.or.kr/egen/search_pharmacy.do?searchType=general)
- [전국 시군구/읍면동 데이터](http://www.gisdeveloper.co.kr/?p=2332)

#### 재훈

- 고민중

---

## 인터넷 레퍼런스

### 약국 관련

- [처방전 없이 살 수 있는 의약품](https://www.pharm114.or.kr/common_files/sub3_page1.asp)
- [서울시 공공 야간약국 지정 운영 안내](https://news.seoul.go.kr/welfare/archives/522756)
- [서울시,일차의료기관과 연계한 '공공 야간 약국' 추진, 2020년 8월](https://www.yakup.com/news/index.html?mode=view&nid=247631)
- [서울특별시 공공 야간·휴일 일차의료 활성화 지원에 관한 조례 만화](blog.naver.com/seoulcouncil/222123186282)
- [국립중앙의료원 연구보고서 및 통계집](https://www.ppm.or.kr/board/thumbnailList.do?MENUID=A04030000)
  
### 개발 관련

- [공간정보 데이터의 구성 및 기본 이해하기](https://yganalyst.github.io/spatial_analysis/spatial_analysis_1/)
- [자판기를 어디에 설치할 것인가? _ MCLP 모델링, 2020 빅콘](https://minkithub.github.io/2020/10/05/bicgon5/)