# public-late-night-pharmacies

## 2022 통계데이터 분석·활용대회

**경희대학교 소프트웨어융합학과 문지원**

**경희대학교 소프트웨어융합학과 박재훈**

**경희대학교 소프트웨어융합학과 최인열**

---

### git clone

```git
git clone https://github.com/ChoiInYeol/public-late-night-pharmacies.git
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
```

## 참조

- colab.research.google.com/drive/1GCGwWWrgXB6RupXQC1ZUvjM7Oq8rKM9U?usp=sharing (제주 공간분석 1)
- github.com/Hwan-I/Dacon_Analyze_Jejuspace (제주 공간분석 2)
- compas.lh.or.kr/gis?pageIndex=1&pageSize=10&searchText=&searchKey=both (COMPAS 공간분석 활용)
- github.com/wansook0316/GoyangCityOptimalBicycleStationSuggestion (COMPAS 고양시 공공자전거 입지선정)
- Etc...
