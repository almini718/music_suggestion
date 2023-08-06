from flask import Flask, render_template, request, url_for
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

random_samples = None

# 데이터 불러오기
df = pd.read_csv("Music_Suggestion/project/Data.csv", engine="python", encoding="utf-8")


# goto index.html
@app.route("/")
def index():
    return render_template("index.html")


# goto about.html
@app.route("/about")
def hello():
    return render_template("about.html")


# 음악 추천 시작하기 링크를 클릭하면 인덱스 페이지로 리다이렉트합니다.
@app.route("/survey")
def survey():
    global random_samples
    # 랜덤하게 8개의 행 선택
    random_samples = df.sample(8)
    # 랜덤 데이터를 딕셔너리로 변환하여 템플릿에 전달
    data_list = random_samples.to_dict(orient="records")
    return render_template("survey.html", data_list=data_list)


# 라벨링 + 인공지능 학습 + 결과 데이터 전송
@app.route("/result", methods=["POST"])
def result():
    # 데이터 받기
    global random_samples
    user_ratings = [int(request.form[f"rating_{i}"]) for i in range(8)]
    label_encoder = LabelEncoder()

    # x_train, x_test 생성 + x_test 라벨링
    x_test = df.drop(random_samples.index)
    x_test = x_test.drop(columns=["Song Title", "Spotify URI"])
    for column in x_test.select_dtypes(include=[object]):
        x_test[column] = label_encoder.fit_transform(x_test[column])

    # 랜덤 포레스트 모델 학습과 예측
    x_train = random_samples.drop(columns=["Song Title", "Spotify URI"])
    y_train = user_ratings

    # x_train 라벨링
    for column in x_train.select_dtypes(include=[object]):
        x_train[column] = label_encoder.fit_transform(x_train[column])

    model = RandomForestRegressor(n_estimators=1000)
    model.fit(x_train, y_train)

    # 예측 + 예측 결과 정렬
    y_pred = model.predict(x_test)
    sorted_indices = y_pred.argsort()[::-1][:10]
    top_songs = df.iloc[sorted_indices]

    # Song Title과 Spotify URI 남기기
    top_songs = top_songs[["Song Title", "Spotify URI", "Artist"]]

    # ratings 컬럼 추가
    top_songs["ratings"] = y_pred[sorted_indices]

    # top_songs 데이터 전달
    top_songs_data = top_songs.to_dict(orient="records")
    return render_template("result.html", top_songs=top_songs_data)


if __name__ == "__main__":
    app.run(debug=True, port=7777)
