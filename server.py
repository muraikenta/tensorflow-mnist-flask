"""
Flaskサーバー
"""

from io import BytesIO
import base64

from flask import Flask, render_template, request
from PIL import Image
import numpy as np

from model import build_model, preprocess_data
from class_names import class_names_ja

APP = Flask(__name__)

# アップロードサイズを1MBに制限
APP.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

@APP.route("/", methods=['GET', 'POST'])
def index():
    """
    POSTでリクエストを受け付けた場合、画像をtensorflowで分析して予測結果を返す
    """
    if request.method == 'GET':
        return render_template('index.html')

    # NOTE: POST request
    file = request.files['file']
    file_body = file.read()
    # tensorflowに読み込ませるように画像を変換
    image_data = preprocess_data(255 - np.array(
        Image.open(BytesIO(file_body)).resize((28, 28)).convert('L')
    ))
    # TODO: 毎回modelを訓練するのではなく、保存したものを使い回すようにする
    # NOTE: なぜかトップレベルでmodelを定義すると"Tensor(...) is not an element of this graph."というエラーが出る
    model = build_model()
    # 判定
    prediction = model.predict(np.array([image_data]))[0]
    # 小数点2桁で四捨五入
    prediction = [round(float(n), 4) for n in prediction]
    print(prediction)
    return render_template(
        'index.html',
        image_source=base64.b64encode(file_body).decode("utf-8"), # 画像の保存があるとなにかと面倒くさいのでbase64で
        prediction=sorted(
            dict(zip(class_names_ja, prediction)).items(),
            key=lambda x: x[1],
            reverse=True
        ),
    )


if __name__ == "__main__":
    APP.run()
