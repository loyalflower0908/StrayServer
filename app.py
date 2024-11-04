import base64
import json
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
import firebase_admin
from firebase_admin import credentials, firestore, storage, firestore_async
from datetime import datetime, timedelta

# 라벨 이름과 인덱스를 매핑한 딕셔너리
label_names = {0: '아비시니안', 1: '뱅갈', 2: '버먼', 3: '봄베이',
               4: '브리티시 쇼트헤어', 5: '이집션 마우', 6: '메인쿤',
               7: '페르시안', 8: '랙돌', 9: '러시안 블루', 10: '샴',
               11: '스핑크스', 12: '아메리칸 불독', 13: '아메리칸 핏불 테리어',
               14: '바셋하운드', 15: '비글', 16: '복서', 17: '치와와',
               18: '잉글리쉬 코카 스파니엘', 19: '잉글리쉬 세터', 20: '저먼 쇼트헤어드 포인터',
               21: '그레이트 피레니즈', 22: '하바니즈', 23: '제페니즈 친', 24: '케이스혼드',
               25: '레온베르거', 26: '미니어처 핀셔(미니핀)', 27: '뉴펀들랜드', 28: '포메라니안',
               29: '퍼그', 30: '세인트 버나드', 31: '사모예드', 32: '스코티쉬 테리어',
               33: '시바견', 34: '스태퍼드셔 불 테리어', 35: '아이리쉬 소프트코티드 휘튼 테리어', 36: '요크셔 테리어'}

#BREED ID: 1-25:Cat 1:12:Dog
def dog_cat_label(key):
    return "고양이" if(key in range(0,12)) else "강아지"

cred = credentials.Certificate("fbmyaa.json")
firebase_admin.initialize_app(cred, {'storageBucket': "myaa-76db3.appspot.com"})

# 파이어스토어 클라이언트 생성
db = firestore.client()

markerList = []

# 이미지 URL을 담을 리스트
image_urls = []

# images 폴더 내의 파일 목록 가져오기
blobs = storage.bucket().list_blobs(prefix='images/')

for blob in blobs:
    # 현재 시간을 기준으로 1시간 뒤의 만료 시간을 계산
    #expiration_time = datetime.utcnow() + timedelta(hours=1)
    expiration_time = datetime.max
    # 각 이미지의 URL을 서명된 URL로 생성하여 리스트에 추가
    image_url = blob.generate_signed_url(expiration=expiration_time)
    image_urls.append(image_url)

# "멍냥이" 컬렉션의 모든 문서를 가져오기
collection_ref = db.collection("멍냥이")
documents = collection_ref.stream()

count = 0
# 문서를 순회하면서 필드 값 출력
for doc in documents:
    doc_dict = doc.to_dict()
    markerList.append({'image_link':image_urls[count], 'species':doc_dict['종'], 'time':doc_dict['발견시간'], 'latitude':doc_dict['위도'], 'longitude':doc_dict['경도']})
    count += 1
count = 0

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html", markerList=markerList)
    #return 'This is Home!'


@app.route("/hi", methods=['POST'])
def hi():
   if request.method == 'POST':
      return "안녕하세요"

@app.route("/isimage/", methods=['POST'])
def isimage():
   if request.method == 'POST':
      json_data = request.get_json()
      dict_data = json.loads(json_data)
      img_data = dict_data.get('img')
      if img_data:
         img = base64.b64decode(img_data)
         img = BytesIO(img)
         img = Image.open(img)
         img_np = np.array(img)
         if img != None:
            # 이미지 경로 및 모델 경로 설정
            #img_path = 'img/Sphynx03.jpeg'
            tflite_model_path = 'model/PetBreedAI_Improved.tflite'

            # TensorFlow Lite 모델 로드
            interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
            interpreter.allocate_tensors()

            # 입력 및 출력 텐서 정보 가져오기
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # 이미지 전처리  image = Image.open(img_np).resize(input_details[0]['shape'][1:3])
            image = Image.fromarray(img_np).resize(input_details[0]['shape'][1:3])
            image = np.expand_dims(np.array(image) / 255.0, axis=0).astype(np.float32)

            # 추론 실행
            interpreter.set_tensor(input_details[0]['index'], image)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # 결과 해석 및 라벨 출력
            predicted_label_index = np.argmax(output_data)
            predicted_label_name = label_names[predicted_label_index]
            confidence = np.max(output_data)
            confidence_percentage = confidence * 100  # 백분율로 변환
            if(int(confidence_percentage) < 50):
                print("알 수 없는 종")
            result_list = predicted_label_name+"/"+dog_cat_label(predicted_label_index)
            return jsonify(result_list), 200
         else:
            return jsonify("no Img"), 200
      else:
         return jsonify("No image data provided"), 200



if __name__ == '__main__':
   app.run('0.0.0.0',port=5000,debug=True)