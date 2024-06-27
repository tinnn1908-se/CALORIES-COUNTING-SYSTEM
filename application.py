import json
from flask import Flask, jsonify, redirect, request, url_for
from flask_cors import CORS
from PIL import Image
import subprocess
import os

app = Flask(__name__)
CORS(app)


def read_file():
  f = open("pixel_info_list.txt", "r")
  total = 0
  foods = []
  for s in f:
    print(s)
    s = s.strip()
    name = s.split(':')[0]
    calories = s.split(':')[1]
    total += int(calories)
    foods.append({'name':name,'calories':calories})
  return [foods,total]
  

@app.route('/result', methods=['GET'])
def get_foods():
  #Run model
  subprocess.run(['python', 'model.py'])
  result = [foods,total] = read_file()
  return jsonify(result),200

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
      # Save Image  
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        image = request.files['image']
        image.filename = "picture.jpg"
        image.save(os.path.join('C:/Tin N Nguyen/msu/FALL2023/CSC450/Project/WebProject/be/uploads/',image.filename))
        return jsonify({'message': 'Image uploaded successfully'}), 200
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
   app.run(port=5000)