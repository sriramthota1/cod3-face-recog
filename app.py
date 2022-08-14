from flask import Flask,request, url_for, redirect, render_template
from datetime import datetime
import numpy as np
from werkzeug.utils import secure_filename
import face_recognition as fr
import cv2
import numpy as np
import os,shutil
import base64


app = Flask(__name__)

#Here i have read the data from gogle drive
def get_encoded_faces():
  encoded={}
  for dirpath,dname,fname in os.walk("static/images/"):
    for f in fname:
      if f.endswith(".jpeg") or f.endswith(".jpg"):
        face=fr.load_image_file("static/images/"+ f)
        encoding=fr.face_encodings(face)[0]
        encoded[f.split(".")[0]]=encoding
  return (encoded)

# Here i am recognising the face of test data by providing certain data
def classify_face(im):
  faces = get_encoded_faces()
  faces_encoded = list(faces.values())
  known_faces_names = list(faces.keys())

  img = cv2.imread(im)
  face_locations = fr.face_locations(img)
  unknown_face_encodings = fr.face_encodings(img, face_locations)
  face_names = []
  for face_encoding in unknown_face_encodings:
    name = "Unknown"
    matches = fr.compare_faces(faces_encoded, face_encoding)
    face_distances = fr.face_distance(faces_encoded, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
      name = known_faces_names[best_match_index]
    face_names.append(name)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
      cv2.rectangle(img, (left - 20, top - 20), (right + 20, bottom + 20), (255, 0, 0), 2)
      cv2.rectangle(img, (left - 20, bottom - 15), (right + 20, bottom + 20), (255, 0, 0), cv2.FILLED)
      cv2.putText(img, name, (left - 20, bottom + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
  while (True):
    #cv2_imshow(img)
    return (face_names)





@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method=="GET":
        return "no picture"
    elif request.method == "POST":
        features = [(x) for x in request.form.values()]
        print(features)
        for i in features:
            print(i)
            print(i[23:])

        imgstring=i[23:]
        imgdata = base64.b64decode(imgstring)
        from datetime import datetime
        import os.path
        directory = './studentimgs/'
        now = datetime.now()
        f_name = now.strftime("%d%m%Y %H%M%S")
        filename = f_name + '.png'
        filepath = os.path.join(directory, filename)
        # print(f_name)

        with open(filepath, 'wb') as f:
            f.write(imgdata)
        f.close()
        print("Done")

        print(classify_face(filepath))
        output=classify_face(filepath)
        return render_template('index.html',
                           pred='Hii{} Attendence recorded'.format(output))
if __name__ == '__main__':
    app.run(debug=True) #runs the flask server #debug=checks errors and shows in terminal
