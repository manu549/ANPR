from flask import Flask, render_template, request
import cv2
import pytesseract
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = r'./tesseract.exe'
cascade = cv2.CascadeClassifier("./haarcascade_russian_plate_number.xml")

g = None

@app.route("/")
def hello_world():
    return render_template('new.html')


@app.route("/uploader", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        print(f.filename)
        f.save(secure_filename(f.filename))
        path = "./static/cars/"+f.filename
        image = cv2.imread(path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        coordinates = cascade.detectMultiScale(gray_image, 1.1, 4)

        for (x, y, w, h) in coordinates:
            a, b = (int(0.02 * image.shape[0]), int(0.025 * image.shape[1]))
            plate = image[y + a:y + h - a, x + b:x + w - b, :]
            plate_path = './static/assets/cars/crop_' + f.filename
            cv2.imwrite(plate_path, plate)
            # image processing
            kernel = np.ones((1, 1), np.uint8)
            plate = cv2.dilate(plate, kernel, iterations=1)
            plate = cv2.erode(plate, kernel, iterations=1)
            plate = cv2.bilateralFilter(plate, 9, 50, 50)
            gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            retval, plate = cv2.threshold(gray_plate, 127, 255, cv2.THRESH_BINARY)
            cv2.rectangle(image, (x, y), (x + w, y + h), (51, 51, 255), 2)
            name = './static/assets/cars/d_'+f.filename
            cv2.imwrite(name,image)
            number = pytesseract.image_to_string(plate)
            number = ''.join(n for n in number if n.isalnum())
            if len(number) < 1:
                plate = plate_path
            print(number)
            path = './static/assets/cars/'+f.filename

            return render_template('new1.html', number=number, file=path, detect=name, plate=plate)

if __name__ == '__main__':
    app.run(debug=True)


