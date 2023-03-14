from flask import *
import os
import shutil
import json
import cv2
from collections import defaultdict

from model_util import ImageClassifier
from bodysplitter import crop_image

app = Flask(__name__)
BASE_DIR = os.path.dirname('')
classifier = ImageClassifier()
BODY_CROPS_NAME = ['head', 'hips', 'body', 'body_and_hips', 'right_foot', 'left_foot']

if os.path.exists('./archive.zip'):
    classifier.prepare_archive('./archive.zip', BASE_DIR)


def creat_crop_images(path):
    image = cv2.imread(path)
    crops = crop_image(image, take_crops=True)
    count: int = 0
    for crop in crops:
        cv2.imwrite(f"{count}.jpg", crop)
        count += 1

@app.route('/upload', methods=['GET', 'POST'])
def upload_archive():
    if request.method == 'POST':
        file_a = request.files.get("archive")
        _, ext = os.path.splitext(file_a.filename)

        file_i = request.files.get("file")
        _, ext = os.path.splitext(file_i.filename)

        if file_a:
            if os.path.exists(os.path.join(BASE_DIR, "archive")):
                shutil.rmtree(os.path.join(BASE_DIR, "archive")) # remove previous images
            path_zip = os.path.join(BASE_DIR, "archive.zip")
            file_a.save(path_zip)
            try:
                classifier.remove_all()
                classifier.prepare_archive(path_zip, BASE_DIR)
            except:
                return json.dumps({"success": False, "reason": "Archive file is not correct"})

        if not file_i:
            return json.dumps({"success": False, "reason": "File is not found"})

        if classifier.all_skus == {}:
            return json.dumps({"success": False, "reason": "There is no uploaded archive"})

        path_img = os.path.join(BASE_DIR, "image.jpg")
        file_i.save(path_img)
        result_dict = defaultdict(dict)

        try:
            creat_crop_images(path_img)
            for i, part in enumerate(BODY_CROPS_NAME):
                if part != 'head':
                    classes, dists = classifier.predict(f'./{i}.jpg')
                    result_dict[part]['classes'] = classes
                    result_dict[part]['dists'] = dists
        except:
            return json.dumps({"success": False, "reason": "Image is not correct"})

        return json.dumps({"success": True, **result_dict}, ensure_ascii=False).encode('utf8')
    return '''
    <!doctype html>
    <h1>Upload new archive</h1>
    <form action="" method=post enctype=multipart/form-data>
        <p>Upload archive <input type=file name=archive></p>
        <p>Upload image <input type=file name=file></p>
        <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=8000)
