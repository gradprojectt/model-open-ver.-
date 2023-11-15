# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 20:55:36 2023

@author: user
"""
from flask_cors import CORS
from flask import Flask, jsonify, request
import main
import pymysql


app = Flask(__name__)
app.config['DEBUG'] = True
CORS(app)




# 예외 처리 함수
@app.errorhandler(500)
def internal_server_error(e):
    response = {
        "error": "Internal Server Error",
        "message": "An internal server error occurred while processing your request."
    }
    return jsonify(response), 500

@app.route('/classifier',methods=['POST'])
def react_to_flask():
    import cv2
    import dlib
    import numpy
    import matplotlib.pyplot as plt
    import math
    import io
    import cv2
    import numpy as np
    from skimage import io
    from matplotlib import pyplot as plt
    import scipy.ndimage
    import sys
    from skimage import io
    import joblib 
    from joblib import dump, load

    conn=pymysql.connect(database="what",
                     host="3.35.249.235",
                     port=int(3306),
                     user="rainbow",
                     password="1234")

    cur = conn.cursor() 
    
    def white_balance_loops(img):

        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        for x in range(result.shape[0]):
            for y in range(result.shape[1]):
                l, a, b = result[x, y, :]
                # fix for CV correction
                l *= 100 / 255.0
                result[x, y, 1] = a - ((avg_a - 128) * (l / 100.0) * 1.1)
                result[x, y, 2] = b - ((avg_b - 128) * (l / 100.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result

    def annotate_landmarks(im, landmarks):
        im = im.copy()
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(im, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))
            cv2.circle(im, pos, 3, color=(0, 255, 255))
        return im

    def read_im_and_landmarks(im):
        #im = cv2.imread(fname, cv2.IMREAD_COLOR)

        im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                             im.shape[0] * SCALE_FACTOR))
        s = get_landmarks(im)

        return im, s

    def get_landmarks(im):
        rects = detector(im, 1)

        if len(rects) > 1:
            raise TooManyFaces
        if len(rects) == 0:
            raise NoFaces

        return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

    def warm_or_cool(skin_ab_value):
        value = loaded_model.predict(skin_ab_value)

        if value == 0:
            return "warm"
        else:
            return "cool"


    def cut_eyes(im,landmarks):
        im = im[landmarks[37][0, 1]+1:landmarks[40][0, 1]+1, landmarks[37][0, 0]-2:landmarks[38][0, 0]+3, :]

        return im

    def select_index_to_use():
        max = 0
        name_of_index = ""
        index_to_use = []

        if len(index_dark) > len(index_medium):
            max = len(index_dark)
            index_to_use = index_dark
            name_of_index = "index_dark"
        else:
            max = len(index_medium)
            index_to_use = index_medium
            name_of_index = "index_medium"


        if max < len(index_light):
            print("index_light")
            index_to_use = index_light
        else:
            print(name_of_index)

        return index_to_use

    def pccs_finder(season, s, v):
        i = 0
        min = 2000
        skin_type = ""

        if season == "spring":
            while i < len(spring_list):
                distance = math.sqrt((s - spring_list[i][0])**2 + (v - spring_list[i][1])**2)
                print(distance)
                if min > distance:
                    min = distance
                    print("Calculating... " + spring_list[i][2] + " : " + str(distance))
                    skin_type = spring_list[i][2]

                i += 1

        elif season == "summer":
            while i < len(summer_list):
                distance = math.sqrt((s - summer_list[i][0])**2 + (v - summer_list[i][1])**2)
                print(distance)
                if min > distance:
                    min = distance
                    print("Calculating... " + summer_list[i][2] + " : " + str(distance))
                    skin_type = summer_list[i][2]

                i += 1

        elif season == "autumn":
            while i < len(autumn_list):
                distance = math.sqrt((s - autumn_list[i][0])**2 + (v - autumn_list[i][1])**2)
                print(distance)
                if min > distance:
                    min = distance
                    print("Calculating... " + autumn_list[i][2] + " : " + str(distance))
                    skin_type = autumn_list[i][2]

                i += 1

        else:
            while i < len(winter_list):
                distance = math.sqrt((s - winter_list[i][0])**2 + (v - winter_list[i][1])**2)
                if min > distance:
                    min = distance
                    print("Calculating... " + winter_list[i][2] + " : " + str(distance))
                    skin_type = winter_list[i][2]

                i += 1

        return skin_type
    
    request_file = request.files.get('file')
    userId = request.form.get('userId')
    #request_filename = request.form.get('fileName')
    request_file.save('image/' + request_file.filename)
    filename, img_type = request_file.filename.split('.') ## jpg or png etc...

    PREDICTOR_PATH = r"shape_predictor_68_face_landmarks.dat"

    SCALE_FACTOR = 1
    FEATHER_AMOUNT = 11
    FACE_POINTS = list(range(17, 68))
    MOUTH_POINTS = list(range(48, 61))
    RIGHT_BROW_POINTS = list(range(17, 22))
    LEFT_BROW_POINTS = list(range(22, 27))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))
    NOSE_POINTS = list(range(27, 35))
    JAW_POINTS = list(range(0, 17))

    # Points used to line up the images.
    ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    class TooManyFaces(Exception):
        pass

    class NoFaces(Exception):
        pass

    path='image/'+filename+'.jpg'
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    #img_white = white_balance_loops(img)
    im, landmarks = read_im_and_landmarks(img)
    im_annotated = annotate_landmarks(im, landmarks)

    cv2.imwrite(r'result/'+filename+'_landmarks.jpg', im_annotated) # save landmarked image file as ~_landmarks.jpg

    color_location1 = ((landmarks[54]+landmarks[11]+landmarks[45])/3).astype(int) # left cheek
    color_location2 = ((landmarks[48]+landmarks[4]+landmarks[36])/3).astype(int) # right cheek

    rgb1 = im[color_location1[0, 1], color_location1[0, 0]][2], im[color_location1[0, 1], color_location1[0, 0]][1], im[color_location1[0, 1], color_location1[0, 0]][0]
    rgb2 = im[color_location2[0, 1], color_location2[0, 0]][2], im[color_location2[0, 1], color_location2[0, 0]][1], im[color_location2[0, 1], color_location2[0, 0]][0]

    rgb = ((int(rgb1[0])+int(rgb2[0]))/2, (int(rgb1[1])+int(rgb2[1]))/2, (int(rgb1[2])+int(rgb2[2]))/2)

    cv2.putText(im, str("Target1"), (color_location1[0, 0]+5, color_location1[0, 1]),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=0.5,
                        color=(0, 0, 255))
    cv2.circle(im, (color_location1[0, 0], color_location1[0, 1]), 3, color=(0, 0, 255), thickness = -1)

    cv2.putText(im, str("Target2"), (color_location2[0, 0]+5, color_location2[0, 1]),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=0.5,
                        color=(0, 0, 255))
    cv2.circle(im, (color_location2[0, 0], color_location2[0, 1]), 3, color=(0, 0, 255), thickness = -1)

    cv2.putText(im, str("Target1"), (color_location1[0, 0]+5, color_location1[0, 1]),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=0.5,
                        color=(0, 0, 255))
    cv2.circle(im, (color_location1[0, 0], color_location1[0, 1]), 3, color=(0, 0, 255), thickness = -1)

    cv2.putText(im, str("Target2"), (color_location2[0, 0]+5, color_location2[0, 1]),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=0.5,
                        color=(0, 0, 255))
    cv2.circle(im, (color_location2[0, 0], color_location2[0, 1]), 3, color=(0, 0, 255), thickness = -1)

    im_annotated = annotate_landmarks(im, landmarks)

    cv2.imwrite(r'result/'+filename+'_target.jpg', im_annotated)

    #a* b* value comparison
    im = cv2.imread(path)
    lab_colors = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)

    a_value1 = lab_colors[color_location1[0, 1], color_location1[0, 0]][1]
    a_value2 = lab_colors[color_location2[0, 1], color_location2[0, 0]][1]

    a_value = (int(a_value1) + int(a_value2))/2


    b_value1 = lab_colors[color_location1[0, 1], color_location1[0, 0]][2]
    b_value2 = lab_colors[color_location2[0, 1], color_location2[0, 0]][2]

    b_value = (int(b_value1) + int(b_value2))/2

    loaded_model = load('classifier1.sav')

    skin_ab_value = []

    info = []
    info.append(a_value)
    info.append(b_value)

    skin_ab_value.append(info)

    #Extracting Eye Area

    cut_eyes = cut_eyes(im, landmarks)
    cv2.imwrite(r'result/'+filename+'_eye.jpg', cut_eyes)

    #Pupil & Light (on Eye) Detection

    import numpy as np
    import matplotlib.pyplot as plt
    gray = cv2.cvtColor(cut_eyes, cv2.COLOR_BGR2GRAY)

    etval, thresholded = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)

    index_dark = []

    i = 0

    while i < cut_eyes.shape[0]:
        j = 0
        index_of_black = []
        while j < cut_eyes.shape[1]:
            if thresholded[i][j] == 0:
                index_of_black.append(i)
                index_of_black.append(j)
                index_dark.append(index_of_black)
                index_of_black = []
            j += 1
        i += 1
        
    # Pupil Detection (thresh = 20)
    etval, thresholded = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)

    i = 0

    while i < cut_eyes.shape[0]:
        j = 0
        index_deleted = []
        while j < cut_eyes.shape[1]:
            if thresholded[i][j] == 0:
                index_deleted.append(i)
                index_deleted.append(j)
                if index_deleted in index_dark:
                    index_dark.remove(index_deleted)
                index_deleted = []
            j += 1
        i += 1
        
    etval, thresholded = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    index_medium = []

    i = 0

    while i < cut_eyes.shape[0]:
        j = 0
        index_of_black = []
        while j < cut_eyes.shape[1]:
            if thresholded[i][j] == 0:
                index_of_black.append(i)
                index_of_black.append(j)
                index_medium.append(index_of_black)
                index_of_black = []
            j += 1
        i += 1
    etval, thresholded = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    i = 0

    while i < cut_eyes.shape[0]:
        j = 0
        index_deleted = []
        while j < cut_eyes.shape[1]:
            if thresholded[i][j] == 0:
                index_deleted.append(i)
                index_deleted.append(j)
                if index_deleted in index_medium:
                    index_medium.remove(index_deleted)
                index_deleted = []
            j += 1
        i += 1

    etval, thresholded = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    index_light = []

    i = 0

    while i < cut_eyes.shape[0]:
        j = 0
        index_of_black = []
        while j < cut_eyes.shape[1]:
            if thresholded[i][j] == 0:
                index_of_black.append(i)
                index_of_black.append(j)
                index_light.append(index_of_black)
                index_of_black = []
            j += 1
        i += 1

    etval, thresholded = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY)
    i = 0

    while i < cut_eyes.shape[0]:
        j = 0
        index_deleted = []
        while j < cut_eyes.shape[1]:
            if thresholded[i][j] == 0:
                index_deleted.append(i)
                index_deleted.append(j)
                if index_deleted in index_light:
                    index_light.remove(index_deleted)
                index_deleted = []
            j += 1
        i += 1

    index_to_use = select_index_to_use()

    #L*value of Eye (15% is 0)
    eye_lab = cv2.cvtColor(cut_eyes, cv2.COLOR_RGB2LAB)
    eye_l_value = []

    i = 0
    while i < len(index_to_use):
        eye_l_value.append(eye_lab[index_to_use[i][0], index_to_use[i][1], 0])
        i += 1
        
    i = 0
    while i < int(len(eye_l_value)*15/85):
        eye_l_value.append(0)
        i += 1

    tone = warm_or_cool(skin_ab_value)


    eye_l_value = np.array(eye_l_value)
    std = math.sqrt(np.sum((eye_l_value - np.mean(eye_l_value))**2)/(eye_l_value.size))

    eye_brightness = ""

    if std < 38.28:
        eye_brightness = "dark"
    else:
        eye_brightness = "light"

    def season_matching(skin_tone, eye_brightness):
        if skin_tone == "warm":
            if eye_brightness == "dark":
                return "autumn"
            else:
                return "spring"
        else:
            if eye_brightness == "dark":
                return "winter"
            else:
                return "summer"

    season = season_matching(warm_or_cool(skin_ab_value), eye_brightness)

    color_info = [[0.87464539937997 * 255, 181, 'vivid'],
     [0.6920954876849125 * 255, 207, 'bright'],
     [0.8348038859274655 * 255, 170, 'strong'],
     [0.8901501134915047 * 255, 137, 'deep'],
     [0.3300064593086124 * 255, 221, 'light'],
     [0.39800595815638706 * 255, 182, 'soft'],
     [0.5135643825656158 * 255, 142, 'dull'],
     [0.6815737217178439 * 255, 93, 'dark'],
     [0.11073418459625402 * 255, 226, 'pale'],
     [0.22483614835441365 * 255, 110, 'grayish'],
     [0.26913149326755986 * 255, 58, 'dark_grayish'],
     [0.20022609113608203 * 255, 186, 'light_grayish']]

    spring_list = []
    spring_list.append(color_info[0])
    spring_list.append(color_info[1])
    spring_list.append(color_info[4])
    spring_list.append(color_info[8])

    summer_list = []
    summer_list.append(color_info[4])
    summer_list.append(color_info[8])
    summer_list.append(color_info[5])
    summer_list.append(color_info[6])
    summer_list.append(color_info[7])
    summer_list.append(color_info[9])
    summer_list.append(color_info[10])
    summer_list.append(color_info[11])

    autumn_list = []
    autumn_list.append(color_info[3])
    autumn_list.append(color_info[5])
    autumn_list.append(color_info[6])
    autumn_list.append(color_info[9])
    autumn_list.append(color_info[3])
    autumn_list.append(color_info[7])

    winter_list = []
    winter_list.append(color_info[0])
    winter_list.append(color_info[2])
    winter_list.append(color_info[3])
    winter_list.append(color_info[7])
    winter_list.append(color_info[10])

    skin_v_value = max(rgb[0], rgb[1], rgb[2])
    skin_s_value = (1-min(rgb[0], rgb[1], rgb[2])/skin_v_value) * 255

    pccs = pccs_finder(color_info, skin_s_value, skin_v_value)
    
    content = [tone, season, pccs]
    
    response_data = {
        "tone": content[0],
        "season": content[1],
        "pccs": content[2]
    }
    
    
    cur.execute("UPDATE user SET pccs = %s , season = %s, tone = %s WHERE user_id = %s;",(content[2], content[1], content[0], str(userId)))
    conn.commit()
    cur.close()
    
    return jsonify(response_data)
    

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000",  debug=True)



