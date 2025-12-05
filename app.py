import matplotlib.pyplot as plt
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
import imutils
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn import metrics
import ftplib
import math
import sqlite3
import requests
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import smtplib
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)

app.secret_key = 'brain_tumor'  

global filename
global accuracy
X = []
Y = []
global classifier
disease = ['No Tumor Detected','Tumor Detected']
filename = "./dataset"

load_dotenv()
# Gmail credentials from .env file
GMAIL_USERNAME = os.getenv("GMAIL_USERNAME")
GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")


@app.before_first_request
def create_users_table():
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')

        conn.commit()
        print("Table created successfully.")
    except sqlite3.Error as e:
        print(f"Error creating table: {e}")
    finally:
        conn.close()

@app.route('/')
def index():
    if 'user_email' not in session:
        return redirect('/login')
    return render_template('index.html', user_name=session.get('user_name'))
    
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')

    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')

    if not name or not email or not password:
        return "Name, Email, and Password are required!", 400

    hashed_password = generate_password_hash(password)

    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()

        if user:
            return "This email is already registered. Please login instead."

        cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, hashed_password))
        conn.commit()
        return redirect('/login')

    except sqlite3.Error as e:
        return f"An error occurred: {e}"
    finally:
        conn.close()

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        if 'user_email' in session:
            return redirect('/')
        return render_template('login.html')

    email = request.form.get('email')
    password = request.form.get('password')

    if not email or not password:
        return "Email and Password are required!", 400

    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()

        if user and check_password_hash(user[3], password):
            session['user_name'] = user[1]
            session['user_email'] = user[2]
            return redirect('/')
        else:
            return "Invalid email or password. Please try again."

    except sqlite3.Error as e:
        return f"An error occurred: {e}"
    finally:
        conn.close()

@app.route('/predict', methods=['POST'])
def predict():
    global disease
    global classifier

    if 'classifier' not in globals():
        classifier = None

    if classifier is None:
        generateModel()
        CNN()

    # Get the image file
    file = request.files['image']
    
    # Get user email from session
    user_email = session.get('user_email', None)

    # Get location data from the request
    latitude = float(request.form.get('latitude', 0.0))
    longitude = float(request.form.get('longitude', 0.0))
    location = {"latitude": latitude, "longitude": longitude}

    # Process the image
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(image, (128, 128)).reshape(1, 128, 128, 1)
    prediction = classifier.predict(np.array(img_resized))
    cls = np.argmax(prediction)

    if cls == 1:  # Tumor Detected
        _, binary_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        scale_factor = 0.01  # Adjust based on actual dataset scale
        tumor_size_cm = calculate_tumor_size_in_cm(binary_img, scale_factor)
        t_length = math.sqrt(math.sqrt(tumor_size_cm))
        s_m = suggest_medicine(t_length)
        if s_m == "Suggested Medicine: Go to hospital":
            nearest_hospital = find_nearest_hospital_with_osm(location)
        else:
            nearest_hospital = s_m
        result = {
            "prediction": disease[cls],
            "tumor_size": f"{t_length:.2f} cm",
            "medicine": nearest_hospital
        }
    else:
        result = {
            "prediction": disease[cls],
            "tumor_size": "    --",
            "medicine": "    --"
        }
        
    if user_email:
        send_email(user_email, result)

    return jsonify(result)


def generateModel():
    global X
    global Y
    X.clear()
    Y.clear()
    if os.path.exists('Model/myimg_data.txt.npy'):
        X = np.load('Model/myimg_data.txt.npy')
        Y = np.load('Model/myimg_label.txt.npy')
    else:
        for root, dirs, directory in os.walk(filename+"/no"):
            for i in range(len(directory)):
                name = directory[i]
                img = cv2.imread(filename+"/no/"+name,0)
                ret2,th2 = cv2.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
                img = cv2.resize(img, (128,128))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(128,128,1)
                X.append(im2arr)
                Y.append(0)

        for root, dirs, directory in os.walk(filename+"/yes"):
            for i in range(len(directory)):
                name = directory[i]
                img = cv2.imread(filename+"/yes/"+name,0)
                ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
                img = cv2.resize(img, (128,128))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(128,128,1)
                X.append(im2arr)
                Y.append(1)
                
        X = np.asarray(X)
        Y = np.asarray(Y)            
        np.save("Model/myimg_data.txt",X)
        np.save("Model/myimg_label.txt",Y)

def CNN():
    global accuracy
    global classifier
    
    YY = to_categorical(Y)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    x_train = X[indices]
    y_train = YY[indices]

    if os.path.exists('Model/model.json'):
        with open('Model/model.json', "r") as json_file:
           loaded_model_json = json_file.read()
           classifier = model_from_json(loaded_model_json)

        classifier.load_weights("Model/model_weights.h5")  
        f = open('Model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        print("Accuracy:", accuracy)    
    else:
        X_trains, X_tests, y_trains, y_tests = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)
        classifier = Sequential() 
        classifier.add(Convolution2D(32, 3, 3, input_shape = (128, 128, 1), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 128, activation = 'relu'))
        classifier.add(Dense(output_dim = 2, activation = 'softmax'))
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = classifier.fit(x_train, y_train, batch_size=16, epochs=10,validation_split=0.2, shuffle=True, verbose=2)
        classifier.save_weights('Model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("Model/model.json", "w") as json_file:
            json_file.write(model_json)
        f = open('Model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('Model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[9] * 100
        print("Accuracy:", accuracy)

@app.route('/update-location', methods=['POST'])
def update_location():
    data = request.get_json()
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    return {"latitude": latitude, "longitude": longitude}
    
def calculate_tumor_size_in_cm(binary_img, scale_factor):
    """
    Calculate tumor size in cm² based on binary image and scale factor.
    """
    # Find contours of the tumor region
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area_pixels = 0

    for contour in contours:
        total_area_pixels += cv2.contourArea(contour)

    # Convert pixel area to cm²
    tumor_size_cm = total_area_pixels * scale_factor
    return tumor_size_cm    

def suggest_medicine(tumor_size):
    """
    Suggest medicines based on tumor size.
    """
    if tumor_size <= 3:
        return "Suggested Medicine: Aspirin (for pain relief)"
    elif 3 < tumor_size <= 4.5:
        return "Suggested Medicine: Dexamethasone (to reduce swelling)"
    elif 4.5 < tumor_size <= 5.5:
        return "Suggested Medicine: Temozolomide (chemotherapy drug)"
    else:
        return "Suggested Medicine: Go to hospital"

def find_nearest_hospital_with_osm(location):
    """
    Find the nearest hospital using OpenStreetMap's Overpass API with improved accuracy.
    """
    try:
        overpass_url = "http://overpass-api.de/api/interpreter"
        query = f"""
        [out:json];
        node
          ["amenity"="hospital"]
          (around:3000,{location['latitude']},{location['longitude']});
        out body;
        """
        
        response = requests.get(overpass_url, params={"data": query})
        if response.status_code == 200:
            data = response.json()
            if "elements" in data and len(data["elements"]) > 0:
                hospitals = []
                for hospital in data["elements"]:
                    lat, lon = hospital["lat"], hospital["lon"]
                    distance = geodesic((location['latitude'], location['longitude']), (lat, lon)).meters
                    hospitals.append((distance, hospital))

                # Sort hospitals by distance (smallest first)
                hospitals.sort(key=lambda x: x[0])

                nearest_hospital = hospitals[0][1]
                name = nearest_hospital.get("tags", {}).get("name", "Brain Hospital")
                lat, lon = nearest_hospital["lat"], nearest_hospital["lon"]
                return f"{name} (Latitude: {lat}, Longitude: {lon})"
            else:
                return "No nearby hospitals found."
        else:
            return "Error fetching hospital data from OSM."
    except Exception as e:
        print(f"Error: {e}")
        return "Unable to fetch hospital data."

def send_email(to_gmail, result):
    subject = "Brain Tumor Prediction Result"
    
    msg_content = f"""Dear {session['user_name']},

Here is your brain tumor prediction result:

- Prediction: {result['prediction']}
- Tumor Size: {result['tumor_size']}
- Suggested Medicine / Nearest Hospital: {result['medicine']}

Please consult a doctor if necessary.

Best regards,  
Brain Tumor Detection Project Team  
- Sk. Ajith  
- G. HariChandan  
- M. Bhargav  
- L. Jaya Sankar
"""

    try:
        # Establish connection inside the function
        connection = smtplib.SMTP('smtp.gmail.com', 587)
        connection.starttls()
        connection.login(GMAIL_USERNAME, GMAIL_PASSWORD)

        # Create email message
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USERNAME
        msg['To'] = to_gmail
        msg['Subject'] = subject
        msg.attach(MIMEText(msg_content, 'plain'))

        # Send email
        connection.sendmail(GMAIL_USERNAME, to_gmail, msg.as_string())

        # Close the connection
        connection.quit()
        print("Email sent successfully!")

    except Exception as e:
        print("Error Sending Email:", e)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')


if __name__ == '__main__':
    app.run(debug=True)
