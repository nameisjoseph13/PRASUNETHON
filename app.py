from flask import Flask, render_template, request
import pickle
import numpy as np
import requests
import ee
import folium
from datetime import datetime, timedelta

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
fer_model = pickle.load(open('fertilizer.pkl','rb'))

ee.Authenticate()
ee.Initialize(project='satellite-427815')

soiltype_mapping = {
    'Sandy' : 0,
    'Loamy' : 1,
    'Black' : 2,
    'Red' : 3,
    'Clayey' : 4
}


@app.route('/')
def home():
    latitude, longitude = get_location()
    return render_template('index.html', latitude=latitude, longitude=longitude)

@app.route('/predict', methods=['POST'])
def predict():
    N = int(request.form['N'])
    P = int(request.form['P'])
    K = int(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    moisture = float(request.form['moisture'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    soiltype = request.form['soiltype']
    soiltype_value = soiltype_mapping[soiltype]
    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])
    print(soiltype, soiltype_value)

    crop_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    crop_prediction = model.predict(crop_input)
    crop_val = crop_prediction[0]

    fer_input = np.array([[temperature, humidity, moisture, N, K, P, soiltype_value ]])
    fer_prediction = fer_model.predict(crop_input)
    fer_val = fer_prediction[0]
    print(fer_val)

    point = ee.Geometry.Point([longitude, latitude])

    today = datetime.today().date()
    three_months_ago = today - timedelta(days=3 * 30)
    start_date = three_months_ago.strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')

    sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR') \
    .filterDate(start_date, end_date) \
    .filterBounds(point) \
    .sort('CLOUDY_PIXEL_PERCENTAGE') \
    .first()

    ndvi = sentinel2.normalizedDifference(['B8', 'B4']).rename('NDVI')

    evi = sentinel2.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
        'NIR': sentinel2.select('B8'),
        'RED': sentinel2.select('B4'),
        'BLUE': sentinel2.select('B2')
    }).rename('EVI')

    ndvi_value = ndvi.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=30).getInfo()['NDVI']
    evi_value = evi.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=30).getInfo()['EVI']

    print(f'NDVI: {ndvi_value}, EVI: {evi_value}')

    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        temperature_max = data['daily']['temperature_2m_max'][0]  
        temperature_min = data['daily']['temperature_2m_min'][0]  
        rainfall = data['daily']['precipitation_sum'][0]  
        print(f"Max Temperature: {temperature_max} °C")
        print(f"Min Temperature: {temperature_min} °C")
        print(f"Daily Rainfall: {rainfall} mm")
    else:
        print("Error:", response.status_code, response.text)

    level = ''
    des = ''
    if(ndvi_value <= 0.05) :
        level = 'Bad'
        des = 'This area is typically associated with non-vegetated surfaces like water bodies, snow-covered areas, or barren land'
        crop_val = "No crop can grow"
    elif(ndvi_value > 0.05 and ndvi_value <= 0.23) :
        level = 'Moderate'
        des = 'This  area has sparse vegetation cover, areas with stress (such as drought stress or nutrient deficiencies), or non-vegetated surfaces like bare soil or urban areas.'
    elif(ndvi_value > 0.23 and ndvi_value <= 0.67) :
        level = "Good"
        des = 'This area have good vegetation cover and health. This area is useful for good vegetation conditions in agricultural lands, grasslands, and mixed-use areas'
    else:
        level = "Very Good"
        des = 'This are have very good vegetation, which indicates healthy and dense vegetation cover. This area is beneficial for robust plant growth, such as forests, healthy crops, or areas recovering from disturbance'

    return render_template('result.html', predicted_class=crop_val, fer_class=fer_val, ndvi=ndvi_value, evi=evi_value,
                        level=level, desc=des, max_temp=temperature_max, min_temp=temperature_min, rain=rainfall   )

@app.route('/get_location', methods=['GET'])
def get_location():
    ip_response = requests.get('https://api64.ipify.org?format=json')
    ip_data = ip_response.json()
    ip_address = ip_data['ip']

    location_response = requests.get(f'https://ipinfo.io/{ip_address}/json')
    location_data = location_response.json()
    loc = location_data['loc'].split(',')
    latitude = float(loc[0])
    longitude = float(loc[1])
    
    return latitude, longitude


if __name__ == '__main__':
    app.run(debug = True)
    
