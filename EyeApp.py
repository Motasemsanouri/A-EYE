import os
import random
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
from datetime import datetime, timedelta
import tensorflow as tf

app = Flask(__name__)

# إعدادات رفع الملفات
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# التصنيفات والنصائح
FUNDUS_CLASSES = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal', 'Other']
OCT_CLASSES = ['CNV', 'DME', 'Drusen', 'Normal', 'Other']

FUNDUS_ADVICE = {
    'Cataract': [
        "Use prescribed anti-inflammatory eye drops to manage symptoms",
        "Wear UV-protective sunglasses when outdoors",
        "Increase consumption of leafy greens and vitamin-rich foods",
        "Schedule follow-up every 6 months to monitor progression"
    ],
    'Diabetic Retinopathy': [
        "Maintain HbA1c levels below 7% through diet and medication",
        "Complete annual comprehensive dilated eye exams",
        "Monitor and control blood pressure below 130/80 mmHg",
        "Immediately report any vision changes to your ophthalmologist"
    ],
    'Glaucoma': [
        "Administer prescribed eye drops at the same time daily",
        "Avoid heavy lifting and inverted yoga positions",
        "Use proper lighting to reduce eye strain",
        "Practice regular aerobic exercise to help lower IOP"
    ],
    'Normal': [
        "Continue annual comprehensive eye exams",
        "Wear sunglasses with UV protection",
        "Maintain a diet rich in omega-3s and antioxidants",
        "Practice the 20-20-20 rule to reduce digital eye strain"
    ],
    'Other': [
        "Retake image with proper pupil dilation",
        "Ensure camera is properly focused on retina",
        "Avoid blinking during image capture",
        "Clean lens surface before imaging"
    ]
}

OCT_ADVICE = {
    'CNV': [
        "Strictly adhere to your anti-VEGF injection schedule",
        "Use an Amsler grid daily to monitor central vision",
        "Increase intake of omega-3 fatty acids and lutein",
        "Protect eyes from bright light with amber-tinted glasses"
    ],
    'DME': [
        "Maintain blood glucose levels between 80-130 mg/dL",
        "Elevate your head while sleeping to reduce fluid accumulation",
        "Reduce dietary sodium to minimize fluid retention",
        "Attend all scheduled laser treatment sessions"
    ],
    'Drusen': [
        "Take AREDS2 formula supplements as recommended",
        "Use blue light filters on digital devices",
        "Consume dark leafy greens and colorful vegetables daily",
        "Quit smoking and avoid secondhand smoke exposure"
    ],
    'Normal': [
        "Continue regular retinal screenings as recommended",
        "Protect eyes from excessive blue light exposure",
        "Maintain healthy blood pressure and cholesterol levels",
        "Stay physically active to promote ocular health"
    ],
    'Other': [
        "Remain still and focus on the fixation target during scan",
        "Request technician assistance for proper head positioning",
        "Avoid caffeine before OCT imaging to reduce eye movements",
        "Schedule scan for when you're well-rested"
    ]
}

def get_surgical_recommendation(condition):
    surgical_conditions = {
        'Cataract': 14,
        'Diabetic Retinopathy': 3,
        'Glaucoma': 7,
        'CNV': 2,
        'DME': 5
    }
    if condition in surgical_conditions:
        date = datetime.now() + timedelta(days=surgical_conditions[condition])
        return {
            'needs_surgery': True,
            'date': date.strftime("%B %d, %Y"),
            'time': "10:00 AM",
            'pre_op_instructions': "Avoid eating 8 hours before procedure and arrange transportation"
        }
    return {'needs_surgery': False}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

fundus_model = None
oct_model = None

def load_fundus_model():
    global fundus_model
    if fundus_model is None:
        fundus_model = tf.lite.Interpreter(model_path='eye_diseases_model.tflite')
        fundus_model.allocate_tensors()
    return fundus_model

def load_oct_model():
    global oct_model
    if oct_model is None:
        oct_model = tf.lite.Interpreter(model_path='oct_modelT.tflite')
        oct_model.allocate_tensors()
    return oct_model

def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

def predict_fundus(image_path):
    model = load_fundus_model()
    input_data = preprocess_image(image_path)
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.set_tensor(input_details[0]['index'], input_data)
    model.invoke()
    prediction = model.get_tensor(output_details[0]['index'])
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))
    predicted_class = 'Other' if confidence < 0.6 else FUNDUS_CLASSES[predicted_class_idx]
    advice = random.choice(FUNDUS_ADVICE[predicted_class])
    surgical_info = get_surgical_recommendation(predicted_class)
    return {
        'class': predicted_class,
        'confidence': confidence,
        'advice': advice,
        'surgical_recommendation': surgical_info,
        'all_predictions': {name: float(pred) for name, pred in zip(FUNDUS_CLASSES, prediction[0])}
    }

def predict_oct(image_path):
    model = load_oct_model()
    input_data = preprocess_image(image_path)
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.set_tensor(input_details[0]['index'], input_data)
    model.invoke()
    prediction = model.get_tensor(output_details[0]['index'])
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))
    predicted_class = 'Other' if confidence < 0.6 else OCT_CLASSES[predicted_class_idx]
    advice = random.choice(OCT_ADVICE[predicted_class])
    surgical_info = get_surgical_recommendation(predicted_class)
    return {
        'class': predicted_class,
        'confidence': confidence,
        'advice': advice,
        'surgical_recommendation': surgical_info,
        'all_predictions': {name: float(pred) for name, pred in zip(OCT_CLASSES, prediction[0])}
    }

#  الصفحة الرئيسية
@app.route('/')
def index():
    return render_template('Home.html')

# صفحة unknown classifier

@app.route('/predict_fundus', methods=['POST'])
def predict_fundus_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        try:
            result = predict_fundus(filename)
            os.remove(filename)
            return jsonify(result)
        except Exception as e:
            if os.path.exists(filename):
                os.remove(filename)
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/predict_oct', methods=['POST'])
def predict_oct_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        try:
            result = predict_oct(filename)
            os.remove(filename)
            return jsonify(result)
        except Exception as e:
            if os.path.exists(filename):
                os.remove(filename)
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/unknown')
def unknown_page():
    return render_template('unknown_classifier.html')

@app.route('/predict_unknown', methods=['POST'])
def predict_unknown_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            img = Image.open(filepath).convert('RGB')
            img_np = np.array(img)

            # هل الصورة رمادية (تقريبًا R = G = B)
            is_grayscale = (
                np.allclose(img_np[..., 0], img_np[..., 1], atol=2) and
                np.allclose(img_np[..., 1], img_np[..., 2], atol=2)
            )

            # احسب التباين
            contrast = img_np.std()

            # تحديد نوع الصورة وتنبؤها
            if is_grayscale and contrast < 40:
                result = predict_oct(filepath)
                result['image_type'] = 'OCT'
            elif not is_grayscale and contrast >= 40:
                result = predict_fundus(filepath)
                result['image_type'] = 'Fundus'
            else:
                result = {'prediction': 'Unknown or Unclassified'}
                result['image_type'] = 'Other'

            # إذا كانت النتيجة Other، غيّر نوع الصورة إلى Other
            if result.get('prediction', '').lower() == 'other':
                result['image_type'] = 'Other'

            os.remove(filepath)
            return jsonify(result)

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)