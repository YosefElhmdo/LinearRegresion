from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5500"}})

# تمكين CORS للتفواج بين المصادر المختلفة
CORS(app)

# بناء النموذج وتدريبه
file_path = 'newlearningmasheen4.xlsx'
data = pd.read_excel(file_path)
features = data[['Company','Core','Generation', 'GPUType', 'RAMSize', 'HardType', 'HardSize',
                 'Material','ColoreKeyboard','TypeScreen', 'ScreenSize', 'Quality', 'condition']]
target = data['price']

model = LinearRegression()
model.fit(features, target)

@app.route('/predict', methods=['POST'])
def predict_price():
    try:
        # قراءة بيانات المواصفات من الطلب الوارد
        user_input = request.json
        laptop_core = user_input.get('core')
        laptop_company = user_input.get('company')
        laptop_generation = user_input.get('generation')
        laptop_gpu_type = user_input.get('gpu_type')
        laptop_ram_size = user_input.get('ram_size')
        laptop_hard_type = user_input.get('hard_type')
        laptop_hard_size = user_input.get('hard_size')
        laptop_material = user_input.get('material')
        laptop_colorekeyboard = user_input.get('colorekeyboard')
        laptop_typescreen = user_input.get('typescreen')
        laptop_screen_size = user_input.get('screen_size')
        laptop_quality = user_input.get('quality')
        laptop_condition = user_input.get('condition')

        # تحويل المواصفات إلى DataFrame للتنبؤ بالسعر
        user_input_df = pd.DataFrame({
            'Company': [laptop_company],
            'Core': [laptop_core],
            'Generation': [laptop_generation],
            'GPUType': [laptop_gpu_type],
            'RAMSize': [laptop_ram_size],
            'HardType': [laptop_hard_type],
            'HardSize': [laptop_hard_size],
            'Material': [laptop_material],
            'ColoreKeyboard': [laptop_colorekeyboard],
            'TypeScreen': [laptop_typescreen],
            'ScreenSize': [laptop_screen_size],
            'Quality': [laptop_quality],
            'condition': [laptop_condition]
        })

        # توقع سعر اللابتوب باستخدام النموذج
        predicted_price = model.predict(user_input_df)

        # تجهيز الرد كمخرج JSON
        response = {
            'predicted_price': predicted_price[0]
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # تمكين CORS للوصول من مصدر معين (مثلاً, https://preview.flutlab.io)
    CORS(app, origins=["https://preview.flutlab.io"])
    CORS(app, origins=["http://localhost:5000"])
    # بدء تشغيل الخادم على المنفذ 5000
    app.run(host='0.0.0.0', port=5000)

