# استيراد المكتبات
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

file_path = 'newlearningmasheen4.xlsx'
# اقرأ ملف الإكسل
data = pd.read_excel(file_path)
# اختيار الميزات والهدف
features = data[['Core','GPUType','RAMSize', 
                 'HardType', 'HardSize', 'ScreenSize',
                 'Quality', 'condition']]
target = data['price']

# تقسيم البيانات إلى مجموعات تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=905)

# بناء النموذج وتدريبه
model = LinearRegression()
model.fit(X_train, y_train)

# توقع أسعار اللابتوب باستخدام النموذج ومجموعة الاختبار
predicted_prices = model.predict(X_test)

# حساب متوسط مربعات الخطأ (MSE)
mse = mean_squared_error(y_test, predicted_prices)
print("Mean Squared Error (MSE):", mse)

def predict_laptop_price(core, gpu_type, ram_size, hard_type, hard_size, screen_size, quality, condition):
    # تحضير البيانات كمدخلات للنموذج
    laptop_features = [[core, gpu_type, ram_size, hard_type, hard_size, screen_size, quality, condition]]

    # توقع سعر اللابتوب باستخدام النموذج
    predicted_price = model.predict(laptop_features)

    return predicted_price[0]

# المواصفات المذكورة
laptop_core = 5  # Core i5
laptop_gpu_type = 3  # NIVIDIA Ge force GT 750 with 4 GB
laptop_ram_size = 8  # 8 GB DDR3
laptop_hard_type = 0  # HARD SSD
laptop_hard_size = 500  # 500GB
laptop_screen_size = 15.6  # لم يتم تقديم حجم الشاشة في المواصفات
laptop_quality = 2.7  # Good
laptop_condition =2  # used

predicted_price = predict_laptop_price(laptop_core, laptop_gpu_type, laptop_ram_size, laptop_hard_type, laptop_hard_size, laptop_screen_size, laptop_quality, laptop_condition)

print("Predicted Laptop Price:", predicted_price)

