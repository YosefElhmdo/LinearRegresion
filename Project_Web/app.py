from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LinearRegression
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# تمكين CORS للتفواج بين المصادر المختلفة
origins = [
    "http://127.0.0.1:5500",
    "https://preview.flutlab.io",  # يمكنك تحديد المزيد من المصادر هنا
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# بناء النموذج وتدريبه
file_path = 'newlearningmasheen4.xlsx'
data = pd.read_excel(file_path)
features = data[['Company','Core','Generation', 'GPUType', 'RAMSize', 'HardType', 'HardSize',
                 'Material','ColoreKeyboard','TypeScreen', 'ScreenSize', 'Quality', 'condition']]
target = data['price']

model = LinearRegression()
model.fit(features, target)

class LaptopSpecs(BaseModel):
    Company: str
    Core: str
    Generation: str
    GPUType: str
    RAMSize: str
    HardType: str
    HardSize: str
    Material: str
    ColoreKeyboard: str
    TypeScreen: str
    ScreenSize: str
    Quality: str
    condition: str

@app.post("/predict")
def predict_price(specs: LaptopSpecs):
    try:
        # تحويل البيانات المدخلة إلى DataFrame
        user_input_df = pd.DataFrame(specs.dict(), index=[0])

        # توقع سعر اللابتوب باستخدام النموذج
        predicted_price = model.predict(user_input_df)

        # تجهيز الرد كمخرج JSON
        response = {
            'predicted_price': predicted_price[0]
        }
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
