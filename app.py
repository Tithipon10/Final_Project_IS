import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_resource
def load_models():
    with open('linear_model.pkl', 'rb') as f:
        linear_model = pickle.load(f)
    neural_network_model = keras.models.load_model('neural_network_model.h5')
    return linear_model, neural_network_model 

linear_model, neural_network_model = load_models()

def preprocess_linear(df):
    try:
        df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
        df['Release Date'] = df['Release Date'].fillna(df['Release Date'].mean())
        df['Release Date'] = (df['Release Date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
        label_encoder = LabelEncoder()
        for col in df.select_dtypes(include='object').columns:
            df[col] = label_encoder.fit_transform(df[col])
        features = ['Artist', 'Album Name', 'Release Date', 'Spotify Streams', 'YouTube Views', 'TikTok Likes', 'AirPlay Spins', 'Spotify Popularity']
        X = df[features]
        X = X.fillna(X.median())
        return X
    except Exception as e:
        st.error(f"Error in preprocess_linear: {e}")
        return None


def preprocess_neural(df):
    try:
        df = df.dropna()
        df = df.drop(columns=['en_name', 'th_name', 'province'])
        mlb = MultiLabelBinarizer()
        df['ingredients'] = df['ingredients'].apply(lambda x: x.split('+'))
        ingredients_encoded = mlb.fit_transform(df['ingredients'])
        cat_cols = ["course", "region"]
        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols)
        ])
        X_cat = preprocessor.fit_transform(df[cat_cols])
        X = np.hstack((ingredients_encoded, X_cat.toarray()))
        y = df['region'].astype("category").cat.codes
        return X, y
    except Exception as e:
        st.error(f"Error in preprocess_neural: {e}")
        return None

def page_linear_explanation():
    st.title("Linear Regression Model Explanation (Machine Learning)")
    st.write("""
    ## แนวทางการพัฒนา
    1. เก็บรวบรวมข้อมูลเพลงจากแหล่งต่างๆ เช่น Spotify, YouTube, TikTok, AirPlay
    2. เตรียมข้อมูล:
        * ทำความสะอาดข้อมูล (Data Cleaning)
        * จัดการค่า Missing Values
        * แปลงข้อมูล Categorical เป็น Numerical (Label Encoding)
        * Scaling ข้อมูล (MinMaxScaler)
    3. เลือก Feature: เลือก Features ที่มีผลต่อ Spotify Popularity เช่น Artist, Album Name, Release Date, จำนวนสตรีม, จำนวนวิว, จำนวนไลค์, จำนวนการเปิด, และ Spotify Popularity ของศิลปิน
    4. สร้างโมเดล: สร้างโมเดล Linear Regression
    5. ฝึกโมเดล: ฝึกโมเดลโดยใช้ข้อมูล Train
    6. ประเมินผล: ประเมินผลโมเดลโดยใช้ข้อมูล Test และวัดค่า MSE (Mean Squared Error) และ R2 Score
    7. ปรับปรุงโมเดล: ปรับปรุงโมเดลโดยปรับ Hyperparameters หรือเพิ่มข้อมูล

    ## ทฤษฎีของ Linear Regression
    Linear Regression เป็นอัลกอริทึมที่ใช้ทำนายค่าตัวเลข (Target) โดยใช้ความสัมพันธ์เชิงเส้นระหว่าง Target และ Features

    สมการ: `y = b0 + b1*x1 + b2*x2 + ... + bn*xn`

    * `y` คือค่า Target ที่ต้องการทำนาย
    * `b0` คือค่า Intercept
    * `b1, b2, ..., bn` คือค่า Coefficients ของ Features
    * `x1, x2, ..., xn` คือค่า Features

    ## ขั้นตอนการพัฒนาโมเดล
    1. โหลดข้อมูล: โหลดข้อมูลเพลงจากไฟล์ CSV
    2. เตรียมข้อมูล (Data Preprocessing):
        * แปลงข้อมูลวันที่ (Release Date) เป็นตัวเลข
        * แปลงข้อมูล Categorical เป็นตัวเลข (Label Encoding)
        * จัดการค่า Missing Values โดยแทนที่ด้วยค่า Median
        * Scaling ข้อมูลโดยใช้ MinMaxScaler
    3. แบ่งข้อมูล: แบ่งข้อมูลเป็น Train และ Test Sets
    4. สร้างและฝึกโมเดล: สร้างโมเดล Linear Regression และฝึกโมเดลโดยใช้ข้อมูล Train
    5. ประเมินผลโมเดล: ประเมินผลโมเดลโดยใช้ข้อมูล Test และวัดค่า MSE และ R2 Score
    6. บันทึกโมเดล: บันทึกโมเดลที่ฝึกแล้วเพื่อนำไปใช้งาน
    """)

def page_neural_explanation():
    st.title("Neural Network Model Explanation (Neural Network)")
    st.write("""
    ## แนวทางการพัฒนา
    1. เก็บรวบรวมข้อมูล: รวบรวมข้อมูลอาหารไทยจากแหล่งต่างๆ พร้อมทั้งข้อมูลส่วนผสม, หมวดหมู่, และภูมิภาค
    2. เตรียมข้อมูล:
        * ทำความสะอาดข้อมูล (Data Cleaning)
        * จัดการค่า Missing Values
        * แปลงข้อมูล Categorical เป็น Numerical (One-Hot Encoding)
        * แปลงส่วนผสมเป็น One-Hot Encoding (MultiLabelBinarizer)
    3. สร้างโมเดล: สร้างโมเดล Neural Network แบบ Sequential
    4. ฝึกโมเดล: ฝึกโมเดลโดยใช้ข้อมูล Train และ Class Weights เพื่อแก้ไขปัญหาข้อมูลไม่สมดุล
    5. ประเมินผล: ประเมินผลโมเดลโดยใช้ข้อมูล Test และวัดค่า Accuracy
    6. ปรับปรุงโมเดล: ปรับปรุงโมเดลโดยปรับ Hyperparameters หรือเพิ่มข้อมูล

    ## ทฤษฎีของ Neural Network
    Neural Network เป็นอัลกอริทึมที่จำลองการทำงานของสมองมนุษย์ โดยมี Layer ต่างๆ ที่เชื่อมต่อกันเพื่อเรียนรู้ Pattern จากข้อมูล

    * **Input Layer:** รับข้อมูลเข้าสู่โมเดล
    * **Hidden Layers:** Layer ที่อยู่ระหว่าง Input และ Output Layers ทำหน้าที่เรียนรู้ Features จากข้อมูล
    * **Output Layer:** แสดงผลลัพธ์ของโมเดล
    * **Activation Function:** ฟังก์ชันที่ใช้ในการแปลงค่าในแต่ละ Layer

    ## ขั้นตอนการพัฒนาโมเดล
    1. โหลดข้อมูล: โหลดข้อมูลอาหารไทยจากไฟล์ CSV
    2. เตรียมข้อมูล (Data Preprocessing):
        * ลบคอลัมน์ที่ไม่จำเป็น (en_name, th_name, province)
        * แปลงส่วนผสมเป็น One-Hot Encoding (MultiLabelBinarizer)
        * แปลงข้อมูล Categorical เป็น One-Hot Encoding (OneHotEncoder)
    3. แบ่งข้อมูล: แบ่งข้อมูลเป็น Train และ Test Sets
    4. คำนวณ Class Weights: คำนวณ Class Weights เพื่อแก้ไขปัญหาข้อมูลไม่สมดุล
    5. สร้างและฝึกโมเดล: สร้างโมเดล Neural Network แบบ Sequential และฝึกโมเดลโดยใช้ข้อมูล Train และ Class Weights
    6. ประเมินผลโมเดล: ประเมินผลโมเดลโดยใช้ข้อมูล Test และวัดค่า Accuracy
    7. บันทึกโมเดล: บันทึกโมเดลที่ฝึกแล้วเพื่อนำไปใช้งาน
    """)

def page_linear_demo():
    st.title("Linear Regression Model Demo")
    uploaded_file = st.file_uploader("Upload CSV file (Spotify Data)", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding="windows-1252")
        st.write("Available columns:", df.columns.tolist())

        X = preprocess_linear(df)
        if X is not None:
            predictions = np.clip(linear_model.predict(X), 0, 100)

            track_col = next((col for col in df.columns if "track" in col.lower() or "title" in col.lower()), None)
            if track_col:
                df[track_col] = df[track_col].fillna("Unknown")

            # เพิ่มกราฟแสดงผล
            if track_col:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.scatterplot(x=df[track_col], y=predictions.flatten(), ax=ax)
                plt.xticks(rotation=45, ha='right')
                plt.xlabel("Song Name")
                plt.ylabel("Predicted Spotify Popularity")
                st.pyplot(fig)

            for i, prediction in enumerate(predictions):
                song_name = df[track_col].iloc[i] if track_col else f"Unknown Song {i+1}"
                st.write(f"Song: {song_name}, Predicted Spotify Popularity: {prediction:.2f}")

            

def page_neural_demo():
    st.title("Neural Network Model Demo")
    uploaded_file = st.file_uploader("Upload CSV file (Thai Food Data)", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        st.write("Available columns:", df.columns.tolist())

        X, y = preprocess_neural(df)
        if X is not None:
            predictions = neural_network_model.predict(X)
            predicted_regions = np.argmax(predictions, axis=1)
            region_names = {0: "ภาคกลาง", 1: "ภาคเหนือ", 2: "ภาคตะวันออกเฉียงเหนือ", 3: "ภาคใต้"}

            for i, region_code in enumerate(predicted_regions):
                food_name = df['th_name'].iloc[i] if 'th_name' in df.columns else f"Unknown Food {i+1}"
                region_name = region_names.get(region_code, 'Unknown Region')
                st.write(f"Food: {food_name}, Predicted Region: {region_name}")

            # เพิ่มกราฟแสดงผล
            if 'th_name' in df.columns:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.countplot(x=[region_names.get(code, 'Unknown') for code in predicted_regions], ax=ax ,encoding="utf-8")
                plt.xlabel("Predicted Region")
                plt.ylabel("Number of Foods")
                st.pyplot(fig)

import streamlit as st

def references():
    st.markdown("<h1 class='main-header'>เอกสารและแหล่งข้อมูลอ้างอิง</h1>", unsafe_allow_html=True)

    # CSS for better Thai text rendering and overall styling
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;700&display=swap');

        * {
            font-family: 'Sarabun', sans-serif;
        }

        .main-header {
            color: #1E88E5;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #1E88E5;
        }

        .sub-header {
            color: #0D47A1;
            font-size: 1.7rem;
            font-weight: 600;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }

        .card {
            background-color: #ffffff;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 20px;
            transition: transform 0.2s;
        }

        .card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        }

        .card h3 {
            color: #1976D2;
            font-size: 1.3rem;
            margin-bottom: 15px;
            border-bottom: 1px solid #E3F2FD;
            padding-bottom: 10px;
        }

        .highlight {
            background-color: #F9FAFE;
        }

        .dataset-card {
            border-left: 4px solid #1E88E5;
            background-color: rgba(30, 136, 229, 0.08);
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
        }

        .dataset-card strong {
            color: #0D47A1;
        }

        .dataset-card p {
            margin-top: 8px;
            color: #333;
            line-height: 1.5;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        table th {
            background-color: #E3F2FD;
            color: #0D47A1;
            padding: 10px;
            text-align: left;
            border-bottom: 2px solid #90CAF9;
        }

        table td {
            padding: 10px;
            border-bottom: 1px solid #E3F2FD;
        }

        table tr:hover {
            background-color: #F5F5F5;
        }

        .code-section {
            background-color: #F8F9FA;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #1E88E5;
            margin-top: 15px;
            overflow-x: auto;
        }

        .code-section pre {
            margin: 0;
            font-family: monospace;
        }

        a {
            color: #1E88E5;
            text-decoration: none;
            transition: color 0.2s;
        }

        a:hover {
            color: #0D47A1;
            text-decoration: underline;
        }
    </style>
    """, unsafe_allow_html=True)

    # References section
    st.markdown("""
    <h2 class="sub-header">แหล่งข้อมูลและบทความวิชาการ</h2>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    # ===== DATASETS SECTION =====
    with col1:
        # Add first dataset
        st.markdown("""
            <div class="dataset-card">
                <strong>Dataset 1:</strong> Most Streamed Spotify Songs 2024<br>
                <a href="https://www.kaggle.com/datasets/nelgiriyewithana/most-streamed-spotify-songs-2024/data" target="_blank">
                    <img src="https://cdn-icons-png.flaticon.com/512/5968/5968848.png" width="20" style="margin-right: 5px; vertical-align: middle;">
                    Kaggle: Most Streamed Spotify Songs 2024
                </a>
                <p>
                    ข้อมูลเพลงที่มียอดสตรีมสูงสุดบน Spotify ในปี 2024
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        # Add second dataset
        st.markdown("""
            <div class="dataset-card">
                <strong>Dataset 2:</strong> Foods in Thailand<br>
                <a href="https://www.kaggle.com/datasets/ponthakornsodchun/foods-in-thailand" target="_blank">
                    <img src="https://cdn-icons-png.flaticon.com/512/5968/5968848.png" width="20" style="margin-right: 5px; vertical-align: middle;">
                    Kaggle: Foods in Thailand
                </a>
                <p>
                    ข้อมูลรายการอาหารไทย
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    # Libraries and tools
    st.markdown("""
    <h2 class="sub-header">ไลบรารีและเครื่องมือที่ใช้</h2>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Data Processing</h3>
            <ul>
                <li><b>pandas</b> - การจัดการและวิเคราะห์ข้อมูล</li>
                <li><b>numpy</b> - การคำนวณทางคณิตศาสตร์</li>
                <li><b>scikit-learn</b> - การเตรียมข้อมูลและอัลกอริทึม ML</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>Model Development</h3>
            <ul>
                <li><b>TensorFlow/Keras</b> - การสร้างโมเดลโครงข่ายประสาทเทียม</li>
                <li><b>scikit-learn</b> - การสร้างโมเดล Linear Regression</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Add more resources
    st.markdown("""
    <h2 class="sub-header">แหล่งข้อมูลเพิ่มเติม</h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>แหล่งเรียนรู้ออนไลน์</h3>
        <ul>
            <li><a href="https://www.tensorflow.org/tutorials" target="_blank">TensorFlow Tutorials</a> - แหล่งเรียนรู้การใช้งาน TensorFlow</li>
            <li><a href="https://scikit-learn.org/stable/tutorial/index.html" target="_blank">Scikit-learn Documentation</a> - เอกสารและตัวอย่างการใช้งาน scikit-learn</li>
            <li><a href="https://docs.streamlit.io/" target="_blank">Streamlit Documentation</a> - เอกสารและตัวอย่างการใช้งาน Streamlit</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.sidebar.title("Final Project IS")
    page = st.sidebar.radio("Go to", ("Linear Explanation", "Neural Explanation", "Linear Demo", "Neural Demo", "references"))

    if page == "Linear Explanation":
        page_linear_explanation()
    elif page == "Neural Explanation":
        page_neural_explanation()
    elif page == "Linear Demo":
        page_linear_demo()
    elif page == "Neural Demo":
        page_neural_demo()
    elif page == "references":
        references()    

if __name__ == "__main__":
    main()
