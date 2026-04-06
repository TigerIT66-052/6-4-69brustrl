# 🏍️ Dashboard นักท่องเที่ยวบุรีรัมย์

แอปพลิเคชัน Streamlit สำหรับวิเคราะห์และพยากรณ์จำนวนนักท่องเที่ยวจังหวัดบุรีรัมย์

## 📁 โครงสร้างไฟล์

```
├── app.py                        # Streamlit main app
├── dataCI02-09-03-2569.csv       # ชุดข้อมูลนักท่องเที่ยว
├── requirements.txt              # Python dependencies
└── README.md
```

## 🚀 วิธีรัน Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy บน Streamlit Cloud (ผ่าน GitHub)

1. Push ไฟล์ทั้งหมดขึ้น GitHub repository
2. ไปที่ [share.streamlit.io](https://share.streamlit.io)
3. เชื่อมต่อ GitHub → เลือก repo → ระบุ `app.py` เป็น main file
4. กด **Deploy!**

> ⚠️ ต้องอัปโหลดไฟล์ `dataCI02-09-03-2569.csv` ไว้ใน root ของ repo ด้วย

## 📊 ฟีเจอร์หลัก

| แท็บ | รายละเอียด |
|------|-----------|
| 🏠 ภาพรวม | KPI cards, แนวโน้มรายปี, สัดส่วนไทย/ต่างชาติ |
| 📈 การทำนาย | พยากรณ์ปี 2569, Sensitivity Analysis |
| 📅 รายเดือน | กราฟรายเดือน/ไตรมาส, Heatmap เปรียบเทียบ |
| 🎪 ผลกระทบเหตุการณ์ | MotoGP, Marathon, Phanom Rung, COVID |
| 🤖 เปรียบเทียบโมเดล | MAE / RMSE / R², Radar Chart, Feature Importance |

## 🤖 โมเดล ML ที่ใช้

- Linear Regression
- Ridge Regression  
- **Random Forest Regression** ← โมเดลที่ดีที่สุดสำหรับชุดข้อมูลนี้
- Gradient Boosting Regression

ประเมินด้วย MAE, RMSE, R²

## 📌 หมายเหตุ

- ข้อมูลปี 2556–2565 อยู่ในรูปแบบรายไตรมาส
- ข้อมูลปี 2566–2568 อยู่ในรูปแบบรายเดือน
- ปี 2563–2564 ได้รับผลกระทบจาก COVID-19
