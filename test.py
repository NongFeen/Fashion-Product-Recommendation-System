import pandas as pd

# ระบุ path ของไฟล์ CSV
csv_path = 'dataset/fix_styles.csv'

# อ่านไฟล์ CSV
df = pd.read_csv(csv_path, delimiter=';')

# แสดงข้อมูล 5 แถวแรก เพื่อดูโครงสร้างของข้อมูล
print(df.head())

# เข้าถึงข้อมูลของ product 44065 (สมมติว่า product ID อยู่ในคอลัมน์แรก)


