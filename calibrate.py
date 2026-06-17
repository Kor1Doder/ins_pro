import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as  np


# 1. Veriyi yükle (Dosya adını doğru yazdığından emin ol)
# Dosyan boşlukla ayrılmışsa 'sep=" "' kullan
df = pd.read_csv('project_src/groundtruth.txt', sep=' ', header=0, 
                 names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
# Zaman farkı (dt) hesapla
df['dt'] = df['timestamp'].diff()

# Hız hesapla (Delta Konum / Delta Zaman)
df['vx'] = df['tx'].diff() / df['dt']
df['vy'] = df['ty'].diff() / df['dt']
df['vz'] = df['tz'].diff() / df['dt']

# Toplam hız (Magnitude)
df['v_total'] = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)

# Eşik değeri belirle (Örneğin hız 0.05 m/s'yi geçtiği an hareket başlamıştır)
start_index = df[df['v_total'] > 0.05].index[0]
print(f"Drone {start_index} zamanında harekete başladı.")
exit()
# 2. 3D Yörünge Grafiği (Trajectory)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(df['tx'], df['ty'], df['tz'], label='Drone Yolu')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Drone Uçuş Yolu (Ground Truth)')
plt.legend()
plt.show()

# 3. Zamanla Konum Değişimi (Titreme/Stabilite Kontrolü)
plt.figure(figsize=(10, 4))
plt.plot(df['timestamp'], df['tz'], label='Z Konumu (Yükseklik)')
plt.title('Z Ekseni Stabilitesi (İniş/Yükseklik)')
plt.xlabel('Zaman (s)')
plt.ylabel('Yükseklik (m)')
plt.grid(True)
plt.show()


import numpy as np

def quaternion_to_euler(q):
    # 1. Normalizasyon (Önleyici tedbir)
    norm = np.linalg.norm(q)
    if norm == 0: return [0, 0, 0] # Sıfır hatasını önle
    q = q / norm
    
    qx, qy, qz, qw = q

    # Roll (X-ekseni)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx**2 + qy**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (Y-ekseni) - Burası en kritik yer
    sinp = 2 * (qw * qy - qz * qx)
    # 1.0'ı aşan değerleri [-1, 1] arasına kırp (NaN hatasının sebebi bu)
    sinp = np.clip(sinp, -1.0, 1.0) 
    pitch = np.arcsin(sinp)

    # Yaw (Z-ekseni)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy**2 + qz**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees([roll, pitch, yaw])

import pandas as pd
import numpy as np

# 'comment' parametresi o baştaki # işaretini görmezden gelmesini sağlar.
# 'sep' olarak '\s+' kullanıyoruz çünkü dosyadaki boşluklar değişken (tab veya boşluk olabilir).
df = pd.read_csv('project_src/groundtruth.txt', 
                 sep='\s+', 
                 comment='#', 
                 header=None, 
                 names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])

# Şimdi sayısal olduklarından emin olalım (gereksiz boşlukları temizler)
cols = ['tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

# NaN olan satır varsa (başlık satırı vs gibi) uçuralım
df = df.dropna()

# 1. Önce Quaternion sütunlarını normalize edelim (her ihtimale karşı)
# Normalize etmezsen euler açıların saçmalar.
def normalize_quaternion(row):
    q = np.array([row['qx'], row['qy'], row['qz'], row['qw']])
    norm = np.linalg.norm(q)
    return q / norm

# 2. DataFrame'e uygulayalım
# Önce normalizasyon
df[['qx', 'qy', 'qz', 'qw']] = df.apply(lambda row: pd.Series(normalize_quaternion(row)), axis=1)

# 3. Senin fonksiyonunu kullanarak yeni sütunlar oluşturalım
def get_euler_from_row(row):
    q = [row['qx'], row['qy'], row['qz'], row['qw']]
    return quaternion_to_euler(q)

# 'apply' ile her satır için fonksiyonu çalıştır
euler_results = df.apply(lambda row: get_euler_from_row(row), axis=1)

# Sonuçları DataFrame'e 'roll', 'pitch', 'yaw' olarak ekle
df[['roll', 'pitch', 'yaw']] = pd.DataFrame(euler_results.tolist(), index=df.index)

# Şimdi kontrol et bakalım, değerler gelmiş mi?
print(df[['roll', 'pitch', 'yaw']].head())

plt.figure(figsize=(10, 5))
plt.plot(df['timestamp'], df['roll'], label='Roll')
plt.plot(df['timestamp'], df['pitch'], label='Pitch')
plt.plot(df['timestamp'], df['yaw'], label='Yaw')
plt.legend()
plt.title("Drone Oryantasyon (Euler) Analizi")
plt.grid(True)
plt.show()




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# 1. Veriyi Yükle
df = pd.read_csv('project_src/groundtruth.txt', sep='\s+', comment='#', header=None, 
                 names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])

# 2. İlk 50 satırı "Sessizlik/Başlangıç" olarak kabul et
initial_quat = df.iloc[:50][['qx', 'qy', 'qz', 'qw']].mean().values
initial_quat /= np.linalg.norm(initial_quat) # Normalizasyon

# 3. Tüm veriyi bu başlangıç oryantasyonuna göre hizala (Relative Rotation)
# Yani: Yeni_Oryantasyon = Mevcut * (Başlangıç_Tersine)
def align_orientation(row):
    q_curr = R.from_quat([row['qx'], row['qy'], row['qz'], row['qw']])
    q_init = R.from_quat(initial_quat)
    # Relative rotation (Hizalanmış açı)
    q_rel = q_init.inv() * q_curr
    return q_rel.as_euler('zyx', degrees=True)

df[['roll', 'pitch', 'yaw']] = df.apply(align_orientation, axis=1, result_type='expand')

# 4. Görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(df['timestamp'], df['roll'], label='Hizalı Roll')
plt.plot(df['timestamp'], df['pitch'], label='Hizalı Pitch')
plt.plot(df['timestamp'], df['yaw'], label='Hizalı Yaw')
plt.axhline(0, color='black', linestyle='--') # Sıfır çizgisi
plt.title('Hizalanmış Başlangıç Oryantasyonu (Relative to t=0)')
plt.legend()
plt.grid(True)
plt.show()