import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. VERİYİ OKUMA VE HIZLANDIRMA (DOWNSAMPLING)
# Senin belirttiğin proje klasöründeki dosya yolu
ground_truth = './project_src/groundtruth.txt' 

kolon_isimleri = ['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
df_gt = pd.read_csv(ground_truth, sep='\s+', comment='#', names=kolon_isimleri)

# 500 Hz animasyonda kasmasın diye 50 Hz'e düşürüyoruz (10 satırda bir alıyoruz)
df_gt = df_gt.iloc[::10].reset_index(drop=True)
print(df_gt)
 
# 2. QUATERNION'DAN 3 EKSENLİ ROTASYON MATRİSİ ÇIKARMA
x = df_gt['qx']
y = df_gt['qy']
z = df_gt['qz']
w = df_gt['qw']

# X Ekseni (Kırmızı - Dronun Burnu / Roll Ekseni)
dx1 = 1 - 2*(y**2 + z**2)
dy1 = 2*(x*y + w*z)
dz1 = 2*(x*z - w*y)

# Y Ekseni (Yeşil - Dronun Sağ-Sol Kanadı / Pitch Ekseni)
dx2 = 2*(x*y - w*z)
dy2 = 1 - 2*(x**2 + z**2)
dz2 = 2*(y*z + w*x)

# Z Ekseni (Mavi - Dronun Üstü-Altı / Yaw Ekseni)
dx3 = 2*(x*z + w*y)
dy3 = 2*(y*z - w*x)
dz3 = 1 - 2*(x**2 + y**2)

# Okların Uzunluğu (Sol ekran için 0.4 metre idealdir)
ok_boyu = 0.4

# Sol Ekran için uç noktaların dünya koordinatlarını hesaplama
df_gt['tip_x_X'] = df_gt['tx'] + ok_boyu * dx1
df_gt['tip_y_X'] = df_gt['ty'] + ok_boyu * dy1
df_gt['tip_z_X'] = df_gt['tz'] + ok_boyu * dz1

df_gt['tip_x_Y'] = df_gt['tx'] + ok_boyu * dx2
df_gt['tip_y_Y'] = df_gt['ty'] + ok_boyu * dy2
df_gt['tip_z_Y'] = df_gt['tz'] + ok_boyu * dz2

df_gt['tip_x_Z'] = df_gt['tx'] + ok_boyu * dx3
df_gt['tip_y_Z'] = df_gt['ty'] + ok_boyu * dy3
df_gt['tip_z_Z'] = df_gt['tz'] + ok_boyu * dz3

# 3. İKİLİ EKRAN (SUBPLOT) ORTAMINI HAZIRLAMA
fig = plt.figure(figsize=(16, 7))

# --- SOL PANEL: UZAYDAKİ KONUM VE HAREKET (TRANSLATION) ---
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_xlim([df_gt['tx'].min() - 1, df_gt['tx'].max() + 1])
ax1.set_ylim([df_gt['ty'].min() - 1, df_gt['ty'].max() + 1])
ax1.set_zlim([df_gt['tz'].min() - 1, df_gt['tz'].max() + 1])
ax1.set_title('Odadaki Anlık Konum (Translation)')
ax1.set_xlabel('X Pozisyonu (m)')
ax1.set_ylabel('Y Pozisyonu (m)')
ax1.set_zlabel('Z Pozisyonu (m)')

# Sol panelin görsel aktörleri
cizgi_kuyruk, = ax1.plot([], [], [], color='black', linewidth=1, alpha=0.3, label='Uçuş Rotası')
nokta_drone, = ax1.plot([], [], [], marker='o', color='black', markersize=4)
sol_ok_X, = ax1.plot([], [], [], color='red', linewidth=2, label='X Ekseni (Burun)')
sol_ok_Y, = ax1.plot([], [], [], color='green', linewidth=2, label='Y Ekseni (Kanat)')
sol_ok_Z, = ax1.plot([], [], [], color='blue', linewidth=2, label='Z Ekseni (Üst)')
ax1.legend(loc='upper left')

# --- SAĞ PANEL: MERKEZE SABİTLENMİŞ YÖNELİM (ATTITUDE INDICATOR) ---
ax2 = fig.add_subplot(122, projection='3d')
# Sağ taraf sadece dönüşe odaklandığı için eksenleri [-1, 1] arasına çaktık
ax2.set_xlim([-1, 1])
ax2.set_ylim([-1, 1])
ax2.set_zlim([-1, 1])
ax2.set_title('Sadece Gövde Yönelimi / Attitude (Merkeze Sabit)')
ax2.set_xlabel('Roll Ekseni')
ax2.set_ylabel('Pitch Ekseni')
ax2.set_zlabel('Yaw Ekseni')

# Merkeze (0,0,0) sabitlenmiş kalın oklar (Ekranda net dursun diye boyunu 2.5 kat büyüttük)
sag_ok_X, = ax2.plot([], [], [], color='red', linewidth=5, label='Burun (X)')
sag_ok_Y, = ax2.plot([], [], [], color='green', linewidth=5, label='Kanat (Y)')
sag_ok_Z, = ax2.plot([], [], [], color='blue', linewidth=5, label='Tavan (Z)')
ax2.legend(loc='upper left')

plt.suptitle('AE 484 - Otonom Drone Entegre GCS Dashboard', fontsize=14, fontweight='bold')

# 4. ANİMASYON DÖNGÜSÜ
def animasyon_adimi(frame):
    anlik_tx = df_gt['tx'].iloc[frame]
    anlik_ty = df_gt['ty'].iloc[frame]
    anlik_tz = df_gt['tz'].iloc[frame]
    
    # Her bir eksenin anlık yön vektör bileşenlerini çekiyoruz
    tX_x, tX_y, tX_z = dx1.iloc[frame] * ok_boyu, dy1.iloc[frame] * ok_boyu, dz1.iloc[frame] * ok_boyu
    tY_x, tY_y, tY_z = dx2.iloc[frame] * ok_boyu, dy2.iloc[frame] * ok_boyu, dz2.iloc[frame] * ok_boyu
    tZ_x, tZ_y, tZ_z = dx3.iloc[frame] * ok_boyu, dy3.iloc[frame] * ok_boyu, dz3.iloc[frame] * ok_boyu
    
    # --- SOL PANEL GÜNCELLEME (HAREKETLİ) ---
    cizgi_kuyruk.set_data(df_gt['tx'].iloc[:frame], df_gt['ty'].iloc[:frame])
    cizgi_kuyruk.set_3d_properties(df_gt['tz'].iloc[:frame])
    
    nokta_drone.set_data([anlik_tx], [anlik_ty])
    nokta_drone.set_3d_properties([anlik_tz])
    
    sol_ok_X.set_data([anlik_tx, df_gt['tip_x_X'].iloc[frame]], [anlik_ty, df_gt['tip_y_X'].iloc[frame]])
    sol_ok_X.set_3d_properties([anlik_tz, df_gt['tip_z_X'].iloc[frame]])
    
    sol_ok_Y.set_data([anlik_tx, df_gt['tip_x_Y'].iloc[frame]], [anlik_ty, df_gt['tip_y_Y'].iloc[frame]])
    sol_ok_Y.set_3d_properties([anlik_tz, df_gt['tip_z_Y'].iloc[frame]])
    
    sol_ok_Z.set_data([anlik_tx, df_gt['tip_x_Z'].iloc[frame]], [anlik_ty, df_gt['tip_y_Z'].iloc[frame]])
    sol_ok_Z.set_3d_properties([anlik_tz, df_gt['tip_z_Z'].iloc[frame]])
    
    # --- SAĞ PANEL GÜNCELLEME (MERKEZE SABİT - 0,0,0) ---
    # Çizgileri hep orijinden başlatıp sadece yön bileşenlerine kadar uzatıyoruz
    sag_ok_X.set_data([0, tX_x * 2.2], [0, tX_y * 2.2])
    sag_ok_X.set_3d_properties([0, tX_z * 2.2])
    
    sag_ok_Y.set_data([0, tY_x * 2.2], [0, tY_y * 2.2])
    sag_ok_Y.set_3d_properties([0, tY_z * 2.2])
    
    sag_ok_Z.set_data([0, tZ_x * 2.2], [0, tZ_y * 2.2])
    sag_ok_Z.set_3d_properties([0, tZ_z * 2.2])
    
    return cizgi_kuyruk, nokta_drone, sol_ok_X, sol_ok_Y, sol_ok_Z, sag_ok_X, sag_ok_Y, sag_ok_Z

# 5. MOTORU ÇALIŞTIR
ani = FuncAnimation(fig, animasyon_adimi, frames=len(df_gt), interval=20, blit=False)

plt.show()