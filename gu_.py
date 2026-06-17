import pandas as pd
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import cv2  # Kamerayı açmak için

# sistemi bölem için 
frame_set=10

# 1. VERİYİ OKUMA VE HIZLANDIRMA
ground_truth = './project_src/groundtruth.txt' 
kolon_isimleri = ['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
df_gt = pd.read_csv(ground_truth, sep='\s+', comment='#', names=kolon_isimleri)

# 2. QUATERNION VE MATEMATİKSEL HESAPLAR
x, y, z, w = df_gt['qx'], df_gt['qy'], df_gt['qz'], df_gt['qw']

# 3 Eksen Yönleri
dx1 = 1 - 2*(y**2 + z**2); dy1 = 2*(x*y + w*z); dz1 = 2*(x*z - w*y)
dx2 = 2*(x*y - w*z); dy2 = 1 - 2*(x**2 + z**2); dz2 = 2*(y*z + w*x)
dx3 = 2*(x*z + w*y); dy3 = 2*(y*z - w*x); dz3 = 1 - 2*(x**2 + y**2)

ok_boyu = 0.4
# Uç noktalar (Sol 3D ekran için)
df_gt['tip_x_X'] = df_gt['tx'] + ok_boyu * dx1; df_gt['tip_y_X'] = df_gt['ty'] + ok_boyu * dy1; df_gt['tip_z_X'] = df_gt['tz'] + ok_boyu * dz1
df_gt['tip_x_Y'] = df_gt['tx'] + ok_boyu * dx2; df_gt['tip_y_Y'] = df_gt['ty'] + ok_boyu * dy2; df_gt['tip_z_Y'] = df_gt['tz'] + ok_boyu * dz2
df_gt['tip_x_Z'] = df_gt['tx'] + ok_boyu * dx3; df_gt['tip_y_Z'] = df_gt['ty'] + ok_boyu * dy3; df_gt['tip_z_Z'] = df_gt['tz'] + ok_boyu * dz3


# --- KAMERA VERİLERİNİ OKUMA VE SENKRONİZE ETME ---
# 1. comment='#' ekledik ki üstteki yazıları atlasın
df_left = pd.read_csv('./project_src/left_images.txt', sep='\s+', comment='#', names=['id', 'timestamp', 'file_left'])
df_right = pd.read_csv('./project_src/right_images.txt', sep='\s+', comment='#', names=['id', 'timestamp', 'file_right'])

# 2. İşi garantiye alıyoruz: Zaman sütununu KESİNLİKLE ondalıklı sayı (float) yapıyoruz
df_left['timestamp'] = df_left['timestamp'].astype(float)
df_right['timestamp'] = df_right['timestamp'].astype(float)
df_gt['timestamp'] = df_gt['timestamp'].astype(float)
# =================================================================
# YENİ EKLENECEK MÜHENDİSLİK HİLESİ: ZAMAN KESİŞİMİ (OVERLAP) BULMA
# =================================================================
# Üç verinin de en GEC başlayanını bul (Ortak başlangıç noktası)
ortak_baslangic = max(df_gt['timestamp'].min(), df_left['timestamp'].min(), df_right['timestamp'].min())

# Üç verinin de en ERKEN bitenini bul (Ortak bitiş noktası)
ortak_bitis = min(df_gt['timestamp'].max(), df_left['timestamp'].max(), df_right['timestamp'].max())

# Ground Truth tablosunu sadece bu ortak zaman aralığına göre kırpıyoruz
df_gt = df_gt[(df_gt['timestamp'] >= ortak_baslangic) & (df_gt['timestamp'] <= ortak_bitis)]

# İndeksleri sıfırlayalım ki alttaki for döngüsü (iloc) patlamasın
df_gt = df_gt.reset_index(drop=True)
# =================================================================
# 3. Sıralama ve Birleştirme (Burası aynı)
df_gt = df_gt.sort_values('timestamp')
df_left = df_left.sort_values('timestamp')
df_right = df_right.sort_values('timestamp')

df_gt = pd.merge_asof(df_gt, df_left[['timestamp', 'file_left']], on='timestamp', direction='nearest')
df_gt = pd.merge_asof(df_gt, df_right[['timestamp', 'file_right']], on='timestamp', direction='nearest')

def quaternion_to_euler(qx, qy, qz, qw):
    # Roll (X ekseni)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (Y ekseni)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = np.copysign(np.pi / 2, sinp) if np.abs(sinp) >= 1 else np.arcsin(sinp)

    # Yaw (Z ekseni)
    siny_cosp = 2 * (qw * qz + qx * qy)
    # İŞTE DÜZELTİLEN SATIR BURASI: qy * qy oldu
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz) 
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

# =====================================================================
# 3. ANA UYGULAMA (TKINTER GUI + MATPLOTLIB BİRLEŞİMİ)
# =====================================================================
root = tk.Tk()
root.title("AE 484 - Otonom Drone GCS Dashboard")
root.geometry("1400x700")
root.configure(bg="#1e1e1e")

# --- SOL PANEL: DİJİTAL TELEMETRİ (Tkinter) ---
frame_telemetri = tk.Frame(root, bg="#1e1e1e", width=300)
frame_telemetri.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)

tk.Label(frame_telemetri, text="CANLI TELEMETRİ", font=("Arial", 16, "bold"), fg="#00ff00", bg="#1e1e1e").pack(pady=10)
lbl_zaman = tk.Label(frame_telemetri, text="Zaman: 0.00 s", font=("Courier", 14), fg="white", bg="#1e1e1e")
lbl_zaman.pack(pady=5)

tk.Label(frame_telemetri, text="-- KONUM --", font=("Arial", 12, "bold"), fg="cyan", bg="#1e1e1e").pack(pady=10)
lbl_x = tk.Label(frame_telemetri, text="X: 0.00 m", font=("Courier", 14), fg="white", bg="#1e1e1e")
lbl_y = tk.Label(frame_telemetri, text="Y: 0.00 m", font=("Courier", 14), fg="white", bg="#1e1e1e")
lbl_z = tk.Label(frame_telemetri, text="Z: 0.00 m", font=("Courier", 14), fg="white", bg="#1e1e1e")
lbl_x.pack(); lbl_y.pack(); lbl_z.pack()

tk.Label(frame_telemetri, text="-- YÖNELİM --", font=("Arial", 12, "bold"), fg="yellow", bg="#1e1e1e").pack(pady=10)
lbl_roll = tk.Label(frame_telemetri, text="Roll : 0.0°", font=("Courier", 14), fg="white", bg="#1e1e1e")
lbl_pitch= tk.Label(frame_telemetri, text="Pitch: 0.0°", font=("Courier", 14), fg="white", bg="#1e1e1e")
lbl_yaw  = tk.Label(frame_telemetri, text="Yaw  : 0.0°", font=("Courier", 14), fg="white", bg="#1e1e1e")
lbl_roll.pack(); lbl_pitch.pack(); lbl_yaw.pack()

# --- SAĞ PANEL: 3D GRAFİKLER (Matplotlib) ---
fig = plt.figure(figsize=(10, 6))
fig.patch.set_facecolor('#1e1e1e') # Grafiğin arkaplanını da koyu yapalım uyumlu olsun

# Sol 3D (Hareket)
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_xlim([df_gt['tx'].min()-1, df_gt['tx'].max()+1]); ax1.set_ylim([df_gt['ty'].min()-1, df_gt['ty'].max()+1]); ax1.set_zlim([df_gt['tz'].min()-1, df_gt['tz'].max()+1])
ax1.set_title('Uzaydaki Konum', color='white')
cizgi_kuyruk, = ax1.plot([], [], [], color='white', alpha=0.3)
nokta_drone, = ax1.plot([], [], [], marker='o', color='white')
sol_ok_X, = ax1.plot([], [], [], color='red', linewidth=2); sol_ok_Y, = ax1.plot([], [], [], color='green', linewidth=2); sol_ok_Z, = ax1.plot([], [], [], color='blue', linewidth=2)

# Sağ 3D (Yönelim)
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_xlim([-1, 1]); ax2.set_ylim([-1, 1]); ax2.set_zlim([-1, 1])
ax2.set_title('Gövde Yönelimi (Attitude)', color='white')
sag_ok_X, = ax2.plot([], [], [], color='red', linewidth=5); sag_ok_Y, = ax2.plot([], [], [], color='green', linewidth=5); sag_ok_Z, = ax2.plot([], [], [], color='blue', linewidth=5)

# Grafiklerin eksen arkaplanlarını düzenleme
for ax in [ax1, ax2]:
    ax.set_facecolor('#1e1e1e')
    ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white'); ax.zaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white'); ax.tick_params(axis='z', colors='white')

# MATPLOTLIB FİGÜRÜNÜ TKINTER'A GÖMME İŞLEMİ (Sihir burada)
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# 4. ORTAK GÜNCELLEME (HEM GRAFİK HEM DİJİTAL YAZILAR)
def animasyon_adimi(gercek_indeks):
    # gercek_indeks artık 0, 1, 2 değil; 0, 10, 20, 30 olarak gelecek!
    row = df_gt.iloc[gercek_indeks]
    roll, pitch, yaw = quaternion_to_euler(row['qx'], row['qy'], row['qz'], row['qw'])
    
    lbl_zaman.config(text=f"Zaman: {row['timestamp']:.2f} s")
    lbl_x.config(text=f"X: {row['tx']:6.2f} m"); lbl_y.config(text=f"Y: {row['ty']:6.2f} m"); lbl_z.config(text=f"Z: {row['tz']:6.2f} m")
    lbl_roll.config(text=f"Roll : {roll:6.1f}°"); lbl_pitch.config(text=f"Pitch: {pitch:6.1f}°"); lbl_yaw.config(text=f"Yaw  : {yaw:6.1f}°")

    # 2. Matplotlib 3D Grafikleri Güncelle
    tx, ty, tz = row['tx'], row['ty'], row['tz']
    
    # Eksen yönlerini çekerken de gerçek indeksi kullanıyoruz
    tX_x, tX_y, tX_z = dx1.iloc[gercek_indeks]*ok_boyu, dy1.iloc[gercek_indeks]*ok_boyu, dz1.iloc[gercek_indeks]*ok_boyu
    tY_x, tY_y, tY_z = dx2.iloc[gercek_indeks]*ok_boyu, dy2.iloc[gercek_indeks]*ok_boyu, dz2.iloc[gercek_indeks]*ok_boyu
    tZ_x, tZ_y, tZ_z = dx3.iloc[gercek_indeks]*ok_boyu, dy3.iloc[gercek_indeks]*ok_boyu, dz3.iloc[gercek_indeks]*ok_boyu
    
    # MÜHENDİSLİK HİLESİ: Kuyruğu çizerken de [::frame_set] diyerek sadece 50Hz'lik noktaları çizdiriyoruz ki Matplotlib binlerce noktayı çizip kasmasın
    cizgi_kuyruk.set_data(df_gt['tx'].iloc[:gercek_indeks:frame_set], df_gt['ty'].iloc[:gercek_indeks:frame_set])
    cizgi_kuyruk.set_3d_properties(df_gt['tz'].iloc[:gercek_indeks:frame_set])
    
    nokta_drone.set_data([tx], [ty]); nokta_drone.set_3d_properties([tz])
    
    sol_ok_X.set_data([tx, tx+tX_x], [ty, ty+tX_y]); sol_ok_X.set_3d_properties([tz, tz+tX_z])
    sol_ok_Y.set_data([tx, tx+tY_x], [ty, ty+tY_y]); sol_ok_Y.set_3d_properties([tz, tz+tY_z])
    sol_ok_Z.set_data([tx, tx+tZ_x], [ty, ty+tZ_y]); sol_ok_Z.set_3d_properties([tz, tz+tZ_z])
    
    sag_ok_X.set_data([0, tX_x*2.2], [0, tX_y*2.2]); sag_ok_X.set_3d_properties([0, tX_z*2.2])
    sag_ok_Y.set_data([0, tY_x*2.2], [0, tY_y*2.2]); sag_ok_Y.set_3d_properties([0, tY_z*2.2])
    sag_ok_Z.set_data([0, tZ_x*2.2], [0, tZ_y*2.2]); sag_ok_Z.set_3d_properties([0, tZ_z*2.2])




    resim_sol_adi = row['file_left']
    resim_sag_adi = row['file_right']
    
    # Dosyaların bilgisayarındaki tam yollarını oluştur (Başına klasör yolunu ekleyerek)
    # Eğer txt içindeki isimler zaten tam yolsa bu birleştirmeyi yapmana gerek kalmaz
    yol_sol = f"./project_src/{resim_sol_adi}"
    yol_sag = f"./project_src/{resim_sag_adi}"
    
    # OpenCV ile resimleri diskten okuyoruz
    img_l = cv2.imread(yol_sol)
    img_r = cv2.imread(yol_sag)
    
    # Eğer dosyalar klasörde varsa ve başarıyla okunduysa ekrana bas
    if img_l is not None and img_r is not None:
        # Ekranı kaplamasınlar diye pencereleri %50 oranında küçültüyoruz
        img_l = cv2.resize(img_l, (0, 0), fx=0.8, fy=0.8)
        img_r = cv2.resize(img_r, (0, 0), fx=0.8, fy=0.8)
        
        # İki resmi yatay olarak yan yana yapıştırıyoruz (Stereo Canlı Yayın)
        stereo_ekran = np.hstack((img_l, img_r))
        
        # Harici OpenCV penceresinde görüntüyü patlatıyoruz
        cv2.imshow("GCS - Canli Stereo Kamera Feed (Sol | Sag)", stereo_ekran)
        
        # ÇOK KRİTİK: OpenCV penceresinin donmasını ve kilitlenmesini engellemek için 
        # arayüze 1 milisaniyelik nefes alma süresi tanıyoruz.
        cv2.waitKey(1)
    
    return cizgi_kuyruk, nokta_drone, sol_ok_X, sol_ok_Y, sol_ok_Z, sag_ok_X, sag_ok_Y, sag_ok_Z

# İŞTE BURASI ÇOK KRİTİK: frames parametresine len(df_gt) yerine range verdik!
# 0'dan başla, dosyanın sonuna kadar git, ama frame_set (10) kadar atlaya atlaya git.
ani = FuncAnimation(fig, animasyon_adimi, frames=range(0, len(df_gt), frame_set), interval=20, blit=False)

# Uygulamayı Başlat
root.mainloop() 