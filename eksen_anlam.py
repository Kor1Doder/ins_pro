import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math

# ============================================================
# DATA
# ============================================================

df = pd.read_csv(
    "project_src/groundtruth.txt",
    sep=r"\s+",
    comment="#",
    header=None,
    names=[
        "timestamp",
        "tx","ty","tz",
        "qx","qy","qz","qw"
    ]
)

row = df.iloc[0]

q = row[["qx","qy","qz","qw"]].values

rot = R.from_quat(q)

roll,pitch,yaw = rot.as_euler('xyz',degrees=True)
print(f"roll{roll}////pitch{pitch}///yaw{yaw}")
Rmat = rot.as_matrix()

# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(111, projection='3d')

# ============================================================
# WORLD FRAME (FIXED)
# ============================================================

world_axes = np.eye(3)

world_names = ["Xw","Yw","Zw"]

for axis,name in zip(world_axes,world_names):

    ax.quiver(
        0,0,0,
        axis[0],
        axis[1],
        axis[2],
        color='black',
        linewidth=4
    )

    ax.text(
        axis[0]*1.15,
        axis[1]*1.15,
        axis[2]*1.15,
        name,
        fontsize=18,
        color='black',
        fontweight='bold'
    )

# ============================================================
# BODY FRAME
# ============================================================

body_axes = np.eye(3)

colors = ['red','green','blue']

body_names = ['Xb','Yb','Zb']

for axis,color,name in zip(body_axes,colors,body_names):

    v = rot.apply(axis)

    ax.quiver(
        0,0,0,
        v[0],
        v[1],
        v[2],
        color=color,
        linewidth=4
    )

    ax.text(
        v[0]*1.15,
        v[1]*1.15,
        v[2]*1.15,
        name,
        fontsize=18,
        color=color,
        fontweight='bold'
    )

# ============================================================
# ANNOTATIONS
# ============================================================

ax.text(
    0.25,
    0.25,
    -0.35,
    "WORLD FRAME\n(FIXED / GLOBAL)",
    fontsize=12,
    bbox=dict(
        facecolor='white',
        edgecolor='black'
    )
)

ax.text(
    0.6,
    0.25,
    0.6,
    "DRONE BODY FRAME",
    fontsize=12,
    color='blue',
    bbox=dict(
        facecolor='white',
        edgecolor='blue'
    )
)

# ============================================================
# AXES
# ============================================================

ax.set_xlabel("X [m]",fontsize=14)
ax.set_ylabel("Y [m]",fontsize=14)
ax.set_zlabel("Z [m]",fontsize=14)

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])

ax.set_box_aspect([1,1,1])

ax.grid(True)

# Kamera açısı
ax.view_init(
    elev=25,
    azim=45
)

# ============================================================
# QUATERNION BOX
# ============================================================

quat_text = (
    f"QUATERNION\n\n"
    f"qx = {q[0]:.4f}\n"
    f"qy = {q[1]:.4f}\n"
    f"qz = {q[2]:.4f}\n"
    f"qw = {q[3]:.4f}"
)

fig.text(
    0.76,
    0.73,
    quat_text,
    fontsize=12,
    family='monospace',
    bbox=dict(
        facecolor='white',
        edgecolor='blue',
        boxstyle='round,pad=0.5'
    )
)

# ============================================================
# EULER BOX
# ============================================================

euler_text = (
    f"EULER ANGLES (XYZ)\n\n"
    f"Roll  = {roll:.2f} deg\n"
    f"Pitch = {pitch:.2f} deg\n"
    f"Yaw   = {yaw:.2f} deg"
)

fig.text(
    0.76,
    0.52,
    euler_text,
    fontsize=12,
    family='monospace',
    bbox=dict(
        facecolor='white',
        edgecolor='green',
        boxstyle='round,pad=0.5'
    )
)

# ============================================================
# ROTATION MATRIX BOX
# ============================================================

mat_text = (
    "ROTATION MATRIX\n\n"
    + np.array2string(
        Rmat,
        precision=3,
        suppress_small=True
    )
)

fig.text(
    0.76,
    0.25,
    mat_text,
    fontsize=10,
    family='monospace',
    bbox=dict(
        facecolor='white',
        edgecolor='gray',
        boxstyle='round,pad=0.5'
    )
)

# ============================================================
# LEGEND BOX
# ============================================================

legend_text = (
    "WORLD FRAME\n"
    "Black = Fixed Frame\n\n"
    "BODY FRAME\n"
    "Red   = Xb\n"
    "Green = Yb\n"
    "Blue  = Zb"
)

fig.text(
    0.05,
    0.08,
    legend_text,
    fontsize=11,
    bbox=dict(
        facecolor='white',
        edgecolor='black'
    )
)

# ============================================================
# TITLE
# ============================================================

plt.suptitle(
    "World (Fixed) ve Drone (Body) Frame",
    fontsize=20,
    fontweight='bold'
)

plt.tight_layout()
#plt.show()


#####################################################################3

## IMU  KSIMI 

# IMU dosyasını doğru sütun isimleriyle oku
df_imu = pd.read_csv(
    "project_src/imu.txt",
    sep=r"\s+",
    comment="#",
    header=None,
    names=[
        "timestamp", 
        "ang_vel_x", "ang_vel_y", "ang_vel_z", 
        "lin_acc_x", "lin_acc_y", "lin_acc_z"
    ]
)


# 1. Set index and shift to t=0
df_imu.set_index('timestamp', inplace=True)
df_imu.sort_index(inplace=True)
t_start = df_imu.index[0]
df_imu.index = df_imu.index - t_start

# 2. Slice data
relative_end_time = 4878.790299 - t_start ## bu  4886.45  benn belriledim öyle gözümel 
df_subset = df_imu.loc[:relative_end_time]
df_subset[['ang_vel_x', 'ang_vel_y', 'ang_vel_z']] = df_subset[['ang_vel_x', 'ang_vel_y', 'ang_vel_z']] * (180 / np.pi)
 
 
# 1. Jiroskop (Angular Velocity) grafikleri
fig_gyro, axes_gyro = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
gyro_cols = ['ang_vel_x', 'ang_vel_y', 'ang_vel_z']
colors_gyro = ['red', 'green', 'blue']

for i, col in enumerate(gyro_cols):
    axes_gyro[i].plot(df_subset.index, df_subset[col], color=colors_gyro[i], linewidth=0.5)
    axes_gyro[i].set_title(f'Angular Velocity - {col.split("_")[-1].upper()} Axis')
    axes_gyro[i].set_ylabel('deg/s')
    axes_gyro[i].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()

# 2. İvmeölçer (Linear Acceleration) grafikleri
fig_acc, axes_acc = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
acc_cols = ['lin_acc_x', 'lin_acc_y', 'lin_acc_z']
colors_acc = ['orange', 'purple', 'brown']

for i, col in enumerate(acc_cols):
    axes_acc[i].plot(df_subset.index, df_subset[col], color=colors_acc[i], linewidth=0.5)
    axes_acc[i].set_title(f'Linear Acceleration - {col.split("_")[-1].upper()} Axis')
    axes_acc[i].set_ylabel('m/s^2')
    axes_acc[i].grid(True, linestyle='--', alpha=0.7)

axes_acc[2].set_xlabel('Elapsed Time (s)')


plt.tight_layout()


# 1. Bias (Mean) ve Gürültü (Std) hesapla
gyro_bias_df = df_subset[['ang_vel_x', 'ang_vel_y', 'ang_vel_z']].agg(['mean', 'std', 'min', 'max']).transpose()

gyro_x_bias=gyro_bias_df.loc["ang_vel_x", "mean"]
gyro_y_bias=gyro_bias_df.loc["ang_vel_y", "mean"]
gyro_z_bias=gyro_bias_df.loc["ang_vel_z", "mean"]


# 2. Tabloyu düzgün formatta yazdır
print("--- IMU Bias and Noise Statistics (deg/s) ---")
print("gyro_bias:",gyro_bias_df)

# 3. İsteğe bağlı: Bu tabloyu görsel olarak matplotlib ile de gösterebiliriz
fig, ax = plt.subplots(figsize=(8, 2))
ax.axis('off')
table = ax.table(cellText=np.round(gyro_bias_df.values, 4), 
                 rowLabels=gyro_bias_df.index, 
                 colLabels=gyro_bias_df.columns, 
                 loc='center', 
                 cellLoc='center')
table.scale(1, 2)
plt.title("Sensor Bias & Noise Analysis Table")

print("pitch-roll-yaw",pitch,roll,yaw)
#########################################  accelration  düzeltmeleri için de şu yapıalbilir 


# önce bi eksendeki bütü n g leri tutalım elimzide tek tek 

def calculate_real_g(gx=0.078163,gy=-9.27130891,gz=-3.1945492):
    g=math.sqrt(gx**2+gy**2+gz**2)
    return g 

real_g=calculate_real_g()



def get_body_to_world_matrix(roll_deg, pitch_deg, yaw_deg):
    # Dereceyi radyana çevir
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    # Trig değerlerini hesapla
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    # Body -> World Rotasyon Matrisi (Z-Y-X Sıralaması)
    # Bu matris, drone'un sensör verisini dünya eksenine "çevirir"
    R = [
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp,     sr * cp,                cr * cp               ]
    ]
    return np.array(R)

roll = -1.1107403083648595
pitch = 0.2594864661962269
yaw = -129.11373169763877

# Matrisi al
rot_matrix = get_body_to_world_matrix(roll, pitch, yaw)# bodyden direkt worlde aktarıyoz 

trasnferred_to_body=rot_matrix.T@np.array([0,0, real_g])

gx1,gy1,gz1=trasnferred_to_body[0],trasnferred_to_body[1],trasnferred_to_body[2]
modified_df=df_imu.loc[:relative_end_time]
modified_df["lin_acc_x"]=modified_df["lin_acc_x"]-gx1
modified_df["lin_acc_y"]=modified_df["lin_acc_y"]-gy1
modified_df["lin_acc_z"]=modified_df["lin_acc_z"]-gz1
fig_accel, axes_accel = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
# Plot X
axes_accel[0].plot(modified_df.index, modified_df['lin_acc_x'], color='red', linewidth=0.5)
axes_accel[0].set_title('Linear Acceleration X - Gravity Compensated')
axes_accel[0].set_ylabel('m/s^2')
axes_accel[0].grid(True, linestyle='--', alpha=0.7)

# Plot Y
axes_accel[1].plot(modified_df.index, modified_df['lin_acc_y'], color='green', linewidth=0.5)
axes_accel[1].set_title('Linear Acceleration Y - Gravity Compensated')
axes_accel[1].set_ylabel('m/s^2')
axes_accel[1].grid(True, linestyle='--', alpha=0.7)

# Plot Z (Buradaki Z, artık yerçekimi olmadan motor ivmesini göstermeli)
axes_accel[2].plot(modified_df.index, modified_df['lin_acc_z'], color='blue', linewidth=0.5)
axes_accel[2].set_title('Linear Acceleration Z - Gravity Compensated')
axes_accel[2].set_ylabel('m/s^2')
axes_accel[2].set_xlabel('Elapsed Time (s)')
axes_accel[2].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
 # Sadece istatistiksel tabloyu hesapla ve yazdır
accel_stats = modified_df[['lin_acc_x', 'lin_acc_y', 'lin_acc_z']].agg(['mean', 'std', 'min', 'max']).transpose()

print("--- Gravity Compensated Accelerometer Statistics (m/s^2) ---")
print(accel_stats)

accel_x_bias=accel_stats.loc["lin_acc_x", "mean"]
accel_y_bias=accel_stats.loc["lin_acc_y", "mean"]
accel_z_bias=accel_stats.loc["lin_acc_z", "mean"]





##########################################gerçek  verielr burda depoalnsı n 



df_imu_modified = df_imu.copy()
df_imu_modified['lin_acc_x'] = df_imu['lin_acc_x'] - accel_x_bias
df_imu_modified['lin_acc_y'] = df_imu['lin_acc_y'] - accel_y_bias
df_imu_modified['lin_acc_z'] = df_imu['lin_acc_z'] - accel_z_bias


# Bias'ı veriden çıkar
df_imu_modified['ang_vel_x'] -= gyro_x_bias
df_imu_modified['ang_vel_y'] -= gyro_y_bias
df_imu_modified['ang_vel_z'] -= gyro_z_bias


 
# Eğer timestamp yoksa index'i timestamp yap
if 'timestamp' not in df_imu.columns:
    df_imu = df_imu.reset_index().rename(columns={'index': 'timestamp'})

# 1. Kopyala ve temizle
df_state = df_imu.copy()

# 2. İsimleri değiştir
rename_map = {
    'lin_acc_x': 'rx', 
    'lin_acc_y': 'ry', 
    'lin_acc_z': 'rz',
    'ang_vel_x': 'vx', 
    'ang_vel_y': 'vy', 
    'ang_vel_z': 'vz'
}
df_state = df_state.rename(columns=rename_map)

# 3. İstenen tüm kolonları listele (Sıralama tam istediğin gibi)
cols_order = ['timestamp', 'rx', 'ry', 'rz', 'vx', 'vy', 'vz', 'roll', 'pitch', 'yaw']

# 4. Eksik olanları NaN ile doldur ve sadece istediğimiz kolonları seç
target_cols = ['rx', 'ry', 'rz', 'vx', 'vy', 'vz', 'roll', 'pitch', 'yaw']
for col in target_cols:
    df_state[col] = np.nan

df_state = df_state[cols_order]


# 1. İlk satırı (index 0) güncelliyoruz
df_state.loc[0, 'rx'] = 7.60671347395862
df_state.loc[0, 'ry'] = 0.246221449460932
df_state.loc[0, 'rz'] = -0.880823587767568
df_state.loc[0, 'vx'] = 0.0
df_state.loc[0, 'vy'] = 0.0
df_state.loc[0, 'vz'] = 0.0

# 2. Bildiğin başlangıç açılarını giriyoruz
df_state.loc[0, 'roll'] = roll
df_state.loc[0, 'pitch'] = pitch
df_state.loc[0, 'yaw'] = yaw
print((df_state.index))

print(len(df_state.index))


def rotationmat(roll_deg, pitch_deg, yaw_deg):
    # Dereceyi radyana çevir
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    # Trig değerlerini hesapla
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    # Body -> World Rotasyon Matrisi (Z-Y-X Sıralaması)
    # Bu matris, drone'un sensör verisini dünya eksenine "çevirir"
    R = [
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp,     sr * cp,                cr * cp               ]
    ]
    return np.array(R)
dt=0.005

# R_prev başlatma (iloc yerine .loc kullan)
rol_prev = df_state.loc[0, "roll"]
pitch_prev = df_state.loc[0, "pitch"]
yaw_prev = df_state.loc[0, "yaw"]
R_prev = rotationmat(rol_prev, pitch_prev, yaw_prev)
loop_limit = min(len(df_state), len(modified_df))
 
print(len(df_state), len(modified_df))
for i in range(1, len(df_state)+1-45000):
    prev = i - 1
    imu_data = df_imu_modified.iloc[prev]
    wx, wy, wz = imu_data["ang_vel_x"], imu_data["ang_vel_y"], imu_data["ang_vel_z"]
    
    # Doğru phi matrisi
    phi = np.array([[0, -wz, wy],
                    [wz, 0, -wx],
                    [-wy, wx, 0]])
    
    # Matris güncelleme
    R_new = R_prev @ (np.eye(3) + phi * dt)
    
    # ORTOGONALİTE DÜZELTME (Bunu eklemezsen uçuşun 5. saniyesinde açıların saçmalar)
    U, _, Vt = np.linalg.svd(R_new)
    R_new = U @ Vt
    
    # Acceleration world transformation
    spec_force = np.array([imu_data["lin_acc_x"], imu_data["lin_acc_y"], imu_data["lin_acc_z"]])
    acel_world = R_new @ spec_force - np.array([0, 0, real_g])
    
    # Velocity ve Position update
    v_new = df_state.loc[prev, ["vx", "vy", "vz"]].values + acel_world * dt
    r_new = df_state.loc[prev, ["rx", "ry", "rz"]].values + v_new * dt
    print(r_new)
    # Tabloya kaydet
    df_state.loc[i, ["rx", "ry", "rz", "vx", "vy", "vz"]] = np.concatenate([r_new, v_new])
    
    # Euler açılarını hesapla
    r11, r12, r13 = R_new[0,0], R_new[0,1], R_new[0,2]
    r21, r22, r23 = R_new[1,0], R_new[1,1], R_new[1,2]
    r31, r32, r33 = R_new[2,0], R_new[2,1], R_new[2,2]

    df_state.loc[i, 'pitch'] = np.degrees(-np.arcsin(r31))
    df_state.loc[i, 'roll'] = np.degrees(np.arctan2(r32, r33))
    df_state.loc[i, 'yaw'] = np.degrees(np.arctan2(r21, r11))
    
    R_prev = R_new
# en sonun da tekrardan kıyaslama ypamak  için bi  path çizdirmek laızm 

df_gt = pd.read_csv(
    "project_src/groundtruth.txt",
    sep=r"\s+",
    comment="#",
    header=None,
    names=[
        "timestamp",
        "tx","ty","tz",
        "qx","qy","qz","qw"
    ]
)

fig_comp = plt.figure(figsize=(12, 9))
ax_comp = fig_comp.add_subplot(111, projection='3d')

# 1. Senin INS algoritmanla hesapladığın yörünge (Tahmin)
ax_comp.plot(df_state['rx'], df_state['ry'], df_state['rz'], 
        label='INS Algoritma (Tahmin)', color='red', alpha=0.7, linewidth=1.5)

# 2. Ground Truth (Gerçek değerler)
ax_comp.plot(df_gt['tx'], df_gt['ty'], df_gt['tz'], 
        label='Ground Truth (Gerçek)', color='green', linestyle='--', linewidth=2)

# Görselleştirmeyi güzelleştir
ax_comp.set_xlabel('X (m)')
ax_comp.set_ylabel('Y (m)')
ax_comp.set_zlabel('Z (m)')
ax_comp.set_title('INS Tahmin vs. Ground Truth Karşılaştırması')
ax_comp.legend()
ax_comp.grid(True)

# Başlangıç noktasını belirginleştir
ax_comp.scatter(0, 0, 0, color='black', s=50, label='Başlangıç (0,0,0)')

plt.show()
#plt.show()
