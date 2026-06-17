import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# 1. Veriyi hazırla
df = pd.read_csv("./project_src/imu.txt", sep='\s+')
df['dt'] = df['timestamp'].diff().fillna(0.002)

# Global pozisyon ve hız tutucuları
pos = np.array([0.0, 0.0, 0.0])
vel = np.array([0.0, 0.0, 0.0])
# Yönelimi tutmak için (Identity quaternion)
quat = R.from_euler('xyz', [0, 0, 0])

trajectory = []

# 2. İteratif Dead Reckoning
for i in range(len(df)):
    dt = df.loc[i, 'dt']
    
    # Gyro verisinden rotasyon artışı
    gyro_data = df.loc[i, ['ang_vel_x', 'ang_vel_y', 'ang_vel_z']].values
    rot_step = R.from_rotvec(gyro_data * dt)
    quat = quat * rot_step # Güncel yönelim
    
    # İvmeyi Global'e çevir
    acc_body = df.loc[i, ['lin_acc_x', 'lin_acc_y', 'lin_acc_z']].values
    # Yerçekimini çıkar (basitleştirilmiş)
    acc_body[2] -= 9.81
    
    acc_global = quat.apply(acc_body)
    
    # Hız ve Pozisyon güncelle
    vel += acc_global * dt
    pos += vel * dt
    
    trajectory.append(pos.copy())

# 3. Sonuçları Çizdir
trajectory = np.array(trajectory)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Global Trajectory')
ax.set_title('Dinamik Dönüşüm ile 3D Yörünge')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.legend()
plt.show()



 

# 1. Ground Truth verisini yükle (Senin verdiğin yeni format)
gt_cols = ['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
gt_df = pd.read_csv("./project_src/groundtruth.txt", sep='\s+', names=gt_cols)

# 2. Zaman hizalama (İki dataframe'i en yakın timestamp'e göre eşleştir)
# IMU verinin 'result' dataframe'i olduğunu varsayıyorum
merged_df = pd.merge_asof(result.sort_values('timestamp'), 
                          gt_df.sort_values('timestamp'), 
                          on='timestamp', direction='nearest')

# 3. Hata (Error) hesapla
merged_df['err_x'] = merged_df['pos_x'] - merged_df['tx']
merged_df['err_y'] = merged_df['pos_y'] - merged_df['ty']
merged_df['err_z'] = merged_df['pos_z'] - merged_df['tz']

# 4. Toplam Hata (RMSE - Root Mean Square Error)
merged_df['euclidean_err'] = np.sqrt(merged_df['err_x']**2 + merged_df['err_y']**2 + merged_df['err_z']**2)

print(f"Ortalama Hata: {merged_df['euclidean_err'].mean():.4f} metre")