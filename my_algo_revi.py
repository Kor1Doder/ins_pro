import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math
from yedekler_burda.bias_std_est import *
from yedekler_burda.find_bias_sigma import estimate_imu_bias
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
 ###görselleştiremde görmek istediklerimi buradn ayrladım 
imu_gyro_dem=False
imu_accel_dem=False
imu_gyro_istatistik=False
imu_accel_istatistik=False


GT_FILE  = 'project_src/groundtruth.txt'
IMU_FILE = 'project_src/imu.txt'

# IMU sensor biases (determined from static segment calibration)
GYRO_BIAS = 0.0274094406657   # rad/s (constant over time)
ACCEL_BIAS = 0.145200258157    # m/s^2 (constant over time)


# Gravitational acceleration (NE frame: Down is positive)
G = -9.81  # m/s^2
g_n = np.array([0.0, 0.0, G])



def Transition(phi, theta):
    """
    Euler angle kinematic transformation matrix.---> bodyden aldığımız rateleri  yer sistemine aktracak fonk 
    
    Relates Euler angle rates to body-frame angular velocity:
        [φ_dot, θ_dot, ψ_dot]^T = E(φ, θ) ω_b
    
    Where:
        φ = roll  (rotation about X-axis)
        θ = pitch (rotation about Y-axis)
        ψ = yaw   (rotation about Z-axis)
    
    Note: Singular at θ = ±90° (gimbal lock), not reached in nominal flight.
    """
    tan_theta = np.tan(theta)
    sec_theta = 1.0 / np.cos(theta)
    
    return np.array([
        [1.0, np.sin(phi) * tan_theta, np.cos(phi) * tan_theta],
        [0.0, np.cos(phi),             -np.sin(phi)],
        [0.0, np.sin(phi) * sec_theta, np.cos(phi) * sec_theta],
    ])

def rot_matrix(roll_deg, pitch_deg, yaw_deg):
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cp*cy,  sr*sp*cy - cr*sy,  cr*sp*cy + sr*sy],
        [cp*sy,  sr*sp*sy + cr*cy,  cr*sp*sy - sr*cy],
        [-sp,    sr*cp,             cr*cp            ]
    ])
print("\n" + "=" * 80)
print("IMU-ONLY STRAPDOWN DEAD RECKONING")
print("=" * 80 + "\n")



# Load ground truth (NEU position, quaternion)
df_gt = pd.read_csv(
    GT_FILE,
    sep=r'\s+',
    comment='#',
    header=None,
    names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
)


print("\n" + "=" * 80)
print("IMU-ONLY STRAPDOWN DEAD RECKONING")
print("=" * 80 + "\n")

# Load ground truth (NED position, quaternion)
df_gt = pd.read_csv(
    GT_FILE,
    sep=r'\s+',
    comment='#',
    header=None,
    names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
)
df_gt = df_gt.sort_values('timestamp').reset_index(drop=True)

# Load IMU (angular velocity, specific force)
df_imu_raw = pd.read_csv(
    IMU_FILE,
    sep=r'\s+',
    comment='#',
    header=None,
    names=['timestamp', 'wx', 'wy', 'wz', 'ax', 'ay', 'az']
)
df_imu_raw = df_imu_raw.sort_values('timestamp').reset_index(drop=True)

print(f"Ground truth   : {df_gt['timestamp'].iloc[0]:.6f} -> {df_gt['timestamp'].iloc[-1]:.6f} s")
print(f"               ({len(df_gt)} samples)")
print(f"IMU (raw)      : {df_imu_raw['timestamp'].iloc[0]:.6f} -> {df_imu_raw['timestamp'].iloc[-1]:.6f} s")
print(f"               ({len(df_imu_raw)} samples)")



def plot_accel_3x1_accel(df):
    limit_time = 4961.0
    df_imu_limited = df[df['timestamp'] <= limit_time]
    # Zamanı sıfırlama (Time Normalization)
    df_imu_limited['timestamp'] = df_imu_limited['timestamp'] - df_imu_limited['timestamp'].iloc[0]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(df_imu_limited['timestamp'], df_imu_limited['ax'], 'r', label='ax')
    axes[1].plot(df_imu_limited['timestamp'], df_imu_limited['ay'], 'g', label='ay')
    axes[2].plot(df_imu_limited['timestamp'], df_imu_limited['az'], label='az')
    

    # --- İSTATİSTİKSEL ÖZET (TABLO) ---
    # Sadece accel sütunlarını al
    stats = df_imu_limited[['ax', 'ay', 'az']].agg(['mean', 'std'])
    print("\n" + "="*40)
    print(f"Acceleration İSTATİSTİKLERİ (0 - {df_imu_limited['timestamp'].iloc[-1]:.1f}s)")    
    print("="*40)
    print(stats.to_string())
    print("="*40 + "\n")

    for i in range(3):
        axes[i].set_ylabel('m/s^2')
        axes[i].grid(True)
        axes[i].legend(loc='upper right')
    
    axes[2].set_xlabel('Time (s)')
    plt.suptitle('IMU Accelerometer  VALUES')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_accel_3x1_gyro(df):
    limit_time = 4961.0
    df_imu_limited = df[df['timestamp'] <= limit_time]
    # Zamanı sıfırlama (Time Normalization)
    df_imu_limited['timestamp'] = df_imu_limited['timestamp'] - df_imu_limited['timestamp'].iloc[0]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        # --- İSTATİSTİKSEL ÖZET (TABLO) ---
    # Sadece gyro sütunlarını al
    stats = df_imu_limited[['wx', 'wy', 'wz']].agg(['mean', 'std'])
    print("\n" + "="*40)
    print(f"GYRO İSTATİSTİKLERİ (0 - {df_imu_limited['timestamp'].iloc[-1]:.1f}s)")    
    print("="*40)
    print(stats.to_string())
    print("="*40 + "\n")

    axes[0].plot(df_imu_limited['timestamp'], df_imu_limited['wx'], 'r', label='wx')
    axes[1].plot(df_imu_limited['timestamp'], df_imu_limited['wy'], 'g', label='wy')
    axes[2].plot(df_imu_limited['timestamp'], df_imu_limited['wz'], label='wz')
    
    for i in range(3):
        axes[i].set_ylabel('rad/s')
        axes[i].grid(True)
        axes[i].legend(loc='upper right')
    
    axes[2].set_xlabel('Time (s)')
    plt.suptitle('IMU Gyro Values ')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
# 4961. saniyeye kadar olan veriyi se
########################IMU RAW DATANIN
if imu_accel_dem:
    plot_accel_3x1_accel(df_imu_raw)
if imu_gyro_dem:
    plot_accel_3x1_gyro(df_imu_raw)

####İLK BAŞT AÇALIŞIRKEN NASIL Bİ ORTALAMAYLA ÇALŞYOR ONA BKAALIM 

## burda şu  var ilk 8 saniyede görebiliyoruz adam akıllı sonrası bozuluyor motor çalşamya başlıyor sanırı m odnan 

def mean_accel_3x1_gyro(df):
    limit_time = df['timestamp'][0]+8.2
    df_imu_limited = df[df['timestamp'] <= limit_time]
    # Zamanı sıfırlama (Time Normalization)
    df_imu_limited['timestamp'] = df_imu_limited['timestamp'] - df_imu_limited['timestamp'].iloc[0]
     
    stats = df_imu_limited[['wx', 'wy', 'wz']].agg(['mean', 'std'])
    print("\n" + "="*40)
    print(f"GYRO İSTATİSTİKLERİ (0 - {df_imu_limited['timestamp'].iloc[-1]:.1f}s)")    
    print("="*40)
    print(stats.to_string())
    print("="*40 + "\n")
def mean_accel_3x1_gyro(df):
    limit_time = df['timestamp'][0]+8.2
    df_imu_limited = df[df['timestamp'] <= limit_time]
    # Zamanı sıfırlama (Time Normalization)
    df_imu_limited['timestamp'] = df_imu_limited['timestamp'] - df_imu_limited['timestamp'].iloc[0]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        # --- İSTATİSTİKSEL ÖZET (TABLO) ---
    # Sadece gyro sütunlarını al
    stats = df_imu_limited[['wx', 'wy', 'wz']].agg(['mean', 'std'])
    print("\n" + "="*40)
    print(f"GYRO İSTATİSTİKLERİ (0 - {df_imu_limited['timestamp'].iloc[-1]:.1f}s)")    
    print("="*40)
    print(stats.to_string())
    print("="*40 + "\n")

if imu_gyro_istatistik:
    analyze_gyro(df_imu_raw,duration=8.2)# bu durationı grafikten bakarak belirledim şimdilik kalsın  ya 
if imu_accel_istatistik:
    analyze_accelerometer(df_imu_raw,duration=8.2)




 


p0 = df_gt.iloc[0][['tx', 'ty', 'tz']].to_numpy().astype(float)
q0   = df_gt.iloc[0][["qx", "qy", "qz", "qw"]].values
rot0 = R.from_quat(q0)
roll_init, pitch_init, yaw_init = rot0.as_euler('xyz', degrees=True)
Rmat_init = rot0.as_matrix()
print(f"\nInitial orientation : roll={roll_init:.4f}  pitch={pitch_init:.4f}  yaw={yaw_init:.4f}  (deg)")
v0 = np.array([0 ,0 ,0])

# Initial attitude (from GT quaternion at t=0)
# Fonksiyon artık (roll, pitch, yaw) döndürüyor
print(f"\nInitial state (t=0):")
print(f"  Position     : p_n = {p0} m ()")
print(f"  Velocity     : v_n = {v0} m/s ()")
print(f"  Attitude     : roll={roll_init:.3f}°, pitch={pitch_init:.3f}°, yaw={yaw_init:.3f}°")
print(f"  Gravity      : g_n = {g_n} m/s^2 ")


 

 

# ── Gravity (world frame, Z yukarı) ─────────────────────────────────────────
real_g  = 9.80655
g_world = np.array([0.0, 0.0, real_g])

# ── Zaman aralığı (senin kodundan) ──────────────────────────────────────────
t_start = 4908.791704125
t_end   = 4958.289704125

df_imu_filtered = df_imu_raw[
    (df_imu_raw['timestamp'] >= t_start) &
    (df_imu_raw['timestamp'] <= t_end)
].copy().reset_index(drop=True)

n = len(df_imu_filtered)
p     = np.zeros((n, 3))
v     = np.zeros((n, 3))
euler = np.zeros((n, 3))

# ── Başlangıç orientasyonu (senin kodundan) ──────────────────────────────────
q0    = df_gt.iloc[0][["qx", "qy", "qz", "qw"]].values
rot0  = R.from_quat(q0)
roll_init, pitch_init, yaw_init = rot0.as_euler('xyz', degrees=True)
Rbn   = rot0.as_matrix()

p[0]     = p0
v[0]     = v0
euler[0] = [roll_init, pitch_init, yaw_init]   # derece, sadece kayıt

# ── Bias (senin estimate_imu_bias sonuçların) ────────────────────────────────
bias_noise = estimate_imu_bias(df_imu_raw, q0[0], q0[1], q0[2], q0[3], False)

ba = np.array([bias_noise["acc_bias"][0],
               bias_noise["acc_bias"][1],
               bias_noise["acc_bias"][2]])
bg = np.array([bias_noise["gyro_bias"][0],
               bias_noise["gyro_bias"][1],
               bias_noise["gyro_bias"][2]])

# Gravity'yi body frame'e çevir, accel bias'ı düzelt
g_body0    = Rbn.T @ g_world          # gravity → body frame
accel_mean = df_imu_raw[df_imu_raw['timestamp'] < 4887][['ax','ay','az']].mean().values
print('acccelmea',accel_mean)
accel_bias = accel_mean - g_body0     # doğru bias (gravity dahil)

print(f"Accel bias (düzeltilmiş) : {accel_bias}")
print(f"Gyro  bias               : {bg}")

# ── Bias çıkar (sadece sensor bias, gravity YOK) ─────────────────────────────
df_imu_filtered['wx'] -= bg[0]
df_imu_filtered['wy'] -= bg[1]
df_imu_filtered['wz'] -= bg[2]
df_imu_filtered['ax'] -= accel_bias[0]
df_imu_filtered['ay'] -= accel_bias[1]
df_imu_filtered['az'] -= accel_bias[2]

print(f"az ortalama (bias sonrası, ≈0 olmalı): {df_imu_filtered['az'].mean():.4f}")

# ── Numpy'a çevir ────────────────────────────────────────────────────────────
imu_arr   = df_imu_filtered[['wx','wy','wz','ax','ay','az']].values
t_imu     = df_imu_filtered['timestamp'].values
dt_arr    = np.diff(t_imu)
dt_median = np.median(dt_arr[dt_arr > 0])

# ── Propagasyon ──────────────────────────────────────────────────────────────
for k in range(n - 1):
    dt = dt_arr[k]
    if dt <= 0 or dt > 0.1:
        dt = dt_median

    omega = imu_arr[k, 0:3]    # rad/s  (bias çıkarılmış)
    f_hat = imu_arr[k, 3:6]    # m/s²  (bias çıkarılmış, gravity YOK)

    # 1) Rotasyon matrisini güncelle
    wx_, wy_, wz_ = omega
    skew = np.array([[ 0,   -wz_,  wy_],
                     [ wz_,  0,   -wx_],
                     [-wy_,  wx_,   0  ]])
    R_new = Rbn @ (np.eye(3) + skew * dt)

    # SVD ile yeniden ortogonalize et (birikim hatasını önler)
    U, _, Vt = np.linalg.svd(R_new)
    Rbn = U @ Vt

    # 2) Euler açılarını güncelle (Rbn'den)
    euler[k+1, 1] = np.degrees(-np.arcsin(np.clip(Rbn[2, 0], -1.0, 1.0)))
    euler[k+1, 0] = np.degrees(np.arctan2(Rbn[2, 1], Rbn[2, 2]))
    euler[k+1, 2] = np.degrees(np.arctan2(Rbn[1, 0], Rbn[0, 0]))

    # 3) Force dönüşümü — Euler güncellendikten SONRA, güncel Rbn ile
    accel_world = Rbn @ f_hat - g_world

    # 4) Hız ve konum
    v[k+1] = v[k] + dt * accel_world
    p[k+1] = p[k] + dt * v[k]






visual_dead=False
if visual_dead==True:
    ################################333333
    # Statik segmentin başındaki GT quaternion'unu al
    # (uçuş başlangıcındaki q0 değil, drone yerdeyken)
    t_static_end = t_start  # GT başlamadan öncesi

    # Statik segmentte drone yerdeyken GT'nin ilk noktası
    # Ama GT statik segmenti kapsamıyor olabilir — o zaman IMU verisinden hesapla
    df_static = df_imu_raw[df_imu_raw['timestamp'] < t_start].copy()
    print(f"Statik segment: {len(df_static)} örnek")
    print(f"Statik ax/ay/az ort: {df_static[['ax','ay','az']].mean().values}")

    # Drone statik ve YATAYsa: gravity tamamen body Z'ye düşmeli
    # ax_static ≈ 0, ay_static ≈ 0, az_static ≈ +9.81
    # Eğer öyleyse bias çok basit:
    accel_bias_simple = df_static[['ax','ay','az']].mean().values - np.array([0, 0, 9.80655])
    gyro_bias_simple  = df_static[['wx','wy','wz']].mean().values

    print(f"\naccel_bias (yatay varsayım) : {accel_bias_simple}")
    print(f"gyro_bias                   : {gyro_bias_simple}")
    print(f"\nKontrol — statik ax ort: {df_static['ax'].mean():.4f}  (≈0 olmalı)")
    print(f"Kontrol — statik ay ort: {df_static['ay'].mean():.4f}  (≈0 olmalı)")
    print(f"Kontrol — statik az ort: {df_static['az'].mean():.4f}  (≈9.81 olmalı)")
    # ── INTERPOLATE GT onto IMU timestamps ──────────────────────────────────────
    t_gt  = df_gt['timestamp'].values
    t_ins = t_imu  # IMU zaman damgaları
    def interp_gt(col):
        f = interp1d(t_gt, df_gt[col].values, kind='linear',
                    bounds_error=False, fill_value='extrapolate')
        return f(t_ins)

    gt_x = interp_gt('tx')
    gt_y = interp_gt('ty')
    gt_z = interp_gt('tz')

    # GT quaternion → euler
    gt_qx = interp_gt('qx'); gt_qy = interp_gt('qy')
    gt_qz = interp_gt('qz'); gt_qw = interp_gt('qw')

    gt_roll  = np.zeros(n)
    gt_pitch = np.zeros(n)
    gt_yaw   = np.zeros(n)
    for i in range(n):
        q = np.array([gt_qx[i], gt_qy[i], gt_qz[i], gt_qw[i]])
        q /= np.linalg.norm(q)
        rr, pp, yy = R.from_quat(q).as_euler('xyz', degrees=True)
        gt_roll[i] = rr; gt_pitch[i] = pp; gt_yaw[i] = yy

    t_rel = t_ins - t_ins[0]   # sıfırdan başlayan zaman ekseni

    # Hatalar
    err_x  = p[:, 0] - gt_x
    err_y  = p[:, 1] - gt_y
    err_z  = p[:, 2] - gt_z
    err_3d = np.sqrt(err_x**2 + err_y**2 + err_z**2)

    print(f"Final 3D error : {err_3d[-1]:.3f} m")
    print(f"Mean  3D error : {err_3d.mean():.3f} m")

    # ═══════════════════════════════════════════════════════════════════
    # PLOT 1 — 3D Trajectory
    # ═══════════════════════════════════════════════════════════════════
    fig3d = plt.figure(figsize=(12, 9))
    ax3d  = fig3d.add_subplot(111, projection='3d')
    ax3d.plot(p[:, 0], p[:, 1], p[:, 2],
            color='red', linewidth=1.0, alpha=0.85, label='INS Estimate')
    ax3d.plot(gt_x, gt_y, gt_z,
            color='green', linewidth=2.0, linestyle='--', label='Ground Truth')
    ax3d.scatter(*p[0],  color='black', s=80, zorder=5, label='Start')
    ax3d.scatter(*p[-1], color='red',   s=80, zorder=5, label='End (INS)')
    ax3d.scatter(gt_x[-1], gt_y[-1], gt_z[-1],
                color='green', s=80, zorder=5, label='End (GT)')
    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
    ax3d.set_title('3D Trajectory — INS vs Ground Truth')
    ax3d.legend(); ax3d.grid(True)
    plt.tight_layout()

    # ═══════════════════════════════════════════════════════════════════
    # PLOT 2 — Position X / Y / Z vs time
    # ═══════════════════════════════════════════════════════════════════
    fig_pos, axes_pos = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    labels = ['X (m)', 'Y (m)', 'Z (m)']
    ins_ch = [p[:, 0], p[:, 1], p[:, 2]]
    gt_ch  = [gt_x,    gt_y,    gt_z   ]
    colors = ['red', 'green', 'blue']

    for i, (ax, lbl, ins, gt, col) in enumerate(
            zip(axes_pos, labels, ins_ch, gt_ch, colors)):
        ax.plot(t_rel, ins, color=col,   linewidth=0.8, label=f'INS {lbl[0]}')
        ax.plot(t_rel, gt,  color='black', linewidth=1.5,
                linestyle='--', alpha=0.7, label=f'GT {lbl[0]}')
        ax.set_ylabel(lbl); ax.legend(loc='upper left'); ax.grid(True, alpha=0.4)

    axes_pos[0].set_title('Position Comparison — INS vs Ground Truth')
    axes_pos[2].set_xlabel('Elapsed Time (s)')
    plt.tight_layout()

    # ═══════════════════════════════════════════════════════════════════
    # PLOT 3 — Position Error per axis + 3D
    # ═══════════════════════════════════════════════════════════════════
    fig_err, axes_err = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    err_data   = [err_x, err_y, err_z, err_3d]
    err_labels = ['Error X (m)', 'Error Y (m)', 'Error Z (m)', '3D Error (m)']
    err_colors = ['red', 'green', 'blue', 'black']

    for ax, data, lbl, col in zip(axes_err, err_data, err_labels, err_colors):
        ax.plot(t_rel, data, color=col, linewidth=0.8)
        ax.set_ylabel(lbl); ax.grid(True, alpha=0.4)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    axes_err[0].set_title('Position Error — INS vs Ground Truth')
    axes_err[3].set_xlabel('Elapsed Time (s)')

    # RMSE kutucukları
    for ax, data in zip(axes_err[:3], [err_x, err_y, err_z]):
        rmse = np.sqrt(np.mean(data**2))
        ax.text(0.99, 0.95, f'RMSE={rmse:.3f} m',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, bbox=dict(fc='white', ec='gray', alpha=0.8))
    ax_3d_err = axes_err[3]
    ax_3d_err.text(0.99, 0.95,
                f'Final={err_3d[-1]:.3f} m  Mean={err_3d.mean():.3f} m',
                transform=ax_3d_err.transAxes, ha='right', va='top',
                fontsize=9, bbox=dict(fc='white', ec='gray', alpha=0.8))
    plt.tight_layout()

    # ═══════════════════════════════════════════════════════════════════
    # PLOT 4 — Euler Angles vs GT
    # ═══════════════════════════════════════════════════════════════════
    fig_eul, axes_eul = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    eul_ins    = [euler[:, 0], euler[:, 1], euler[:, 2]]
    eul_gt     = [gt_roll,     gt_pitch,    gt_yaw     ]
    eul_labels = ['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']
    eul_colors = ['red', 'green', 'blue']

    for ax, ins, gt, lbl, col in zip(axes_eul, eul_ins, eul_gt, eul_labels, eul_colors):
        ax.plot(t_rel, ins, color=col,     linewidth=0.8, label=f'INS {lbl.split()[0]}')
        ax.plot(t_rel, gt,  color='black', linewidth=1.5,
                linestyle='--', alpha=0.7, label=f'GT {lbl.split()[0]}')
        ax.set_ylabel(lbl); ax.legend(loc='upper left'); ax.grid(True, alpha=0.4)

    axes_eul[0].set_title('Euler Angles — INS vs Ground Truth')
    axes_eul[2].set_xlabel('Elapsed Time (s)')
    plt.tight_layout()

    # ═══════════════════════════════════════════════════════════════════
    # PLOT 5 — Velocity (INS only, GT yok)
    # ═══════════════════════════════════════════════════════════════════
    fig_vel, axes_vel = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    vel_labels = ['Vx (m/s)', 'Vy (m/s)', 'Vz (m/s)']
    for i, (ax, lbl, col) in enumerate(zip(axes_vel, vel_labels, ['red','green','blue'])):
        ax.plot(t_rel, v[:, i], color=col, linewidth=0.8, label=lbl)
        ax.set_ylabel(lbl); ax.legend(loc='upper left'); ax.grid(True, alpha=0.4)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    axes_vel[0].set_title('INS Velocity (World Frame)')
    axes_vel[2].set_xlabel('Elapsed Time (s)')
    plt.tight_layout()

    # ═══════════════════════════════════════════════════════════════════
    # PLOT 6 — 2D Top-down (X-Y)
    # ═══════════════════════════════════════════════════════════════════
    fig_2d, ax_2d = plt.subplots(figsize=(10, 8))
    ax_2d.plot(p[:, 0], p[:, 1], color='red',   linewidth=1.0, label='INS')
    ax_2d.plot(gt_x,    gt_y,    color='green', linewidth=2.0,
            linestyle='--', label='Ground Truth')
    ax_2d.scatter(p[0, 0],  p[0, 1],  color='black', s=80, zorder=5, label='Start')
    ax_2d.scatter(p[-1, 0], p[-1, 1], color='red',   s=80, zorder=5, label='End INS')
    ax_2d.scatter(gt_x[-1], gt_y[-1], color='green', s=80, zorder=5, label='End GT')
    ax_2d.set_xlabel('X (m)'); ax_2d.set_ylabel('Y (m)')
    ax_2d.set_title('Top-down Trajectory (X-Y Plane)')
    ax_2d.legend(); ax_2d.grid(True, alpha=0.4); ax_2d.set_aspect('equal')
    plt.tight_layout()

    plt.show()





# ══════════════════════════════════════════════════════════════════
# GPS verisi yükle
# ══════════════════════════════════════════════════════════════════

df_gps = pd.read_csv(
    'project_src/gps_loosely.txt',
    sep=r'\s+',
    comment='#',
    header=0
)
df_gps = df_gps.sort_values('timestamp').reset_index(drop=True)

df_gps['lat'] = df_gps['lat'].astype(float)
df_gps['lon'] = df_gps['lon'].astype(float)
df_gps['alt'] = df_gps['alt'].astype(float)

REF_LAT_DEG = 39.925018
REF_LON_DEG = 32.836956
REF_ALT     = 850.0

lat0 = np.radians(REF_LAT_DEG)
lon0 = np.radians(REF_LON_DEG)
alt0 = REF_ALT

R_earth = 6378137.0

def lla_to_neu(lat_deg, lon_deg, alt):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    dlat = lat - lat0
    dlon = lon - lon0
    x = R_earth * dlat
    y = R_earth * dlon * np.cos(lat0)
    z = alt - alt0
    return x, y, z

gps_x, gps_y, gps_z = lla_to_neu(
    df_gps['lat'].values,
    df_gps['lon'].values,
    df_gps['alt'].values
)

gps_vx = df_gps['vn'].values
gps_vy = df_gps['ve'].values
gps_vz = df_gps['vu'].values

gps_pos_std_h = df_gps["pos_std_h"].iloc[0]
gps_pos_std_v = df_gps["pos_std_v"].iloc[0]
gps_vel_std   = df_gps["vel_std"].iloc[0]

gps_timestamps = df_gps['timestamp'].values

print(f"GPS pos_std_h={gps_pos_std_h} m  pos_std_v={gps_pos_std_v} m  vel_std={gps_vel_std} m/s")

# ══════════════════════════════════════════════════════════════════
# 15-STATE EKF TANIMLAR
# ══════════════════════════════════════════════════════════════════
#
# State vektörü (15 x 1):
#   x[0:3]   = [pN, pE, pU]        pozisyon (m)
#   x[3:6]   = [vN, vE, vU]        hız (m/s)
#   x[6:9]   = [roll, pitch, yaw]  attitude (rad)
#   x[9:12]  = [bax, bay, baz]     accel bias (m/s²)   ← YENİ
#   x[12:15] = [bgx, bgy, bgz]     gyro  bias (rad/s)  ← YENİ
#
# Bias modeli: random walk
#   b_dot = w_b,   w_b ~ N(0, sigma_rw²)
#
# GPS ölçüm: 6-state (pos + vel), yaw gözlemlenemiyor

n_state = 15
n_meas  = 6

# ── IMU noise ────────────────────────────────────────────────────
sigma_ax = 0.02735837   # m/s²   accel white noise std
sigma_ay = 0.02330523
sigma_az = 0.05322704
sigma_wx = 0.00277449   # rad/s  gyro white noise std
sigma_wy = 0.00310758
sigma_wz = 0.00185727

# ── Bias random walk std (verilen değerler) ───────────────────────
# Gyro  random walk: 4e-5  rad/s/sqrt(s)
# Accel random walk: 0.002 m/s²/sqrt(s)
sigma_rw_gyro  = 4e-5    # rad/s / sqrt(s)
sigma_rw_accel = 0.002   # m/s²  / sqrt(s)

# ── R matrisi (6 x 6) ─────────────────────────────────────────────
R_meas = np.diag([
    gps_pos_std_h**2,
    gps_pos_std_h**2,
    gps_pos_std_v**2,
    gps_vel_std**2,
    gps_vel_std**2,
    gps_vel_std**2
])

# ── H matrisi (6 x 15) ───────────────────────────────────────────
# Sadece pos ve vel state'lerini ölçüyoruz
H = np.zeros((n_meas, n_state))
H[0, 0] = 1.0   # pN
H[1, 1] = 1.0   # pE
H[2, 2] = 1.0   # pU
H[3, 3] = 1.0   # vN
H[4, 4] = 1.0   # vE
H[5, 5] = 1.0   # vU

# ══════════════════════════════════════════════════════════════════
# EKF BAŞLANGIÇ
# ══════════════════════════════════════════════════════════════════

x_ekf = np.zeros(n_state)

# Pozisyon: GT başlangıcından
x_ekf[0:3] = np.array([
    np.interp(t_start, df_gt['timestamp'].values, df_gt['tx'].values.astype(float)),
    np.interp(t_start, df_gt['timestamp'].values, df_gt['ty'].values.astype(float)),
    np.interp(t_start, df_gt['timestamp'].values, df_gt['tz'].values.astype(float))
])
# Hız: sıfır başlangıç
x_ekf[3:6] = v0

# Attitude: GT quaternion'dan
x_ekf[6]  = np.radians(roll_init)
x_ekf[7]  = np.radians(pitch_init)
x_ekf[8]  = np.radians(yaw_init)

# Bias başlangıcı: bias_noise'dan gelen tahminleri kullan
# accel_bias ve bg zaten ana kodda hesaplandı
x_ekf[9]  = accel_bias[0]
x_ekf[10] = accel_bias[1]
x_ekf[11] = accel_bias[2]
x_ekf[12] = bg[0]
x_ekf[13] = bg[1]
x_ekf[14] = bg[2]

# ── P başlangıcı ─────────────────────────────────────────────────
P = np.diag([
    gps_pos_std_h**2,       # pN
    gps_pos_std_h**2,       # pE
    gps_pos_std_v**2,       # pU

    gps_vel_std**2,         # vN
    gps_vel_std**2,         # vE
    gps_vel_std**2,         # vU

    np.radians(10)**2,      # roll
    np.radians(10)**2,      # pitch

    # Yaw GPS tarafından direkt ölçülmüyor.
    # 2 deg çok küçük kalıyor. Bu yüzden 20-30 deg daha mantıklı.
    np.radians(25)**2,      # yaw

    # Bias başlangıç belirsizliği random-walk ile yazılmaz.
    # Random-walk Q içindir.
    # Burada bias tahminine ne kadar güvendiğini yazıyoruz.
    sigma_ax**2,            # bax
    sigma_ay**2,            # bay
    sigma_az**2,            # baz

    sigma_wx**2,            # bgx
    sigma_wy**2,            # bgy
    sigma_wz**2,            # bgz
])

Rbn_ekf = rot0.as_matrix().copy()

# Kayıt dizileri
ekf_p      = np.zeros((n, 3))
ekf_v      = np.zeros((n, 3))
ekf_euler  = np.zeros((n, 3))
ekf_ba     = np.zeros((n, 3))   # accel bias tahmini
ekf_bg     = np.zeros((n, 3))   # gyro  bias tahmini

ekf_p[0]     = x_ekf[0:3]
ekf_v[0]     = x_ekf[3:6]
ekf_euler[0] = [roll_init, pitch_init, yaw_init]
ekf_ba[0]    = x_ekf[9:12]
ekf_bg[0]    = x_ekf[12:15]

P_hist    = np.zeros((n, n_state, n_state))
P_hist[0] = P.copy()

real_g  = 9.80655
g_world = np.array([0.0, 0.0, real_g])

gps_idx     = 0
gps_half_dt = 0.5 * np.median(np.diff(gps_timestamps))

# ══════════════════════════════════════════════════════════════════
# EKF DÖNGÜSÜ
# ══════════════════════════════════════════════════════════════════
for k in range(n - 1):

    dt = dt_arr[k]
    if dt <= 0 or dt > 0.1:
        dt = dt_median

    # ── Ham IMU oku (bias ÇIKARILMAMIŞ — EKF kendi bias'ını tahmin ediyor) ──
    # imu_arr bias çıkarılmış halde geliyor ana koddan;
    # ama EKF bias state'i taşıdığı için ham sensör verisini kullanmak
    # daha doğru olur. Ancak ana kodda ham veri artık mevcut değil,
    # o yüzden imu_arr üzerine mevcut bias tahminini GERİ EKLIYORUZ
    # sonra EKF'nin kendi bias tahminini çıkarıyoruz.
    omega_raw = imu_arr[k, 0:3] + bg           # ham gyro  (ana bias geri eklendi)
    f_raw     = imu_arr[k, 3:6] + accel_bias   # ham accel (ana bias geri eklendi)

    # EKF'nin kendi bias tahminiyle düzelt
    ba_k = x_ekf[9:12]
    bg_k = x_ekf[12:15]

    omega = omega_raw - bg_k   # bias-corrected gyro
    f_hat = f_raw     - ba_k   # bias-corrected accel

    # ── 1) Attitude predict ──────────────────────────────────────
    wx_, wy_, wz_ = omega
    skew = np.array([
        [ 0,    -wz_,  wy_],
        [ wz_,   0,   -wx_],
        [-wy_,  wx_,   0  ]
    ])
    R_new = Rbn_ekf @ (np.eye(3) + skew * dt)
    U, _, Vt = np.linalg.svd(R_new)
    Rbn_ekf = U @ Vt

    roll_k  = np.arctan2(Rbn_ekf[2, 1], Rbn_ekf[2, 2])
    pitch_k = -np.arcsin(np.clip(Rbn_ekf[2, 0], -1.0, 1.0))
    yaw_k   = np.arctan2(Rbn_ekf[1, 0], Rbn_ekf[0, 0])
 
    # ── 2) Acceleration → world frame ───────────────────────────
    accel_world = Rbn_ekf @ f_hat - g_world

    # ── 3) State predict ─────────────────────────────────────────
    x_pred = x_ekf.copy()
    x_pred[0:3]  = x_ekf[0:3] + dt * x_ekf[3:6]   # pos
    x_pred[3:6]  = x_ekf[3:6] + dt * accel_world   # vel
    x_pred[6]    = roll_k                            # roll
    x_pred[7]    = pitch_k                           # pitch
    x_pred[8]    = yaw_k                             # yaw
    # bias: random walk → predict = mevcut değer (sabit model)
    x_pred[9:12]  = x_ekf[9:12]    # bax, bay, baz
    x_pred[12:15] = x_ekf[12:15]   # bgx, bgy, bgz

    # ── 4) F Jacobian (15 x 15) ──────────────────────────────────
    F = np.eye(n_state)

    # pos ← vel
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    # vel ← yaw (accel_world'ün yaw'a türevi)
    fx, fy, fz = f_hat
    cy, sy = np.cos(x_ekf[8]), np.sin(x_ekf[8])
    F[3, 8] = dt * (-sy * fx - cy * fy)
    F[4, 8] = dt * ( cy * fx - sy * fy)

    # vel ← accel bias (Rbn @ (-ba) → vel)
    # d(vel)/d(ba) = -Rbn * dt
    F[3:6, 9:12] = -Rbn_ekf * dt

    # attitude ← gyro bias (Euler rate ← gyro bias)
    # Basit yaklaşım: attitude değişimi = -bg * dt
    # (tam Jacobian Transition matrix gerektirir ama bu yeterince iyi)
    F[6, 12] = -dt
    F[7, 13] = -dt
    F[8, 14] = -dt

    # bias blokları: F[9:15, 9:15] = I (random walk, zaten np.eye)

    # ── 5) Q matrisi (15 x 15) ───────────────────────────────────
    # Pozisyon: çok küçük (GPS baskın)
    # Hız: accel white noise * dt
    # Attitude: gyro white noise * dt
    # Accel bias: random walk * dt
    # Gyro  bias: random walk * dt
    Q_k = np.zeros((n_state, n_state))

    # pos
    Q_k[0, 0] = (0.01 * dt)**2
    Q_k[1, 1] = (0.01 * dt)**2
    Q_k[2, 2] = (0.01 * dt)**2

    # vel (accel white noise)
    Q_k[3, 3] = (sigma_ax * np.sqrt(dt))**2
    Q_k[4, 4] = (sigma_ay * np.sqrt(dt))**2
    Q_k[5, 5] = (sigma_az * np.sqrt(dt))**2
    # attitude (gyro white noise)
    Q_k[6, 6] = (sigma_wx * np.sqrt(dt))**2
    Q_k[7, 7] = (sigma_wy * np.sqrt(dt))**2

    # yaw GPS ile direkt ölçülmediği için model belirsizliği daha büyük bırakılmalı
    Q_k[8, 8] = (3.0 * sigma_wz * np.sqrt(dt))**2

    # accel bias random walk
    Q_k[9,  9]  = (sigma_rw_accel * np.sqrt(dt))**2
    Q_k[10, 10] = (sigma_rw_accel * np.sqrt(dt))**2
    Q_k[11, 11] = (sigma_rw_accel * np.sqrt(dt))**2

    # gyro bias random walk
    Q_k[12, 12] = (sigma_rw_gyro * np.sqrt(dt))**2
    Q_k[13, 13] = (sigma_rw_gyro * np.sqrt(dt))**2
    Q_k[14, 14] = (sigma_rw_gyro * np.sqrt(dt))**2

    P = F @ P @ F.T + Q_k

    # ── 6) GPS güncelleme ────────────────────────────────────────
    t_current = t_imu[k]

    while gps_idx < len(gps_timestamps) and gps_timestamps[gps_idx] <= t_current:

        if abs(gps_timestamps[gps_idx] - t_current) <= gps_half_dt:

            z_meas = np.array([
                gps_x[gps_idx], gps_y[gps_idx], gps_z[gps_idx],
                gps_vx[gps_idx], gps_vy[gps_idx], gps_vz[gps_idx]
            ])

            z_pred = H @ x_pred
            innov  = z_meas - z_pred

            S = H @ P @ H.T + R_meas
            K = P @ H.T @ np.linalg.inv(S)

            x_pred = x_pred + K @ innov

            # Yaw wrap -π..π
            x_pred[8] = (x_pred[8] + np.pi) % (2 * np.pi) - np.pi

            I_KH = np.eye(n_state) - K @ H
            P    = I_KH @ P @ I_KH.T + K @ R_meas @ K.T
            P    = 0.5 * (P + P.T)   # simetri koru

            # Rbn_ekf'i güncel açılarla senkronize et
            Rbn_ekf = R.from_euler('xyz',
                                   [x_pred[6], x_pred[7], x_pred[8]]
                                   ).as_matrix()

        gps_idx += 1

    x_ekf        = x_pred
    P_hist[k+1]  = P.copy()
    ekf_p[k+1]      = x_ekf[0:3]
    ekf_v[k+1]      = x_ekf[3:6]
    ekf_euler[k+1]  = np.degrees([x_ekf[6], x_ekf[7], x_ekf[8]])
    ekf_ba[k+1]     = x_ekf[9:12]
    ekf_bg[k+1]     = x_ekf[12:15]

print(f"\nEKF (15-state) tamamlandı.")
print(f"Son konum EKF  : {ekf_p[-1]}")
print(f"Son konum INS  : {p[-1]}")
print(f"Son accel bias : {ekf_ba[-1]}")
print(f"Son gyro  bias : {ekf_bg[-1]}")

# ══════════════════════════════════════════════════════════════════
# GÖRSELLEŞTİRME HAZIRLIK
# ══════════════════════════════════════════════════════════════════

gt_timestamps = df_gt['timestamp'].values
gt_p_raw      = df_gt[['tx', 'ty', 'tz']].values
gt_q_raw      = df_gt[['qx', 'qy', 'qz', 'qw']].values

gt_p_interp = np.zeros((n, 3))
for i in range(3):
    gt_p_interp[:, i] = np.interp(t_imu, gt_timestamps, gt_p_raw[:, i])

gt_v_interp = np.zeros((n, 3))
for i in range(3):
    gt_v_interp[:, i] = np.gradient(gt_p_interp[:, i], t_imu)

gt_q_interp = np.zeros((n, 4))
for i in range(4):
    gt_q_interp[:, i] = np.interp(t_imu, gt_timestamps, gt_q_raw[:, i])
gt_q_interp = gt_q_interp / np.linalg.norm(gt_q_interp, axis=1, keepdims=True)

gt_euler_interp = np.zeros((n, 3))
for i in range(n):
    gt_euler_interp[i, :] = R.from_quat(gt_q_interp[i]).as_euler('xyz', degrees=True)

t_rel = t_imu - t_imu[0]

def angle_diff_deg(est_deg, true_deg):
    return (est_deg - true_deg + 180.0) % 360.0 - 180.0

# State ve hata matrisleri (9 state için plot — bias ayrı)
x_est  = np.zeros((n, 9))
x_est[:, 0:3] = ekf_p
x_est[:, 3:6] = ekf_v
x_est[:, 6:9] = ekf_euler

x_true = np.zeros((n, 9))
x_true[:, 0:3] = gt_p_interp
x_true[:, 3:6] = gt_v_interp
x_true[:, 6:9] = gt_euler_interp

err_plot = np.zeros_like(x_est)
err_plot[:, 0:3] = x_est[:, 0:3] - x_true[:, 0:3]
err_plot[:, 3:6] = x_est[:, 3:6] - x_true[:, 3:6]
err_plot[:, 6]   = angle_diff_deg(x_est[:, 6], x_true[:, 6])
err_plot[:, 7]   = angle_diff_deg(x_est[:, 7], x_true[:, 7])
err_plot[:, 8]   = angle_diff_deg(x_est[:, 8], x_true[:, 8])

# 3σ: ilk 9 state için P diyagonali
sigma_band       = np.sqrt(np.maximum(np.diagonal(P_hist, axis1=1, axis2=2), 0.0))
three_sigma_plot = 3.0 * sigma_band.copy()
three_sigma_plot[:, 6:9] = np.degrees(three_sigma_plot[:, 6:9])

state_names = [
    'North Position (m)', 'East Position (m)', 'Up Position (m)',
    'North Velocity (m/s)', 'East Velocity (m/s)', 'Up Velocity (m/s)',
    'Roll (deg)', 'Pitch (deg)', 'Yaw (deg)'
]

# ══════════════════════════════════════════════════════════════════
# PLOT 1 — 3D Trajectory + Position time series
# ══════════════════════════════════════════════════════════════════
plt.figure(figsize=(14, 10))

ax = plt.subplot(2, 1, 1, projection='3d')
ax.plot(ekf_p[:, 0], ekf_p[:, 1], ekf_p[:, 2],
        label='EKF Estimate', color='blue', lw=1.2)
ax.plot(gt_p_interp[:, 0], gt_p_interp[:, 1], gt_p_interp[:, 2],
        label='Ground Truth', color='green', linestyle='--', lw=1.5)
ax.set_title("3D Trajectory — EKF (15-state) vs Ground Truth")
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
ax.legend()

plt.subplot(2, 1, 2)
for i, (col, lbl) in enumerate(zip(['blue', 'orange', 'red'], ['N', 'E', 'U'])):
    plt.plot(t_imu, ekf_p[:, i],       color=col,        label=f'EKF {lbl}')
    plt.plot(t_imu, gt_p_interp[:, i], color=col, ls=':', label=f'GT {lbl}')
plt.title("Position Comparison (Time Domain)")
plt.xlabel("Time (s)"); plt.ylabel("Position (m)")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════════
# PLOT 2 — Estimation Errors, 3x3
# ══════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(3, 3, figsize=(18, 14), sharex=True)
axes1 = axes1.ravel()

for i in range(9):
    ax = axes1[i]
    ax.plot(t_rel, err_plot[:, i], color='blue', linewidth=0.9, label='Error')
    ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
    ax.set_title(state_names[i])
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right')
    rmse = np.sqrt(np.mean(err_plot[:, i]**2))
    ax.text(0.99, 0.95, f'RMSE={rmse:.4f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, bbox=dict(fc='white', ec='gray', alpha=0.8))

for ax in axes1[6:9]:
    ax.set_xlabel('Time (s)')

plt.suptitle('Estimation Errors — 15-state EKF', fontsize=16)
plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════════
# PLOT 3 — Error ± 3σ, 3x3
# ══════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(3, 3, figsize=(18, 14), sharex=True)
axes2 = axes2.ravel()

for i in range(9):
    ax = axes2[i]
    ax.plot(t_rel, err_plot[:, i],           color='blue', linewidth=0.9, label='Error')
    ax.plot(t_rel,  three_sigma_plot[:, i],  color='red',  linewidth=1.2,
            linestyle='--', label='+3σ')
    ax.plot(t_rel, -three_sigma_plot[:, i],  color='red',  linewidth=1.2,
            linestyle='--', label='-3σ')
    ax.axhline(0, color='gray', linewidth=0.6, linestyle='--')
    ax.set_title(state_names[i])
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right')
    rmse = np.sqrt(np.mean(err_plot[:, i]**2))
    ax.text(0.99, 0.95, f'RMSE={rmse:.4f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, bbox=dict(fc='white', ec='gray', alpha=0.8))

for ax in axes2[6:9]:
    ax.set_xlabel('Time (s)')

plt.suptitle('Estimation Error and ±3σ Bounds — 15-state EKF', fontsize=16)
plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════════
# PLOT 4 — Yaw detay
# ══════════════════════════════════════════════════════════════════
fig_yaw, axes_yaw = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes_yaw[0].plot(t_rel, ekf_euler[:, 2],       color='blue',  lw=1.0, label='EKF Yaw')
axes_yaw[0].plot(t_rel, gt_euler_interp[:, 2], color='green', lw=1.5,
                 linestyle='--', label='GT Yaw')
axes_yaw[0].set_ylabel('Yaw (deg)')
axes_yaw[0].set_title('Yaw Angle: EKF vs Ground Truth')
axes_yaw[0].legend(); axes_yaw[0].grid(True, alpha=0.4)

axes_yaw[1].plot(t_rel, err_plot[:, 8],           color='blue', lw=0.9, label='Yaw Error')
axes_yaw[1].plot(t_rel,  three_sigma_plot[:, 8],  color='red',  lw=1.2,
                 linestyle='--', label='+3σ')
axes_yaw[1].plot(t_rel, -three_sigma_plot[:, 8],  color='red',  lw=1.2,
                 linestyle='--', label='-3σ')
axes_yaw[1].axhline(0, color='gray', lw=0.6, linestyle='--')
axes_yaw[1].set_ylabel('Yaw Error (deg)')
axes_yaw[1].set_xlabel('Time (s)')
axes_yaw[1].set_title('Yaw Error ±3σ')
axes_yaw[1].legend(); axes_yaw[1].grid(True, alpha=0.4)
yaw_rmse = np.sqrt(np.mean(err_plot[:, 8]**2))
axes_yaw[1].text(0.99, 0.95, f'RMSE={yaw_rmse:.4f} deg',
                 transform=axes_yaw[1].transAxes, ha='right', va='top',
                 fontsize=9, bbox=dict(fc='white', ec='gray', alpha=0.8))
plt.tight_layout()
plt.show()

# ══════════════════════════════════════════════════════════════════
# PLOT 5 — Bias tahminleri
# ══════════════════════════════════════════════════════════════════
fig_bias, axes_bias = plt.subplots(2, 3, figsize=(18, 8), sharex=True)

bias_labels_a = ['bax (m/s²)', 'bay (m/s²)', 'baz (m/s²)']
bias_labels_g = ['bgx (rad/s)', 'bgy (rad/s)', 'bgz (rad/s)']
true_ba = [accel_bias[0], accel_bias[1], accel_bias[2]]
true_bg = [bg[0], bg[1], bg[2]]

for i in range(3):
    # Accel bias
    ax = axes_bias[0, i]
    ax.plot(t_rel, ekf_ba[:, i], color='blue', lw=0.9, label='EKF estimate')
    ax.axhline(true_ba[i], color='green', lw=1.2, linestyle='--', label='True bias')
    ax.set_title(bias_labels_a[i])
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.4)

    # Gyro bias
    ax = axes_bias[1, i]
    ax.plot(t_rel, ekf_bg[:, i], color='blue', lw=0.9, label='EKF estimate')
    ax.axhline(true_bg[i], color='green', lw=1.2, linestyle='--', label='True bias')
    ax.set_title(bias_labels_g[i])
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.4)

for ax in axes_bias[1, :]:
    ax.set_xlabel('Time (s)')

axes_bias[0, 0].set_ylabel('Accel Bias (m/s²)')
axes_bias[1, 0].set_ylabel('Gyro Bias (rad/s)')
plt.suptitle('IMU Bias Estimates — 15-state EKF', fontsize=16)
plt.tight_layout()
plt.show()












# ══════════════════════════════════════════════════════════════════
# TIGHTLY COUPLED GPS VERİSİ YÜKLE
# ══════════════════════════════════════════════════════════════════

df_tight = pd.read_csv(
    'project_src/gps_tightly.txt',
    sep=r'\s+',
    comment='#',
    header=0
)
df_tight = df_tight.sort_values('timestamp').reset_index(drop=True)

df_tight['timestamp'] = df_tight['timestamp'].astype(float)
tight_timestamps = df_tight['timestamp'].values

# Kaç uydu var otomatik bul
sat_ids = []
for col in df_tight.columns:
    if col.startswith("pr_sat"):
        sat_id = int(col.replace("pr_sat", ""))
        sat_ids.append(sat_id)

sat_ids = sorted(sat_ids)
N_SATS_TIGHT = len(sat_ids)

print(f"Tightly GPS uydu sayısı: {N_SATS_TIGHT}")
print(f"Tightly zaman aralığı : {tight_timestamps[0]:.6f} -> {tight_timestamps[-1]:.6f}")


# ══════════════════════════════════════════════════════════════════
# WGS84 / ECEF yardımcı fonksiyonlar
# ══════════════════════════════════════════════════════════════════

A_WGS84  = 6378137.0
E2_WGS84 = 6.69437999014e-3

REF_LAT_DEG = 39.925018
REF_LON_DEG = 32.836956
REF_ALT     = 850.0

def lla2ecef(lat_deg, lon_deg, alt):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    N = A_WGS84 / np.sqrt(1.0 - E2_WGS84 * np.sin(lat)**2)

    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1.0 - E2_WGS84) + alt) * np.sin(lat)

    return np.array([x, y, z])


def neu2ecef_matrix(lat_deg, lon_deg):
    """
    ECEF <- NEU
    Columns are ECEF directions of North, East, Up.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)

    N = np.array([-sl * co, -sl * so,  cl])
    E = np.array([-so,       co,       0.0])
    U = np.array([ cl * co,  cl * so,  sl])

    return np.column_stack([N, E, U])


def skew_mat(a):
    ax, ay, az = a
    return np.array([
        [0.0, -az,  ay],
        [az,  0.0, -ax],
        [-ay, ax,  0.0]
    ])


def wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def orthonormalize_R(Rmat):
    U, _, Vt = np.linalg.svd(Rmat)
    Rout = U @ Vt

    if np.linalg.det(Rout) < 0:
        U[:, -1] *= -1.0
        Rout = U @ Vt

    return Rout


def euler_from_R_xyz(Rmat):
    eul = R.from_matrix(Rmat).as_euler('xyz', degrees=False)
    eul[2] = wrap_pi(eul[2])
    return eul


ecef_ref = lla2ecef(REF_LAT_DEG, REF_LON_DEG, REF_ALT)
C_neu2ecef = neu2ecef_matrix(REF_LAT_DEG, REF_LON_DEG)


# ══════════════════════════════════════════════════════════════════
# TIGHT GPS MEASUREMENT MODEL
# ══════════════════════════════════════════════════════════════════

def build_tight_measurement(row, x_state):
    """
    Tightly-coupled measurement model.

    State:
      x[0:3]   = p_neu
      x[3:6]   = v_neu
      x[15]    = clock bias  [m]
      x[16]    = clock drift [m/s]

    Measurement per satellite:
      pr  = rho + cb
      prr = rho_dot + cd
    """

    p_neu = x_state[0:3]
    v_neu = x_state[3:6]

    cb = x_state[15]
    cd = x_state[16]

    rx_pos_ecef = ecef_ref + C_neu2ecef @ p_neu
    rx_vel_ecef = C_neu2ecef @ v_neu

    z_list = []
    h_list = []
    H_rows = []
    R_diag = []

    for sid in sat_ids:

        pr_col  = f"pr_sat{sid}"
        prr_col = f"prr_sat{sid}"

        sx_col  = f"sat{sid}_x"
        sy_col  = f"sat{sid}_y"
        sz_col  = f"sat{sid}_z"

        svx_col = f"sat{sid}_vx"
        svy_col = f"sat{sid}_vy"
        svz_col = f"sat{sid}_vz"

        if pr_col not in row or prr_col not in row:
            continue

        pr_meas  = float(row[pr_col])
        prr_meas = float(row[prr_col])

        sat_pos = np.array([
            float(row[sx_col]),
            float(row[sy_col]),
            float(row[sz_col])
        ])

        sat_vel = np.array([
            float(row[svx_col]),
            float(row[svy_col]),
            float(row[svz_col])
        ])

        los_vec = sat_pos - rx_pos_ecef
        rho = np.linalg.norm(los_vec)

        if rho <= 1e-9:
            continue

        u_los = los_vec / rho

        rel_vel = sat_vel - rx_vel_ecef
        rho_dot = np.dot(u_los, rel_vel)

        # Predicted measurements
        pr_pred  = rho + cb
        prr_pred = rho_dot + cd

        # ----------------------------------------------------------
        # Pseudorange Jacobian
        # rho = ||sat - rx||
        # d rho / d rx = -u_los
        # rx = ref + C_neu2ecef p_neu
        # d rho / d p_neu = -u_los^T C_neu2ecef
        # ----------------------------------------------------------
        H_pr = np.zeros(17)
        H_pr[0:3] = -u_los @ C_neu2ecef
        H_pr[15]  = 1.0

        # ----------------------------------------------------------
        # Pseudorange-rate Jacobian
        # rho_dot = u_los · (sat_vel - rx_vel)
        #
        # d rho_dot / d v_neu = -u_los^T C_neu2ecef
        #
        # d rho_dot / d p_neu:
        #   du/drx = -(I - u u^T) / rho
        #   d rho_dot/drx = -rel_vel^T (I - u u^T) / rho
        # ----------------------------------------------------------
        rel_perp = rel_vel - u_los * rho_dot

        H_prr = np.zeros(17)
        H_prr[0:3] = -(rel_perp / rho) @ C_neu2ecef
        H_prr[3:6] = -u_los @ C_neu2ecef
        H_prr[16]  = 1.0

        z_list.append(pr_meas)
        h_list.append(pr_pred)
        H_rows.append(H_pr)
        R_diag.append(PR_NOISE_STD**2)

        z_list.append(prr_meas)
        h_list.append(prr_pred)
        H_rows.append(H_prr)
        R_diag.append(PRR_NOISE_STD**2)

    z = np.array(z_list)
    h = np.array(h_list)
    H = np.vstack(H_rows)
    R_meas = np.diag(R_diag)

    return z, h, H, R_meas


# ══════════════════════════════════════════════════════════════════
# 17-STATE TIGHTLY COUPLED EKF TANIMLAR
# ══════════════════════════════════════════════════════════════════

n_state = 17

# IMU noise — senin loosely kodundaki değerler
sigma_ax = 0.02735837
sigma_ay = 0.02330523
sigma_az = 0.05322704

sigma_wx = 0.00277449
sigma_wy = 0.00310758
sigma_wz = 0.00185727

sigma_rw_gyro  = 4e-5
sigma_rw_accel = 0.002

# Tightly generation kodundaki değerler
PR_NOISE_STD  = 1.0
PRR_NOISE_STD = 0.05


# ══════════════════════════════════════════════════════════════════
# EKF BAŞLANGIÇ
# ══════════════════════════════════════════════════════════════════

x_ekf = np.zeros(n_state)

# Pozisyon: GT başlangıcından
x_ekf[0:3] = np.array([
    np.interp(t_start, df_gt['timestamp'].values, df_gt['tx'].values.astype(float)),
    np.interp(t_start, df_gt['timestamp'].values, df_gt['ty'].values.astype(float)),
    np.interp(t_start, df_gt['timestamp'].values, df_gt['tz'].values.astype(float))
])

# Hız: senin sistemde v0
x_ekf[3:6] = v0

# Attitude: eski iyi çalışan yapı gibi
x_ekf[6] = np.radians(roll_init)
x_ekf[7] = np.radians(pitch_init)
x_ekf[8] = np.radians(yaw_init)

# IMU bias başlangıcı
x_ekf[9]  = accel_bias[0]
x_ekf[10] = accel_bias[1]
x_ekf[11] = accel_bias[2]

x_ekf[12] = bg[0]
x_ekf[13] = bg[1]
x_ekf[14] = bg[2]

# Clock başlangıcı
# Simülasyonda clock_bias_m / clock_drift_mps dosyada var.
# Gerçek sistemde bunları bilmezdik; burada başlangıç transient'i azaltmak için kullanıyoruz.
USE_CLOCK_COLUMNS_FOR_INIT = True

if USE_CLOCK_COLUMNS_FOR_INIT and \
   ("clock_bias_m" in df_tight.columns) and \
   ("clock_drift_mps" in df_tight.columns):

    x_ekf[15] = np.interp(
        t_start,
        tight_timestamps,
        df_tight["clock_bias_m"].values.astype(float)
    )

    x_ekf[16] = np.interp(
        t_start,
        tight_timestamps,
        df_tight["clock_drift_mps"].values.astype(float)
    )

else:
    x_ekf[15] = 0.0
    x_ekf[16] = 0.0


# Rotation matrix
Rbn_ekf = rot0.as_matrix().copy()


# ══════════════════════════════════════════════════════════════════
# P BAŞLANGIÇ
# ══════════════════════════════════════════════════════════════════
#
# Burada GPS loose pos_std yok.
# Pseudorange noise PR_NOISE_STD ve PRR_NOISE_STD var.
#
# Clock bias/drift ayrı state olduğu için pseudorange residual'ın büyük kısmı
# clock'a gitmeli, pozisyona değil.

P = np.diag([
    PR_NOISE_STD**2,        # pN
    PR_NOISE_STD**2,        # pE
    PR_NOISE_STD**2,        # pU

    PRR_NOISE_STD**2,       # vN
    PRR_NOISE_STD**2,       # vE
    PRR_NOISE_STD**2,       # vU

    np.radians(10)**2,      # roll
    np.radians(10)**2,      # pitch
    np.radians(25)**2,      # yaw — GPS yaw direkt ölçmez

    sigma_ax**2,            # bax
    sigma_ay**2,            # bay
    sigma_az**2,            # baz

    sigma_wx**2,            # bgx
    sigma_wy**2,            # bgy
    sigma_wz**2,            # bgz

    PR_NOISE_STD**2,        # clock bias [m]
    PRR_NOISE_STD**2,       # clock drift [m/s]
])

# Eğer clock truth ile initialize ETMİYORSAN clock P büyük olmalı
if not USE_CLOCK_COLUMNS_FOR_INIT:
    if "clock_bias_m" in df_tight.columns:
        cb0_scale = abs(df_tight["clock_bias_m"].iloc[0]) + PR_NOISE_STD
    else:
        cb0_scale = 100.0

    if "clock_drift_mps" in df_tight.columns:
        cd0_scale = abs(df_tight["clock_drift_mps"].iloc[0]) + PRR_NOISE_STD
    else:
        cd0_scale = 5.0

    P[15, 15] = cb0_scale**2
    P[16, 16] = cd0_scale**2

print("\nInitial tightly EKF:")
print("x clock bias  :", x_ekf[15])
print("x clock drift :", x_ekf[16])
print("P clock std   :", np.sqrt(P[15, 15]), np.sqrt(P[16, 16]))


# ══════════════════════════════════════════════════════════════════
# KAYIT DİZİLERİ
# ══════════════════════════════════════════════════════════════════

ekf_p      = np.zeros((n, 3))
ekf_v      = np.zeros((n, 3))
ekf_euler  = np.zeros((n, 3))
ekf_ba     = np.zeros((n, 3))
ekf_bg     = np.zeros((n, 3))
ekf_clock  = np.zeros((n, 2))

ekf_p[0]      = x_ekf[0:3]
ekf_v[0]      = x_ekf[3:6]
ekf_euler[0]  = [roll_init, pitch_init, yaw_init]
ekf_ba[0]     = x_ekf[9:12]
ekf_bg[0]     = x_ekf[12:15]
ekf_clock[0]  = x_ekf[15:17]

P_hist = np.zeros((n, n_state, n_state))
P_hist[0] = P.copy()

real_g  = 9.80655
g_world = np.array([0.0, 0.0, real_g])

tight_idx = np.searchsorted(tight_timestamps, t_imu[0], side='left')
tight_half_dt = 0.5 * np.median(np.diff(tight_timestamps))

tight_update_times = []
tight_nis_hist = []


# ══════════════════════════════════════════════════════════════════
# TIGHTLY COUPLED EKF DÖNGÜSÜ
# ══════════════════════════════════════════════════════════════════
tight_update_times = []
tight_nis_hist = []
tight_innov_hist = []
tight_meas_dim_hist = []
for k in range(n - 1):

    dt = dt_arr[k]
    if dt <= 0 or dt > 0.1:
        dt = dt_median

    # ─────────────────────────────────────────────────────────────
    # 1) IMU raw reconstruction
    # ─────────────────────────────────────────────────────────────
    omega_raw = imu_arr[k, 0:3] + bg
    f_raw     = imu_arr[k, 3:6] + accel_bias

    ba_k = x_ekf[9:12]
    bg_k = x_ekf[12:15]

    omega = omega_raw - bg_k
    f_hat = f_raw     - ba_k

    # ─────────────────────────────────────────────────────────────
    # 2) Attitude predict
    # ─────────────────────────────────────────────────────────────
    wx_, wy_, wz_ = omega

    skew = np.array([
        [0.0, -wz_,  wy_],
        [wz_,  0.0, -wx_],
        [-wy_, wx_,  0.0]
    ])

    R_new = Rbn_ekf @ (np.eye(3) + skew * dt)

    U, _, Vt = np.linalg.svd(R_new)
    Rbn_ekf = U @ Vt

    roll_k  = np.arctan2(Rbn_ekf[2, 1], Rbn_ekf[2, 2])
    pitch_k = -np.arcsin(np.clip(Rbn_ekf[2, 0], -1.0, 1.0))
    yaw_k   = np.arctan2(Rbn_ekf[1, 0], Rbn_ekf[0, 0])

    # ─────────────────────────────────────────────────────────────
    # 3) Acceleration world frame
    # ─────────────────────────────────────────────────────────────
    accel_world = Rbn_ekf @ f_hat - g_world

    # ─────────────────────────────────────────────────────────────
    # 4) State predict
    # ─────────────────────────────────────────────────────────────
    x_pred = x_ekf.copy()

    x_pred[0:3] = x_ekf[0:3] + dt * x_ekf[3:6]
    x_pred[3:6] = x_ekf[3:6] + dt * accel_world

    x_pred[6] = roll_k
    x_pred[7] = pitch_k
    x_pred[8] = yaw_k

    x_pred[9:12]  = x_ekf[9:12]
    x_pred[12:15] = x_ekf[12:15]

    # Clock model
    # cb_dot = cd
    # cd_dot = 0
    x_pred[15] = x_ekf[15] + x_ekf[16] * dt
    x_pred[16] = x_ekf[16]

    # ─────────────────────────────────────────────────────────────
    # 5) F Jacobian
    # ─────────────────────────────────────────────────────────────
    F = np.eye(n_state)

    # pos <- vel
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    # eski iyi çalışan yapı: yaw coupling
    fx, fy, fz = f_hat
    cy, sy = np.cos(x_ekf[8]), np.sin(x_ekf[8])

    F[3, 8] = dt * (-sy * fx - cy * fy)
    F[4, 8] = dt * ( cy * fx - sy * fy)

    # vel <- accel bias
    F[3:6, 9:12] = -Rbn_ekf * dt

    # attitude covariance propagation
    F[6:9, 6:9] = np.eye(3) - skew * dt

    # attitude <- gyro bias
    F[6, 12] = -dt
    F[7, 13] = -dt
    F[8, 14] = -dt

    # clock bias <- clock drift
    F[15, 16] = dt

    # ─────────────────────────────────────────────────────────────
    # 6) Q process covariance
    # ─────────────────────────────────────────────────────────────
    Q_k = np.zeros((n_state, n_state))

    # pos small model noise
    Q_k[0, 0] = (0.01 * dt)**2
    Q_k[1, 1] = (0.01 * dt)**2
    Q_k[2, 2] = (0.01 * dt)**2

    # velocity from accel noise
    Q_k[3, 3] = (sigma_ax * np.sqrt(dt))**2
    Q_k[4, 4] = (sigma_ay * np.sqrt(dt))**2
    Q_k[5, 5] = (sigma_az * np.sqrt(dt))**2

    # attitude from gyro noise
    Q_k[6, 6] = (sigma_wx * np.sqrt(dt))**2
    Q_k[7, 7] = (sigma_wy * np.sqrt(dt))**2

    # yaw direkt ölçülmediği için covariance biraz daha açık kalmalı
    Q_k[8, 8] = (3.0 * sigma_wz * np.sqrt(dt))**2

    # accel bias random walk
    Q_k[9,  9] = (sigma_rw_accel * np.sqrt(dt))**2
    Q_k[10,10] = (sigma_rw_accel * np.sqrt(dt))**2
    Q_k[11,11] = (sigma_rw_accel * np.sqrt(dt))**2

    # gyro bias random walk
    Q_k[12,12] = (sigma_rw_gyro * np.sqrt(dt))**2
    Q_k[13,13] = (sigma_rw_gyro * np.sqrt(dt))**2

    # bgz yaw drift'i etkiler
    Q_k[14,14] = (5.0 * sigma_rw_gyro * np.sqrt(dt))**2

    # clock process
    # Generation'da clock bias lineer, drift sabit.
    # Model zaten cb <- cb + cd*dt dediği için ekstra clock Q çok küçük bırakılır.
    Q_k[15,15] = (PRR_NOISE_STD * dt)**2
    Q_k[16,16] = (PRR_NOISE_STD * np.sqrt(dt) * 0.01)**2

    P = F @ P @ F.T + Q_k
    P = 0.5 * (P + P.T)

    # ─────────────────────────────────────────────────────────────
    # 7) Tightly GPS update
    # ─────────────────────────────────────────────────────────────
    t_current = t_imu[k]

    while tight_idx < len(tight_timestamps) and tight_timestamps[tight_idx] <= t_current:

        if abs(tight_timestamps[tight_idx] - t_current) <= tight_half_dt:

            row = df_tight.iloc[tight_idx]

            z_meas, z_pred, H_tight, R_tight = build_tight_measurement(row, x_pred)

            if len(z_meas) >= 8:   # en az 4 uydu: pr+prr = 8 ölçüm

                innov = z_meas - z_pred

                S = H_tight @ P @ H_tight.T + R_tight
                S = 0.5 * (S + S.T)

                K = P @ H_tight.T @ np.linalg.inv(S)

                x_pred = x_pred + K @ innov

                x_pred[8] = wrap_pi(x_pred[8])

                I_KH = np.eye(n_state) - K @ H_tight
                P = I_KH @ P @ I_KH.T + K @ R_tight @ K.T
                P = 0.5 * (P + P.T)

                # Rbn_ekf'i corrected Euler ile senkronize et
                Rbn_ekf = R.from_euler(
                    'xyz',
                    [x_pred[6], x_pred[7], x_pred[8]]
                ).as_matrix()

                nis = innov.T @ np.linalg.solve(S, innov)

                tight_update_times.append(tight_timestamps[tight_idx])
                tight_nis_hist.append(nis)
                tight_update_times.append(tight_timestamps[tight_idx])
                tight_nis_hist.append(nis)  
                tight_innov_hist.append(innov.copy())
                tight_meas_dim_hist.append(len(innov))

        tight_idx += 1

    # ─────────────────────────────────────────────────────────────
    # 8) Store
    # ─────────────────────────────────────────────────────────────
    x_ekf = x_pred.copy()

    P_hist[k + 1] = P.copy()

    ekf_p[k + 1]      = x_ekf[0:3]
    ekf_v[k + 1]      = x_ekf[3:6]
    ekf_euler[k + 1]  = np.degrees([x_ekf[6], x_ekf[7], x_ekf[8]])
    ekf_ba[k + 1]     = x_ekf[9:12]
    ekf_bg[k + 1]     = x_ekf[12:15]
    ekf_clock[k + 1]  = x_ekf[15:17]


print("\nTightly coupled EKF tamamlandı.")
print(f"Tight GPS update sayısı : {len(tight_update_times)}")
print(f"Son konum EKF           : {ekf_p[-1]}")
print(f"Son konum INS           : {p[-1]}")
print(f"Son accel bias          : {ekf_ba[-1]}")
print(f"Son gyro bias           : {ekf_bg[-1]}")
print(f"Son clock bias/drift    : {ekf_clock[-1]}")

if len(tight_nis_hist) > 0:
    print(f"NIS mean                : {np.mean(tight_nis_hist):.3f}")
    print(f"NIS max                 : {np.max(tight_nis_hist):.3f}")


# ══════════════════════════════════════════════════════════════════
# TIGHTLY COUPLED EKF — GÖRSELLEŞTİRME
# ══════════════════════════════════════════════════════════════════

gt_timestamps = df_gt['timestamp'].values
gt_p_raw      = df_gt[['tx', 'ty', 'tz']].values.astype(float)
gt_q_raw      = df_gt[['qx', 'qy', 'qz', 'qw']].values.astype(float)

t_rel = t_imu - t_imu[0]

# ────────────────────────────────────────────────────────────────
# GT position interpolate
# ────────────────────────────────────────────────────────────────
gt_p_interp = np.zeros((n, 3))
for i in range(3):
    gt_p_interp[:, i] = np.interp(
        t_imu,
        gt_timestamps,
        gt_p_raw[:, i]
    )

# GT velocity
gt_v_interp = np.zeros((n, 3))
for i in range(3):
    gt_v_interp[:, i] = np.gradient(gt_p_interp[:, i], t_imu)

# GT quaternion interpolate
gt_q_interp = np.zeros((n, 4))
for i in range(4):
    gt_q_interp[:, i] = np.interp(
        t_imu,
        gt_timestamps,
        gt_q_raw[:, i]
    )

gt_q_interp = gt_q_interp / np.linalg.norm(
    gt_q_interp,
    axis=1,
    keepdims=True
)

gt_euler_interp = np.zeros((n, 3))
for i in range(n):
    gt_euler_interp[i, :] = R.from_quat(
        gt_q_interp[i]
    ).as_euler('xyz', degrees=True)


def angle_diff_deg(est_deg, true_deg):
    return (est_deg - true_deg + 180.0) % 360.0 - 180.0


# ────────────────────────────────────────────────────────────────
# Error matrices
# ────────────────────────────────────────────────────────────────
x_est = np.zeros((n, 9))
x_est[:, 0:3] = ekf_p
x_est[:, 3:6] = ekf_v
x_est[:, 6:9] = ekf_euler

x_true = np.zeros((n, 9))
x_true[:, 0:3] = gt_p_interp
x_true[:, 3:6] = gt_v_interp
x_true[:, 6:9] = gt_euler_interp

err_plot = np.zeros_like(x_est)

err_plot[:, 0:3] = x_est[:, 0:3] - x_true[:, 0:3]
err_plot[:, 3:6] = x_est[:, 3:6] - x_true[:, 3:6]

err_plot[:, 6] = angle_diff_deg(x_est[:, 6], x_true[:, 6])
err_plot[:, 7] = angle_diff_deg(x_est[:, 7], x_true[:, 7])
err_plot[:, 8] = angle_diff_deg(x_est[:, 8], x_true[:, 8])


# ────────────────────────────────────────────────────────────────
# 3σ bands
# P_hist shape: (n, 17, 17)
# attitude states rad olduğu için degree'e çevriliyor.
# ────────────────────────────────────────────────────────────────
sigma_band = np.sqrt(
    np.maximum(
        np.diagonal(P_hist, axis1=1, axis2=2),
        0.0
    )
)

three_sigma = 3.0 * sigma_band.copy()
three_sigma[:, 6:9] = np.degrees(three_sigma[:, 6:9])

state_names = [
    'North Position (m)', 'East Position (m)', 'Up Position (m)',
    'North Velocity (m/s)', 'East Velocity (m/s)', 'Up Velocity (m/s)',
    'Roll (deg)', 'Pitch (deg)', 'Yaw (deg)'
]


# ══════════════════════════════════════════════════════════════════
# PLOT 1 — 3D Trajectory + Position time domain
# ══════════════════════════════════════════════════════════════════

plt.figure(figsize=(14, 10))

ax = plt.subplot(2, 1, 1, projection='3d')

ax.plot(
    ekf_p[:, 0],
    ekf_p[:, 1],
    ekf_p[:, 2],
    label='Tight EKF Estimate',
    color='blue',
    lw=1.2
)

ax.plot(
    gt_p_interp[:, 0],
    gt_p_interp[:, 1],
    gt_p_interp[:, 2],
    label='Ground Truth',
    color='green',
    linestyle='--',
    lw=1.5
)

ax.scatter(
    ekf_p[0, 0],
    ekf_p[0, 1],
    ekf_p[0, 2],
    color='black',
    s=60,
    label='Start'
)

ax.scatter(
    ekf_p[-1, 0],
    ekf_p[-1, 1],
    ekf_p[-1, 2],
    color='blue',
    s=60,
    label='EKF End'
)

ax.scatter(
    gt_p_interp[-1, 0],
    gt_p_interp[-1, 1],
    gt_p_interp[-1, 2],
    color='green',
    s=60,
    label='GT End'
)

ax.set_title("3D Trajectory — Tightly Coupled EKF vs Ground Truth")
ax.set_xlabel("North (m)")
ax.set_ylabel("East (m)")
ax.set_zlabel("Up (m)")
ax.legend()
ax.grid(True)

plt.subplot(2, 1, 2)

labels = ['N', 'E', 'U']
colors = ['blue', 'orange', 'red']

for i in range(3):
    plt.plot(
        t_imu,
        ekf_p[:, i],
        color=colors[i],
        label=f'EKF {labels[i]}'
    )

    plt.plot(
        t_imu,
        gt_p_interp[:, i],
        color=colors[i],
        linestyle=':',
        label=f'GT {labels[i]}'
    )

plt.title("Position Comparison — Time Domain")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════
# PLOT 2 — Estimation Errors
# ══════════════════════════════════════════════════════════════════

fig_err, axes_err = plt.subplots(3, 3, figsize=(18, 14), sharex=True)
axes_err = axes_err.ravel()

for i in range(9):
    ax = axes_err[i]

    ax.plot(
        t_rel,
        err_plot[:, i],
        color='blue',
        linewidth=0.9,
        label='Error'
    )

    ax.axhline(
        0.0,
        color='gray',
        linewidth=0.6,
        linestyle='--'
    )

    rmse = np.sqrt(np.mean(err_plot[:, i]**2))

    ax.set_title(state_names[i])
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right')

    ax.text(
        0.99,
        0.95,
        f'RMSE={rmse:.4f}',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=8,
        bbox=dict(fc='white', ec='gray', alpha=0.8)
    )

for ax in axes_err[6:9]:
    ax.set_xlabel("Time (s)")

plt.suptitle("Estimation Errors — Tightly Coupled EKF", fontsize=16)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════
# PLOT 3 — Error ±3σ
# ══════════════════════════════════════════════════════════════════

fig_3s, axes_3s = plt.subplots(3, 3, figsize=(18, 14), sharex=True)
axes_3s = axes_3s.ravel()

for i in range(9):
    ax = axes_3s[i]

    ax.plot(
        t_rel,
        err_plot[:, i],
        color='blue',
        linewidth=0.9,
        label='Error'
    )

    ax.plot(
        t_rel,
        three_sigma[:, i],
        color='red',
        linewidth=1.1,
        linestyle='--',
        label='+3σ'
    )

    ax.plot(
        t_rel,
        -three_sigma[:, i],
        color='red',
        linewidth=1.1,
        linestyle='--',
        label='-3σ'
    )

    ax.axhline(
        0.0,
        color='gray',
        linewidth=0.6,
        linestyle='--'
    )

    rmse = np.sqrt(np.mean(err_plot[:, i]**2))

    ax.set_title(state_names[i])
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right')

    ax.text(
        0.99,
        0.95,
        f'RMSE={rmse:.4f}',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=8,
        bbox=dict(fc='white', ec='gray', alpha=0.8)
    )

for ax in axes_3s[6:9]:
    ax.set_xlabel("Time (s)")

plt.suptitle("Estimation Error and ±3σ Bounds — Tightly Coupled EKF", fontsize=16)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════
# PLOT 4 — Velocity comparison
# ══════════════════════════════════════════════════════════════════

fig_vel, axes_vel = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

vel_names = ['North Velocity (m/s)', 'East Velocity (m/s)', 'Up Velocity (m/s)']

for i in range(3):
    ax = axes_vel[i]

    ax.plot(
        t_rel,
        ekf_v[:, i],
        color='blue',
        linewidth=0.9,
        label='Tight EKF'
    )

    ax.plot(
        t_rel,
        gt_v_interp[:, i],
        color='green',
        linewidth=1.2,
        linestyle='--',
        label='GT'
    )

    ax.set_ylabel(vel_names[i])
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right')

axes_vel[0].set_title("Velocity — Tightly Coupled EKF vs Ground Truth")
axes_vel[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════
# PLOT 5 — Euler angles
# ══════════════════════════════════════════════════════════════════

fig_eul, axes_eul = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

eul_names = ['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)']

for i in range(3):
    ax = axes_eul[i]

    ax.plot(
        t_rel,
        ekf_euler[:, i],
        color='blue',
        linewidth=0.9,
        label='Tight EKF'
    )

    ax.plot(
        t_rel,
        gt_euler_interp[:, i],
        color='green',
        linewidth=1.2,
        linestyle='--',
        label='GT'
    )

    ax.set_ylabel(eul_names[i])
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right')

axes_eul[0].set_title("Euler Angles — Tightly Coupled EKF vs Ground Truth")
axes_eul[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════
# PLOT 6 — Clock bias / drift
# ══════════════════════════════════════════════════════════════════

fig_clk, axes_clk = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes_clk[0].plot(
    t_rel,
    ekf_clock[:, 0],
    color='blue',
    linewidth=1.0,
    label='EKF clock bias'
)

axes_clk[1].plot(
    t_rel,
    ekf_clock[:, 1],
    color='blue',
    linewidth=1.0,
    label='EKF clock drift'
)

# Eğer simülasyon truth clock dosyada varsa onu da çiz
if "clock_bias_m" in df_tight.columns:
    clock_bias_true = np.interp(
        t_imu,
        tight_timestamps,
        df_tight["clock_bias_m"].values.astype(float)
    )

    axes_clk[0].plot(
        t_rel,
        clock_bias_true,
        color='green',
        linestyle='--',
        linewidth=1.2,
        label='True clock bias'
    )

if "clock_drift_mps" in df_tight.columns:
    clock_drift_true = np.interp(
        t_imu,
        tight_timestamps,
        df_tight["clock_drift_mps"].values.astype(float)
    )

    axes_clk[1].plot(
        t_rel,
        clock_drift_true,
        color='green',
        linestyle='--',
        linewidth=1.2,
        label='True clock drift'
    )

axes_clk[0].set_ylabel("Clock bias (m)")
axes_clk[1].set_ylabel("Clock drift (m/s)")
axes_clk[1].set_xlabel("Time (s)")

axes_clk[0].set_title("Receiver Clock States — Tightly Coupled EKF")

for ax in axes_clk:
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right')

plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════
# PLOT 7 — IMU bias estimates
# ══════════════════════════════════════════════════════════════════

fig_bias, axes_bias = plt.subplots(2, 3, figsize=(18, 8), sharex=True)

acc_bias_names = ['bax (m/s²)', 'bay (m/s²)', 'baz (m/s²)']
gyr_bias_names = ['bgx (rad/s)', 'bgy (rad/s)', 'bgz (rad/s)']

true_ba = accel_bias
true_bg = bg

for i in range(3):
    ax = axes_bias[0, i]

    ax.plot(
        t_rel,
        ekf_ba[:, i],
        color='blue',
        linewidth=0.9,
        label='EKF estimate'
    )

    ax.axhline(
        true_ba[i],
        color='green',
        linestyle='--',
        linewidth=1.2,
        label='Initial/calib bias'
    )

    ax.set_title(acc_bias_names[i])
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right')

    ax = axes_bias[1, i]

    ax.plot(
        t_rel,
        ekf_bg[:, i],
        color='blue',
        linewidth=0.9,
        label='EKF estimate'
    )

    ax.axhline(
        true_bg[i],
        color='green',
        linestyle='--',
        linewidth=1.2,
        label='Initial/calib bias'
    )

    ax.set_title(gyr_bias_names[i])
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right')

for ax in axes_bias[1, :]:
    ax.set_xlabel("Time (s)")

axes_bias[0, 0].set_ylabel("Accel bias")
axes_bias[1, 0].set_ylabel("Gyro bias")

plt.suptitle("IMU Bias Estimates — Tightly Coupled EKF", fontsize=16)
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════
# PLOT 8 — NIS
# ══════════════════════════════════════════════════════════════════

if len(tight_nis_hist) > 0:

    tight_update_times_rel = np.array(tight_update_times) - t_imu[0]
    tight_nis_hist_arr = np.array(tight_nis_hist)

    plt.figure(figsize=(14, 5))

    plt.plot(
        tight_update_times_rel,
        tight_nis_hist_arr,
        marker='o',
        markersize=3,
        linewidth=0.8,
        label='NIS'
    )

    # Her update'te ölçüm boyutu genelde 2*N_sat.
    if len(tight_meas_dim_hist) > 0:
        meas_dim_arr = np.array(tight_meas_dim_hist)
        plt.plot(
            tight_update_times_rel,
            meas_dim_arr,
            color='gray',
            linestyle='--',
            linewidth=1.0,
            label='Expected mean ≈ measurement dimension'
        )

    plt.title("Tightly Coupled GPS Innovation Consistency — NIS")
    plt.xlabel("Time (s)")
    plt.ylabel("NIS")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════
# PLOT 9 — trace(P)
# ══════════════════════════════════════════════════════════════════

trace_P = np.array([
    np.trace(P_hist[i])
    for i in range(n)
])

plt.figure(figsize=(14, 5))

plt.plot(
    t_rel,
    trace_P,
    color='blue',
    linewidth=1.0,
    label='trace(P)'
)

if len(tight_update_times) > 0:
    for tgps in np.array(tight_update_times) - t_imu[0]:
        plt.axvline(
            tgps,
            color='gray',
            alpha=0.12,
            linewidth=0.7
        )

plt.title("Total EKF Uncertainty — trace(P)")
plt.xlabel("Time (s)")
plt.ylabel("trace(P)")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()


# ══════════════════════════════════════════════════════════════════
# ÖZET METRİKLER
# ══════════════════════════════════════════════════════════════════

pos_err_3d = np.linalg.norm(err_plot[:, 0:3], axis=1)
vel_err_3d = np.linalg.norm(err_plot[:, 3:6], axis=1)

print("\n" + "=" * 70)
print("TIGHTLY COUPLED EKF PERFORMANCE SUMMARY")
print("=" * 70)

print(f"Position RMSE N/E/U : {np.sqrt(np.mean(err_plot[:,0]**2)):.4f}, "
      f"{np.sqrt(np.mean(err_plot[:,1]**2)):.4f}, "
      f"{np.sqrt(np.mean(err_plot[:,2]**2)):.4f} m")

print(f"3D Position RMSE    : {np.sqrt(np.mean(pos_err_3d**2)):.4f} m")
print(f"Final 3D Pos Error  : {pos_err_3d[-1]:.4f} m")

print(f"Velocity RMSE N/E/U : {np.sqrt(np.mean(err_plot[:,3]**2)):.4f}, "
      f"{np.sqrt(np.mean(err_plot[:,4]**2)):.4f}, "
      f"{np.sqrt(np.mean(err_plot[:,5]**2)):.4f} m/s")

print(f"3D Velocity RMSE    : {np.sqrt(np.mean(vel_err_3d**2)):.4f} m/s")

print(f"Euler RMSE R/P/Y    : {np.sqrt(np.mean(err_plot[:,6]**2)):.4f}, "
      f"{np.sqrt(np.mean(err_plot[:,7]**2)):.4f}, "
      f"{np.sqrt(np.mean(err_plot[:,8]**2)):.4f} deg")

if len(tight_nis_hist) > 0:
    print(f"NIS mean            : {np.mean(tight_nis_hist):.4f}")
    print(f"NIS max             : {np.max(tight_nis_hist):.4f}")

print("=" * 70)