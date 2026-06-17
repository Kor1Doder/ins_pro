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






visual_dead=True

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












############################EKF################################3333
# ══════════════════════════════════════════════════════════════════
# GPS verisi yükle
# ══════════════════════════════════════════════════════════════════
############################EKF################################3333
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

# Sütunları sayıya dönüştür
df_gps['lat'] = df_gps['lat'].astype(float)
df_gps['lon'] = df_gps['lon'].astype(float)
df_gps['alt'] = df_gps['alt'].astype(float)
# Referans nokta (WGS84)
REF_LAT_DEG = 39.925018
REF_LON_DEG = 32.836956
REF_ALT     = 850.0
# Referans: GPS üretimindeki aynı referans nokta
lat0 = np.radians(REF_LAT_DEG)
lon0 = np.radians(REF_LON_DEG)
alt0 = REF_ALT

R_earth = 6378137.0  # m

# --------------------------------------------------
# LLA -> NEU
# --------------------------------------------------
def lla_to_neu(lat_deg, lon_deg, alt):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    dlat = lat - lat0
    dlon = lon - lon0

    x = R_earth * dlat                              # North
    y = R_earth * dlon * np.cos(lat0)              # East
    z = alt - alt0                                 # Up
    return x, y, z

gps_x, gps_y, gps_z = lla_to_neu(
    df_gps['lat'].values,
    df_gps['lon'].values,
    df_gps['alt'].values
)

# GPS hızları: NEU sırası ile
gps_vx = df_gps['vn'].values    # North
gps_vy = df_gps['ve'].values    # East
gps_vz = df_gps['vu'].values    # Up
gps_pos_std_h = df_gps["pos_std_h"].iloc[0]
gps_pos_std_v = df_gps["pos_std_v"].iloc[0]
gps_vel_std   = df_gps["vel_std"].iloc[0]
# ══════════════════════════════════════════════════════════════════
# EKF TANIMLAR
# ══════════════════════════════════════════════════════════════════

# State: [px, py, pz, vx, vy, vz, roll, pitch, yaw]
# burada: p = [North, East, Up]
#        v = [Vn, Ve, Vu]
# roll/pitch/yaw: radyan

n_state = 9
n_meas  = 6   # [px, py, pz, vx, vy, vz]

# Gürültü matrisleri
sigma_ax = 0.02735837
sigma_ay = 0.02330523
sigma_az = 0.05322704
sigma_wx = 0.00277449
sigma_wy = 0.00310758
sigma_wz = 0.00185727

# Gürültüleri suni olarak artırıp filtreyi "şüpheci" yapıyoruz
Q = np.diag([
    1e-4, 1e-4, 1e-4,                                   # Pozisyon
    5*sigma_ax**2, 5*sigma_ay**2, 5*sigma_az**2,              # Hız (İvmeölçer gürültüsü)
    (sigma_wx * 10)**2, (sigma_wy * 10)**2, (sigma_wz * 100)**2  # Açı (Jiroskop gürültüsü - ÇARPILMIŞ)
])

 
R_meas = np.diag([
    gps_pos_std_h**2,
    gps_pos_std_h**2,
    gps_pos_std_v**2,
    gps_vel_std**2,
    gps_vel_std**2,
    gps_vel_std**2
])
H = np.zeros((n_meas, n_state))
H[0, 0] = 1.0   # N
H[1, 1] = 1.0   # E
H[2, 2] = 1.0   # U
H[3, 3] = 1.0   # Vn
H[4, 4] = 1.0   # Ve
H[5, 5] = 1.0   # Vu

# ══════════════════════════════════════════════════════════════════
# EKF BAŞLANGIÇ
# ══════════════════════════════════════════════════════════════════

x_ekf = np.zeros(n_state)
x_ekf[0:3] = np.array([
    np.interp(t_start, df_gt['timestamp'].values, df_gt['tx'].values.astype(float)),
    np.interp(t_start, df_gt['timestamp'].values, df_gt['ty'].values.astype(float)),
    np.interp(t_start, df_gt['timestamp'].values, df_gt['tz'].values.astype(float))
])                         # p0: NEU olmalı
x_ekf[3:6] = v0
x_ekf[6]   = np.radians(roll_init)
x_ekf[7]   = np.radians(pitch_init)
x_ekf[8]   = np.radians(yaw_init)

P = np.eye(n_state) * 0.1
Rbn_ekf = rot0.as_matrix().copy()

ekf_p     = np.zeros((n, 3))
ekf_v     = np.zeros((n, 3))
ekf_euler = np.zeros((n, 3))

ekf_p[0]     = x_ekf[0:3]
ekf_v[0]     = x_ekf[3:6]
ekf_euler[0] = [roll_init, pitch_init, yaw_init]

gps_timestamps = df_gps['timestamp'].values
gps_idx = 0

# ══════════════════════════════════════════════════════════════════
# EKF DÖNGÜSÜ
# ══════════════════════════════════════════════════════════════════
P_hist = np.zeros((n, n_state, n_state))
P_hist[0] = P
gps_idx = 0
gps_half_dt = 0.5 * np.median(np.diff(gps_timestamps))
for k in range(n - 1):

    dt = dt_arr[k]
    if dt <= 0 or dt > 0.1:
        dt = dt_median

    omega = imu_arr[k, 0:3]
    f_hat = imu_arr[k, 3:6]

    # 1) Attitude predict
    wx_, wy_, wz_ = omega
    skew = np.array([
        [0,    -wz_,  wy_],
        [wz_,   0,   -wx_],
        [-wy_,  wx_,  0  ]
    ])

    R_new = Rbn_ekf @ (np.eye(3) + skew * dt)
    U, _, Vt = np.linalg.svd(R_new)
    Rbn_ekf = U @ Vt

    roll_k  = np.arctan2(Rbn_ekf[2, 1], Rbn_ekf[2, 2])
    pitch_k = -np.arcsin(np.clip(Rbn_ekf[2, 0], -1.0, 1.0))
    yaw_k   = np.arctan2(Rbn_ekf[1, 0], Rbn_ekf[0, 0])

    # 2) Acceleration predict
    # INS ile aynı gravity modeli kullanılmalı
    # g_world burada aşağı yönlü bileşeni temsil ediyor
    real_g  = 9.80655
    g_world = np.array([0.0, 0.0, real_g])

    accel_world = Rbn_ekf @ f_hat - g_world

    # 3) State predict
    x_pred = x_ekf.copy()
    x_pred[0:3] = x_ekf[0:3] + dt * x_ekf[3:6]
    x_pred[3:6] = x_ekf[3:6] + dt * accel_world
    x_pred[6]   = roll_k
    x_pred[7]   = pitch_k
    x_pred[8]   = yaw_k

# 4) Jacobian (Durum Geçiş Matrisi Türevleri)
    F = np.eye(n_state)
    
    # Pozisyonun Hıza Göre Türevi
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    # EKSİK OLAN KRİTİK KISIM: Hızın Açılara (Attitude) Göre Türevi
    # Dünya çerçevesindeki ivme (yerçekimi çıkarılmadan önceki saf itki)
    fn_x, fn_y, fn_z = Rbn_ekf @ f_hat

    # İvme vektörünün Skew-Symmetric (Çapraz Çarpım) Matrisi
    skew_f = np.array([
        [   0.0, -fn_z,  fn_y],
        [  fn_z,   0.0, -fn_x],
        [ -fn_y,  fn_x,   0.0]
    ])

    # Açısal hataların hız üzerindeki etkisi: - [f^n]x * dt
    F[3:6, 6:9] = -skew_f * dt

    # 5) Covariance predict
    P = F @ P @ F.T + Q * dt

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

            I_KH = np.eye(n_state) - K @ H
            P = I_KH @ P @ I_KH.T + K @ R_meas @ K.T

        gps_idx += 1

    x_ekf = x_pred
    P_hist[k+1] = P
    ekf_p[k+1]     = x_ekf[0:3]
    ekf_v[k+1]     = x_ekf[3:6]
    ekf_euler[k+1] = np.degrees([x_ekf[6], x_ekf[7], x_ekf[8]])

print(f"\nEKF tamamlandı.")
print(f"Son konum EKF : {ekf_p[-1]}")
print(f"Son konum INS : {p[-1]}")
# ==============================================================================
# GÖRSELLEŞTİRME: INS vs EKF vs GROUND TRUTH
# ==============================================================================
# ══════════════════════════════════════════════════════════════════
# GÖRSELLEŞTİRME İÇİN HAZIRLIK
# ══════════════════════════════════════════════════════════════════

# 1. Ground Truth (GT) verisini numpy dizisine al (tx, ty, tz -> x, y, z)
gt_timestamps = df_gt['timestamp'].values
gt_p_raw = df_gt[['tx', 'ty', 'tz']].values

# 2. GT verisini IMU zaman damgalarına (t_imu) enterpole et
# EKF'nin çalıştığı her an için bir GT karşılığına ihtiyacımız var
gt_p_interp = np.zeros_like(ekf_p)
for i in range(3): # x, y, z için ayrı ayrı
    gt_p_interp[:, i] = np.interp(t_imu, gt_timestamps, gt_p_raw[:, i])

# ══════════════════════════════════════════════════════════════════
# GÖRSELLEŞTİRME: EKF vs GROUND TRUTH
# ══════════════════════════════════════════════════════════════════
plt.figure(figsize=(14, 10))

# 3D Yörünge
ax = plt.subplot(2, 1, 1, projection='3d')
ax.plot(ekf_p[:, 0], ekf_p[:, 1], ekf_p[:, 2], label='EKF Estimate', color='blue', lw=1.2)
ax.plot(gt_p_interp[:, 0], gt_p_interp[:, 1], gt_p_interp[:, 2], label='Ground Truth', color='green', linestyle='--', lw=1.2)
ax.set_title("3D Trajectory Comparison (EKF vs Ground Truth)")
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
ax.legend()

# 2D Pozisyon (X, Y, Z ayrı ayrı)
plt.subplot(2, 1, 2)
plt.plot(t_imu, ekf_p[:, 0], label='EKF X', color='blue')
plt.plot(t_imu, gt_p_interp[:, 0], label='GT X', color='blue', linestyle=':')
plt.plot(t_imu, ekf_p[:, 1], label='EKF Y', color='orange')
plt.plot(t_imu, gt_p_interp[:, 1], label='GT Y', color='orange', linestyle=':')
plt.plot(t_imu, ekf_p[:, 2], label='EKF Z', color='red')
plt.plot(t_imu, gt_p_interp[:, 2], label='GT Z', color='red', linestyle=':')
plt.title("Position Comparison (Time Domain)")
plt.xlabel("Time (s)"); plt.ylabel("Position (m)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



# ==========================================================
# ERROR + 3SIGMA PLOTS FOR ALL 9 STATES
# ==========================================================

t_rel = t_imu - t_imu[0]

# ----------------------------------------------------------
# Ground truth interpolation onto IMU timestamps
# ----------------------------------------------------------
gt_timestamps = df_gt['timestamp'].values
gt_p_raw = df_gt[['tx', 'ty', 'tz']].values
gt_q_raw = df_gt[['qx', 'qy', 'qz', 'qw']].values

# Position truth
gt_p_interp = np.zeros((n, 3))
for i in range(3):
    gt_p_interp[:, i] = np.interp(t_imu, gt_timestamps, gt_p_raw[:, i])

# Velocity truth from interpolated position
gt_v_interp = np.zeros((n, 3))
for i in range(3):
    gt_v_interp[:, i] = np.gradient(gt_p_interp[:, i], t_imu)

# Attitude truth from interpolated quaternions
gt_q_interp = np.zeros((n, 4))
for i in range(4):
    gt_q_interp[:, i] = np.interp(t_imu, gt_timestamps, gt_q_raw[:, i])

gt_q_interp = gt_q_interp / np.linalg.norm(gt_q_interp, axis=1, keepdims=True)

gt_euler_interp = np.zeros((n, 3))
for i in range(n):
    gt_euler_interp[i, :] = R.from_quat(gt_q_interp[i]).as_euler('xyz', degrees=True)

# ----------------------------------------------------------
# Helper: wrapped angle difference in degrees
# ----------------------------------------------------------
def angle_diff_deg(est_deg, true_deg):
    return (est_deg - true_deg + 180.0) % 360.0 - 180.0

# ----------------------------------------------------------
# Estimated state vector
# ----------------------------------------------------------
x_est = np.zeros((n, 9))
x_est[:, 0:3] = ekf_p
x_est[:, 3:6] = ekf_v
x_est[:, 6:9] = ekf_euler   # degrees

x_true = np.zeros((n, 9))
x_true[:, 0:3] = gt_p_interp
x_true[:, 3:6] = gt_v_interp
x_true[:, 6:9] = gt_euler_interp   # degrees

# ----------------------------------------------------------
# Errors
# ----------------------------------------------------------
err_plot = np.zeros_like(x_est)
err_plot[:, 0:3] = x_est[:, 0:3] - x_true[:, 0:3]
err_plot[:, 3:6] = x_est[:, 3:6] - x_true[:, 3:6]

# attitude errors: wrapped, in degrees
err_plot[:, 6] = angle_diff_deg(x_est[:, 6], x_true[:, 6])
err_plot[:, 7] = angle_diff_deg(x_est[:, 7], x_true[:, 7])
err_plot[:, 8] = angle_diff_deg(x_est[:, 8], x_true[:, 8])

# 3-sigma from covariance diagonal
sigma = np.sqrt(np.maximum(np.diagonal(P_hist, axis1=1, axis2=2), 0.0))
three_sigma_plot = 3.0 * sigma.copy()

# If attitude states in P are stored in radians, convert only sigma band to degrees
three_sigma_plot[:, 6:9] = np.degrees(three_sigma_plot[:, 6:9])

state_names = [
    'North Position (m)',
    'East Position (m)',
    'Up Position (m)',
    'North Velocity (m/s)',
    'East Velocity (m/s)',
    'Up Velocity (m/s)',
    'Roll (deg)',
    'Pitch (deg)',
    'Yaw (deg)'
]

# ----------------------------------------------------------
# Figure 1: estimation errors
# ----------------------------------------------------------
fig1, axes1 = plt.subplots(3, 3, figsize=(18, 14), sharex=True)
axes1 = axes1.ravel()

for i in range(9):
    ax = axes1[i]
    ax.plot(t_rel, err_plot[:, i], color='blue', linewidth=1.0, label='Error')
    ax.set_title(state_names[i])
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right')

for ax in axes1[6:9]:
    ax.set_xlabel('Time (s)')

plt.suptitle('Estimation Errors for All States', fontsize=16)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Figure 2: error with +/- 3 sigma
# ----------------------------------------------------------
fig2, axes2 = plt.subplots(3, 3, figsize=(18, 14), sharex=True)
axes2 = axes2.ravel()

for i in range(9):
    ax = axes2[i]
    ax.plot(t_rel, err_plot[:, i], color='blue', linewidth=1.0, label='Error')
    ax.plot(t_rel, three_sigma_plot[:, i], 'r--', linewidth=1.2, label='+3σ')
    ax.plot(t_rel, -three_sigma_plot[:, i], 'r--', linewidth=1.2, label='-3σ')
    ax.set_title(state_names[i])
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right')

for ax in axes2[6:9]:
    ax.set_xlabel('Time (s)')

plt.suptitle('Estimation Error and ±3σ Bounds', fontsize=16)
plt.tight_layout()
plt.show()













