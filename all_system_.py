import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import math

# ============================================================
# LOAD GROUND TRUTH
# ============================================================

df_gt = pd.read_csv(
    "project_src/groundtruth.txt",
    sep=r"\s+",
    comment="#",
    header=None,
    names=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
)
df_gt.sort_values('timestamp', inplace=True)

# ============================================================
# LOAD IMU
# ============================================================

df_imu_raw = pd.read_csv(
    "project_src/imu.txt",
    sep=r"\s+",
    comment="#",
    header=None,
    names=["timestamp", "ang_vel_x", "ang_vel_y", "ang_vel_z",
           "lin_acc_x", "lin_acc_y", "lin_acc_z"]
)
df_imu_raw.sort_values('timestamp', inplace=True)

print(f"IMU  raw : {df_imu_raw['timestamp'].iloc[0]:.3f} -> {df_imu_raw['timestamp'].iloc[-1]:.3f}  ({len(df_imu_raw)} samples)")
print(f"GT   raw : {df_gt['timestamp'].iloc[0]:.3f} -> {df_gt['timestamp'].iloc[-1]:.3f}  ({len(df_gt)} samples)")

# ============================================================
# STATIC SEGMENT — BIAS ESTIMATION
# Use the IMU data BEFORE GT starts (drone is stationary on ground)
# This is the natural static window: IMU start -> GT start
# ============================================================

t_gt_start = df_gt['timestamp'].iloc[0]   # e.g. 4908.79
t_imu_start = df_imu_raw['timestamp'].iloc[0]  # e.g. 4878.79 

df_static = df_imu_raw[df_imu_raw['timestamp'] < 4886.5].copy()
print(f"\nStatic segment: {t_imu_start:.3f} -> {t_gt_start:.3f}  ({len(df_static)} samples)")

# Gyro bias — keep in rad/s (same unit as raw IMU data)
gyro_bias_x = df_static['ang_vel_x'].mean()
gyro_bias_y = df_static['ang_vel_y'].mean()
gyro_bias_z = df_static['ang_vel_z'].mean()
print(f"Gyro bias (rad/s) : bx={gyro_bias_x:.6f}  by={gyro_bias_y:.6f}  bz={gyro_bias_z:.6f}")

# Gyro statistics for display (deg/s)
gyro_stats = (df_static[['ang_vel_x', 'ang_vel_y', 'ang_vel_z']] * (180/np.pi))\
             .agg(['mean', 'std', 'min', 'max']).transpose()
print("\nGyro bias statistics (deg/s):\n", gyro_stats)

# ============================================================
# INITIAL ORIENTATION from first GT quaternion
# ============================================================

q0   = df_gt.iloc[0][["qx", "qy", "qz", "qw"]].values
rot0 = R.from_quat(q0)
roll_init, pitch_init, yaw_init = rot0.as_euler('xyz', degrees=True)
Rmat_init = rot0.as_matrix()
print(f"\nInitial orientation : roll={roll_init:.4f}  pitch={pitch_init:.4f}  yaw={yaw_init:.4f}  (deg)")

# ============================================================
# WORLD / BODY FRAME VISUALIZATION
# ============================================================

fig = plt.figure(figsize=(16, 10))
ax  = fig.add_subplot(111, projection='3d')

for axis, name in zip(np.eye(3), ["Xw", "Yw", "Zw"]):
    ax.quiver(0, 0, 0, *axis, color='black', linewidth=4)
    ax.text(*(axis*1.15), name, fontsize=18, color='black', fontweight='bold')

for axis, color, name in zip(np.eye(3), ['red','green','blue'], ['Xb','Yb','Zb']):
    v = rot0.apply(axis)
    ax.quiver(0, 0, 0, *v, color=color, linewidth=4)
    ax.text(*(v*1.15), name, fontsize=18, color=color, fontweight='bold')

ax.text(0.25, 0.25, -0.35, "WORLD FRAME\n(FIXED / GLOBAL)", fontsize=12,
        bbox=dict(facecolor='white', edgecolor='black'))
ax.text(0.6, 0.25, 0.6, "DRONE BODY FRAME", fontsize=12, color='blue',
        bbox=dict(facecolor='white', edgecolor='blue'))
ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
ax.set_box_aspect([1,1,1]); ax.grid(True); ax.view_init(elev=25, azim=45)

fig.text(0.76, 0.73,
         f"QUATERNION\n\nqx = {q0[0]:.4f}\nqy = {q0[1]:.4f}\nqz = {q0[2]:.4f}\nqw = {q0[3]:.4f}",
         fontsize=12, family='monospace',
         bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.5'))
fig.text(0.76, 0.52,
         f"EULER ANGLES (XYZ)\n\nRoll  = {roll_init:.2f} deg\nPitch = {pitch_init:.2f} deg\nYaw   = {yaw_init:.2f} deg",
         fontsize=12, family='monospace',
         bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.5'))
fig.text(0.76, 0.25,
         "ROTATION MATRIX\n\n" + np.array2string(Rmat_init, precision=3, suppress_small=True),
         fontsize=10, family='monospace',
         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
fig.text(0.05, 0.08,
         "WORLD FRAME\nBlack = Fixed Frame\n\nBODY FRAME\nRed   = Xb\nGreen = Yb\nBlue  = Zb",
         fontsize=11, bbox=dict(facecolor='white', edgecolor='black'))
plt.suptitle("World (Fixed) vs Drone (Body) Frame", fontsize=20, fontweight='bold')
plt.tight_layout()

# ============================================================
# GRAVITY
# ============================================================

def calc_g(gx=0.078163, gy=-9.27130891, gz=-3.1945492):
    return math.sqrt(gx**2 + gy**2 + gz**2)

real_g  = calc_g()
g_world = np.array([0.0, 0.0, real_g])
print(f"Gravity magnitude : {real_g:.4f} m/s^2")

# ============================================================
# ROTATION MATRIX HELPER
# ============================================================

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

 

# ============================================================
# ACCEL BIAS ESTIMATION
# While stationary: measured_accel = g_body + sensor_bias
# => sensor_bias = mean(static_accel) - R_init.T @ g_world
# ============================================================

R_init     = rot_matrix(roll_init, pitch_init, yaw_init)
g_body0    = R_init.T @ g_world
accel_mean = df_static[['lin_acc_x','lin_acc_y','lin_acc_z']].mean().values
accel_bias = accel_mean - g_body0
print(f"Accel bias (m/s^2): bx={accel_bias[0]:.6f}  by={accel_bias[1]:.6f}  bz={accel_bias[2]:.6f}")

# ============================================================
# CUT IMU TO GT WINDOW — this is the key fix
# Only keep IMU samples that fall within the GT time window.
# No more manual index offsets like range(15002, n).
# ============================================================

t_gt_end = 4940#df_gt['timestamp'].loc[-1]

df_imu = df_imu_raw[
    (df_imu_raw['timestamp'] >= t_gt_start) &
    (df_imu_raw['timestamp'] <= t_gt_end)
].copy()

# Shift both to t=0 from GT start
df_imu['timestamp'] -= t_gt_start
df_gt  = df_gt.copy()
df_gt ['timestamp'] -= t_gt_start

df_imu.set_index('timestamp', inplace=True)
df_gt .set_index('timestamp', inplace=True)

print(f"\nAligned IMU : 0.000 -> {df_imu.index[-1]:.3f} s  ({len(df_imu)} samples)")
print(f"Aligned GT  : 0.000 -> {df_gt.index[-1]:.3f} s  ({len(df_gt)} samples)")

# ============================================================
# APPLY BIAS CORRECTION TO ALIGNED IMU
# Only sensor bias removed here. Gravity removed per step inside loop.
# ============================================================

df_imu['ang_vel_x'] -= gyro_bias_x   # rad/s - rad/s
df_imu['ang_vel_y'] -= gyro_bias_y
df_imu['ang_vel_z'] -= gyro_bias_z
df_imu['lin_acc_x'] -= accel_bias[0]
df_imu['lin_acc_y'] -= accel_bias[1]
df_imu['lin_acc_z'] -= accel_bias[2]

# ============================================================
# GYRO PLOTS (static segment, display in deg/s)
# ============================================================

gyro_deg = df_static[['ang_vel_x','ang_vel_y','ang_vel_z']].copy()
gyro_deg.index = df_static['timestamp'] - t_imu_start
gyro_deg *= (180/np.pi)

fig_g, axes_g = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

for idx, (col, color, lbl) in enumerate(zip(
        ['ang_vel_x','ang_vel_y','ang_vel_z'], ['red','green','blue'], ['X','Y','Z'])):
    axes_g[idx].plot(gyro_deg.index, gyro_deg[col], color=color, linewidth=0.5)
    axes_g[idx].set_title(f'Angular Velocity - {lbl} Axis (Static Segment)')
    axes_g[idx].set_ylabel('deg/s'); axes_g[idx].grid(True, linestyle='--', alpha=0.7)
axes_g[2].set_xlabel('Elapsed Time (s)')
plt.tight_layout()

# ============================================================
# INS INTEGRATION LOOP
# Starts at i=1, always — no manual offset needed
# ============================================================

n        = len(df_imu)
imu_vals = df_imu.values          # wx wy wz ax ay az
t_imu    = df_imu.index.values    # elapsed time (seconds from GT start)
dt_arr   = np.diff(t_imu)

rx = np.zeros(n); ry = np.zeros(n); rz = np.zeros(n)
vx = np.zeros(n); vy = np.zeros(n); vz = np.zeros(n)
roll_arr  = np.zeros(n)
pitch_arr = np.zeros(n)
yaw_arr   = np.zeros(n)

# Initial conditions from GT first sample
rx[0] = df_gt.iloc[0]['tx']
ry[0] = df_gt.iloc[0]['ty']
rz[0] = df_gt.iloc[0]['tz']
roll_arr[0]  = roll_init
pitch_arr[0] = pitch_init
yaw_arr[0]   = yaw_init

R_prev = rot_matrix(roll_init, pitch_init, yaw_init)

# Tüm flight boyunca accel_world Z ortalaması nedir?
accel_z_all = []
R_test = rot_matrix(roll_init, pitch_init, yaw_init)

# RK4 Döngüsü (her i adımı için)
def rk4_step(v_prev, r_prev, a_current, dt):
    # k1 = f(state_prev)
    k1_v = a_current
    k1_r = v_prev
    
    # k2 = f(state_prev + 0.5*dt*k1)
    k2_v = a_current
    k2_r = v_prev + 0.5 * dt * k1_v
    
    # k3 = f(state_prev + 0.5*dt*k2)
    k3_v = a_current
    k3_r = v_prev + 0.5 * dt * k2_v
    
    # k4 = f(state_prev + dt*k3)
    k4_v = a_current
    k4_r = v_prev + dt * k3_v
    
    # Final güncelleme
    v_new = v_prev + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    r_new = r_prev + (dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
    
    return v_new, r_new
for i in range(len(df_imu)):
    sf = df_imu.values[i, 3:6]
    a_world = R_test @ sf - g_world
    accel_z_all.append(a_world[2])

 
 
for i in range(1, n):
    dt = float(dt_arr[i-1])
    if dt <= 0 or dt > 0.1:
        dt = 0.005

    wx  = imu_vals[i-1, 0]; wy  = imu_vals[i-1, 1]; wz  = imu_vals[i-1, 2]
    ax_ = imu_vals[i-1, 3]; ay_ = imu_vals[i-1, 4]; az_ = imu_vals[i-1, 5]

    # Skew-symmetric angular velocity matrix
    phi = np.array([[ 0,  -wz,  wy],
                    [ wz,  0,  -wx],
                    [-wy,  wx,   0]])

    # Propagate rotation matrix
    R_new = R_prev @ (np.eye(3) + phi * dt)

    # Re-orthogonalize via SVD
    U, _, Vt = np.linalg.svd(R_new)
    R_new = U @ Vt

    # Transform specific force to world frame, subtract gravity
    spec_force  = np.array([ax_, ay_, az_])
    accel_world = R_new @ spec_force - g_world

    # Velocity update
    vx[i],rx[i]=rk4_step(vx[i-1],rx[i-1],accel_world[0],dt)
    vy[i],ry[i]=rk4_step(vy[i-1],ry[i-1],accel_world[1],dt)
    vz[i],rz[i]=rk4_step(vz[i-1],rz[i-1],accel_world[2],dt)

    '''
    vx[i] = vx[i-1] + accel_world[0] * dt
    vy[i] = vy[i-1] + accel_world[1] * dt
    vz[i] = vz[i-1] + accel_world[2] * dt

    # Position update (trapezoidal)
    rx[i] = rx[i-1] + 0.5*(vx[i] + vx[i-1]) * dt
    ry[i] = ry[i-1] + 0.5*(vy[i] + vy[i-1]) * dt
    rz[i] = rz[i-1] + 0.5*(vz[i] + vz[i-1]) * dt
    '''
    # Euler angles from rotation matrix
    pitch_arr[i] = np.degrees(-np.arcsin(np.clip(R_new[2,0], -1.0, 1.0)))
    roll_arr[i]  = np.degrees(np.arctan2(R_new[2,1], R_new[2,2]))
    yaw_arr[i]   = np.degrees(np.arctan2(R_new[1,0], R_new[0,0]))

    R_prev = R_new  
    
print("INS loop complete.")

# ============================================================
# INTERPOLATE GT ONTO IMU TIMESTAMPS (fair 1-to-1 comparison)
# ============================================================

t_gt = df_gt.index.values

def interp_gt(col):
    f = interp1d(t_gt, df_gt[col].values, kind='linear',
                 bounds_error=False, fill_value='extrapolate')
    return f(t_imu)

gt_x = interp_gt('tx')
gt_y = interp_gt('ty')
gt_z = interp_gt('tz')

# ============================================================
# STATE DATAFRAME
# ============================================================

df_state = pd.DataFrame({
    'timestamp': t_imu,
    'rx': rx, 'ry': ry, 'rz': rz,
    'vx': vx, 'vy': vy, 'vz': vz,
    'roll': roll_arr, 'pitch': pitch_arr, 'yaw': yaw_arr,
    'gt_x': gt_x, 'gt_y': gt_y, 'gt_z': gt_z
})

# ============================================================
# 3D TRAJECTORY COMPARISON
# ============================================================

fig3d = plt.figure(figsize=(12, 9))
ax3d  = fig3d.add_subplot(111, projection='3d')
ax3d.plot(rx,   ry,   rz,   color='red',   linewidth=1.0, alpha=0.8, label='INS Estimate')
ax3d.plot(gt_x, gt_y, gt_z, color='green', linewidth=2.0, linestyle='--', label='Ground Truth')
ax3d.scatter(rx[0], ry[0], rz[0], color='black', s=80, zorder=5, label='Start')
ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
ax3d.set_title('INS Estimate vs Ground Truth — 3D Trajectory')
ax3d.legend(); ax3d.grid(True)
plt.tight_layout()

# ============================================================
# 2D AXIS-BY-AXIS COMPARISON (aligned timestamps)
# ============================================================

fig2d, axes2d = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

axes2d[0].plot(t_imu, rx,   color='red',   linewidth=0.8, label='INS X')
axes2d[0].plot(t_imu, gt_x, color='green', linewidth=1.5, linestyle='--', label='GT X')
axes2d[0].set_ylabel('X (m)'); axes2d[0].legend(); axes2d[0].grid(True, alpha=0.5)
axes2d[0].set_title('Position Comparison (aligned timestamps)')

axes2d[1].plot(t_imu, ry,   color='red',   linewidth=0.8, label='INS Y')
axes2d[1].plot(t_imu, gt_y, color='green', linewidth=1.5, linestyle='--', label='GT Y')
axes2d[1].set_ylabel('Y (m)'); axes2d[1].legend(); axes2d[1].grid(True, alpha=0.5)

axes2d[2].plot(t_imu, rz,   color='red',   linewidth=0.8, label='INS Z')
axes2d[2].plot(t_imu, gt_z, color='green', linewidth=1.5, linestyle='--', label='GT Z')
axes2d[2].set_ylabel('Z (m)'); axes2d[2].set_xlabel('Elapsed Time (s)')
axes2d[2].legend(); axes2d[2].grid(True, alpha=0.5)
plt.tight_layout()

# ============================================================
# POSITION ERROR PLOT
# ============================================================

err_x  = rx   - gt_x
err_y  = ry   - gt_y
err_z  = rz   - gt_z
err_3d = np.sqrt(err_x**2 + err_y**2 + err_z**2)

fig_err, axes_err = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
axes_err[0].plot(t_imu, err_x,  color='red',   linewidth=0.8)
axes_err[0].set_ylabel('Error X (m)'); axes_err[0].grid(True, alpha=0.5)
axes_err[0].set_title('INS Position Error (INS - Ground Truth)')

axes_err[1].plot(t_imu, err_y,  color='green', linewidth=0.8)
axes_err[1].set_ylabel('Error Y (m)'); axes_err[1].grid(True, alpha=0.5)

axes_err[2].plot(t_imu, err_z,  color='blue',  linewidth=0.8)
axes_err[2].set_ylabel('Error Z (m)'); axes_err[2].grid(True, alpha=0.5)

axes_err[3].plot(t_imu, err_3d, color='black', linewidth=0.8)
axes_err[3].set_ylabel('3D Error (m)'); axes_err[3].set_xlabel('Elapsed Time (s)')
axes_err[3].grid(True, alpha=0.5)
plt.tight_layout()

plt.show()

print(f"\nFinal 3D position error : {err_3d[-1]:.3f} m")
print(f"Mean  3D position error : {err_3d.mean():.3f} m")