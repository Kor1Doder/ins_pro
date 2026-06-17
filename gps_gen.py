import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================== AYARLAR ==================
GT_FILE   = "./project_src/groundtruth.txt"
OUT_LOOSE = "./project_src/gps_loosely.txt"
OUT_TIGHT = "./project_src/gps_tightly.txt"

GPS_RATE_HZ = 10.0
SEED = 42

# Referans nokta (WGS84)
REF_LAT_DEG = 39.925018
REF_LON_DEG = 32.836956
REF_ALT     = 850.0

# ---- Loosely coupled gürültü parametreleri ----
POS_NOISE_H   = 1   # yatay pozisyon std (m)
POS_NOISE_V   = 2.5    # düşey pozisyon std (m)
VEL_NOISE_STD = 0.05  # hız std (m/s)

# ---- Tightly coupled parametreler ----
N_SATS        = 8
PR_NOISE_STD  = 1.0     # pseudorange noise std (m)
PRR_NOISE_STD = 0.05    # pseudorange-rate noise std (m/s)
CLOCK_BIAS0   = 50.0    # m
CLOCK_DRIFT0  = 0.5     # m/s
ORBIT_RADIUS  = 26560e3 # m
ORBIT_PERIOD  = 43082.0 # s
# ============================================

rng = np.random.default_rng(SEED)

A_WGS84  = 6378137.0
E2_WGS84 = 6.69437999014e-3


def lla2ecef(lat_deg, lon_deg, alt):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    N = A_WGS84 / np.sqrt(1 - E2_WGS84 * np.sin(lat) ** 2)

    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - E2_WGS84) + alt) * np.sin(lat)
    return np.array([x, y, z])


def neu2ecef_matrix(lat_deg, lon_deg):
    """
    ECEF <- NEU
    Columns are the ECEF directions of:
      North, East, Up
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)

    N = np.array([-sl * co, -sl * so,  cl])
    E = np.array([-so,        co,      0.0])
    U = np.array([ cl * co,   cl * so, sl])

    return np.column_stack([N, E, U])


def normalize(v):
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero vector cannot be normalized.")
    return v / n


def perp_basis(v):
    v = normalize(v)
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(v, a)) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    e1 = normalize(np.cross(v, a))
    e2 = np.cross(v, e1)
    return e1, e2


# ==========================================================
# 1) GROUND TRUTH OKU (NEU)
# ==========================================================
df = pd.read_csv(
    GT_FILE,
    sep=r"\s+",
    comment="#",
    header=None,
    names=["timestamp", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
)

df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
df["tx"] = pd.to_numeric(df["tx"], errors="coerce")
df["ty"] = pd.to_numeric(df["ty"], errors="coerce")
df["tz"] = pd.to_numeric(df["tz"], errors="coerce")
df = df.dropna(subset=["timestamp", "tx", "ty", "tz"]).sort_values("timestamp").reset_index(drop=True)

t  = df["timestamp"].to_numpy(dtype=np.float64)
n  = df["tx"].to_numpy(dtype=np.float64)   # North
e  = df["ty"].to_numpy(dtype=np.float64)   # East
u  = df["tz"].to_numpy(dtype=np.float64)   # Up

dt_gps = 1.0 / GPS_RATE_HZ
gps_time = np.arange(t[0], t[-1], dt_gps)

print(f"Groundtruth zaman aralığı : {t[0]:.6f} -> {t[-1]:.6f} s")
print(f"GPS örnek sayısı           : {len(gps_time)}")

# GT'yi GPS zamanlarına interpolate et
n_i = np.interp(gps_time, t, n)
e_i = np.interp(gps_time, t, e)
u_i = np.interp(gps_time, t, u)

# NEU hızları
vn_true = np.gradient(n_i, gps_time)
ve_true = np.gradient(e_i, gps_time)
vu_true = np.gradient(u_i, gps_time)

M = len(gps_time)

# ==========================================================
# 2) LOOSELY COUPLED GPS ÜRETİMİ (NEU -> LLA)
# ==========================================================
R_EARTH = A_WGS84

lat_true = REF_LAT_DEG + (n_i / R_EARTH) * (180.0 / np.pi)
lon_true = REF_LON_DEG + (e_i / (R_EARTH * np.cos(np.radians(REF_LAT_DEG)))) * (180.0 / np.pi)
alt_true = REF_ALT + u_i   # NEU: Up pozitif

lat_noise_deg = (POS_NOISE_H / R_EARTH) * (180.0 / np.pi)
lon_noise_deg = (POS_NOISE_H / (R_EARTH * np.cos(np.radians(REF_LAT_DEG)))) * (180.0 / np.pi)

lat_meas = lat_true + rng.normal(0, lat_noise_deg, size=M)
lon_meas = lon_true + rng.normal(0, lon_noise_deg, size=M)
alt_meas = alt_true + rng.normal(0, POS_NOISE_V, size=M)

vn_meas = vn_true + rng.normal(0, VEL_NOISE_STD, size=M)
ve_meas = ve_true + rng.normal(0, VEL_NOISE_STD, size=M)
vu_meas = vu_true + rng.normal(0, VEL_NOISE_STD, size=M)

loose = pd.DataFrame({
    "timestamp": gps_time,
    "lat": lat_meas,
    "lon": lon_meas,
    "alt": alt_meas,
    "vn": vn_meas,
    "ve": ve_meas,
    "vu": vu_meas,
    "pos_std_h": POS_NOISE_H,
    "pos_std_v": POS_NOISE_V,
    "vel_std": VEL_NOISE_STD,
})

loose.to_csv(OUT_LOOSE, sep=" ", index=False, float_format="%.9f")
print(f"[Loose] kaydedildi -> {OUT_LOOSE} ({len(loose)} satır)")

# ==========================================================
# 3) TIGHTLY COUPLED GPS ÜRETİMİ
#    NEU receiver konumu/hızı -> ECEF
# ==========================================================
ecef_ref = lla2ecef(REF_LAT_DEG, REF_LON_DEG, REF_ALT)
R_neu2ecef = neu2ecef_matrix(REF_LAT_DEG, REF_LON_DEG)
zenith = normalize(ecef_ref)

neu_pos = np.column_stack([n_i, e_i, u_i])            # (M,3)
neu_vel = np.column_stack([vn_true, ve_true, vu_true])# (M,3)

rx_pos_ecef = ecef_ref[None, :] + neu_pos @ R_neu2ecef.T
rx_vel_ecef = neu_vel @ R_neu2ecef.T

omega = 2.0 * np.pi / ORBIT_PERIOD
theta = omega * (gps_time - gps_time[0])

clock_bias  = CLOCK_BIAS0 + CLOCK_DRIFT0 * (gps_time - gps_time[0])
clock_drift = np.full(M, CLOCK_DRIFT0)

e1, e2 = perp_basis(zenith)

tight_data = {"timestamp": gps_time}

for i in range(N_SATS):
    psi = np.radians(rng.uniform(10.0, 65.0))
    az  = rng.uniform(0.0, 2.0 * np.pi)

    rot_axis = np.cos(az) * e1 + np.sin(az) * e2
    r0 = normalize(zenith * np.cos(psi) + rot_axis * np.sin(psi))

    normal = normalize(np.cross(r0, rot_axis))
    b0 = normalize(np.cross(normal, r0))

    sat_dir = np.outer(np.cos(theta), r0) + np.outer(np.sin(theta), b0)
    sat_pos = ORBIT_RADIUS * sat_dir
    sat_vel = ORBIT_RADIUS * omega * (
        -np.outer(np.sin(theta), r0) + np.outer(np.cos(theta), b0)
    )

    los_vec = sat_pos - rx_pos_ecef
    rho_true = np.linalg.norm(los_vec, axis=1)
    u_los = los_vec / rho_true[:, None]

    rel_vel = sat_vel - rx_vel_ecef
    rho_dot_true = np.sum(u_los * rel_vel, axis=1)

    elevation = np.degrees(np.arcsin(np.sum(u_los * zenith, axis=1)))

    pr  = rho_true + clock_bias + rng.normal(0, PR_NOISE_STD, size=M)
    prr = rho_dot_true + clock_drift + rng.normal(0, PRR_NOISE_STD, size=M)

    tag = f"sat{i+1}"
    tight_data[f"pr_{tag}"] = pr
    tight_data[f"prr_{tag}"] = prr
    tight_data[f"{tag}_x"] = sat_pos[:, 0]
    tight_data[f"{tag}_y"] = sat_pos[:, 1]
    tight_data[f"{tag}_z"] = sat_pos[:, 2]
    tight_data[f"{tag}_vx"] = sat_vel[:, 0]
    tight_data[f"{tag}_vy"] = sat_vel[:, 1]
    tight_data[f"{tag}_vz"] = sat_vel[:, 2]
    tight_data[f"{tag}_elev_deg"] = elevation

tight_data["clock_bias_m"] = clock_bias
tight_data["clock_drift_mps"] = clock_drift

tight = pd.DataFrame(tight_data)
tight.to_csv(OUT_TIGHT, sep=" ", index=False, float_format="%.9f")
print(f"[Tight] kaydedildi -> {OUT_TIGHT} ({len(tight)} satır, {N_SATS} uydu)")

# ==========================================================
# 4) PLOTLAR
# ==========================================================
gps_df = pd.read_csv(OUT_LOOSE, sep=r"\s+")

# LLA -> NEU yaklaşık dönüşüm
gps_n = np.radians(gps_df["lat"] - REF_LAT_DEG) * R_EARTH
gps_e = np.radians(gps_df["lon"] - REF_LON_DEG) * R_EARTH * np.cos(np.radians(REF_LAT_DEG))
gps_u = gps_df["alt"] - REF_ALT

# -------- 3D TRAJECTORY --------
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

ax.plot(n, e, u, color="green", linewidth=3, label="Ground Truth (NEU)")
ax.plot(gps_n, gps_e, gps_u, color="red", linewidth=1.5, alpha=0.8, label="Loose GPS")

ax.scatter(n[0], e[0], u[0], color="black", s=80, marker="o", label="Start")
ax.scatter(n[-1], e[-1], u[-1], color="green", s=80, marker="^", label="GT End")
ax.scatter(gps_n.iloc[-1], gps_e.iloc[-1], gps_u.iloc[-1], color="red", s=80, marker="^", label="GPS End")

ax.set_xlabel("North [m]")
ax.set_ylabel("East [m]")
ax.set_zlabel("Up [m]")
ax.set_title(f"Ground Truth vs Loose GPS ({GPS_RATE_HZ} Hz)")
ax.legend()
ax.grid(True)

all_x = np.concatenate([n, gps_n.values])
all_y = np.concatenate([e, gps_e.values])
all_z = np.concatenate([u, gps_u.values])

max_range = max(np.ptp(all_x), np.ptp(all_y), np.ptp(all_z))
mid_x = (all_x.max() + all_x.min()) / 2
mid_y = (all_y.max() + all_y.min()) / 2
mid_z = (all_z.max() + all_z.min()) / 2

ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

plt.tight_layout()
plt.show()

# -------- TOP VIEW --------
plt.figure(figsize=(10, 10))
plt.plot(n, e, color="green", linewidth=3, label="Ground Truth")
plt.plot(gps_n, gps_e, color="red", alpha=0.7, label="Loose GPS")
plt.xlabel("North [m]")
plt.ylabel("East [m]")
plt.title("Top View (N-E Plane)")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()

# -------- TIGHT SANITY CHECK --------
tight_df = pd.read_csv(OUT_TIGHT, sep=r"\s+")
plt.figure(figsize=(10, 5))
plt.plot(gps_time, tight_df["pr_sat1"], label="Pseudorange sat1")
plt.xlabel("Time [s]")
plt.ylabel("Range [m]")
plt.title("Tight GPS Sanity Check - Pseudorange (sat1)")
plt.grid(True)
plt.legend()
plt.show()

print("GPS generation tamamlandı.")