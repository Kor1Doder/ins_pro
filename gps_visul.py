import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# ==========================================================
# LOAD TIGHT GPS DATA
# ==========================================================

tight = pd.read_csv(
    "./project_src/gps_tightly.txt",
    sep=r"\s+"
)

N_SATS = 8

# ==========================================================
# EARTH
# ==========================================================

R_EARTH = 6378137.0

u = np.linspace(0, 2*np.pi, 80)
v = np.linspace(0, np.pi, 40)

earth_x = R_EARTH * np.outer(np.cos(u), np.sin(v))
earth_y = R_EARTH * np.outer(np.sin(u), np.sin(v))
earth_z = R_EARTH * np.outer(np.ones_like(u), np.cos(v))

# ==========================================================
# RECEIVER TRAJECTORY
# ==========================================================

# Receiver position reconstruction
# NED trajectory already known from GT

gt = pd.read_csv(
    "./project_src/groundtruth.txt",
    sep=r"\s+",
    comment="#",
    header=None,
    names=[
        "timestamp",
        "tx","ty","tz",
        "qx","qy","qz","qw"
    ]
)

# ==========================================================
# WGS84
# ==========================================================

A_WGS84  = 6378137.0
E2_WGS84 = 6.69437999014e-3

REF_LAT = 39.925018
REF_LON = 32.836956
REF_ALT = 850.0

def lla2ecef(lat_deg, lon_deg, alt):

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    N = A_WGS84 / np.sqrt(
        1 - E2_WGS84*np.sin(lat)**2
    )

    x = (N+alt)*np.cos(lat)*np.cos(lon)
    y = (N+alt)*np.cos(lat)*np.sin(lon)
    z = ((1-E2_WGS84)*N+alt)*np.sin(lat)

    return np.array([x,y,z])

def ned2ecef_matrix(lat_deg, lon_deg):

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    sl = np.sin(lat)
    cl = np.cos(lat)

    so = np.sin(lon)
    co = np.cos(lon)

    N = np.array([
        -sl*co,
        -sl*so,
         cl
    ])

    E = np.array([
        -so,
         co,
         0
    ])

    D = np.array([
        -cl*co,
        -cl*so,
        -sl
    ])

    return np.column_stack([N,E,D])

ecef_ref = lla2ecef(
    REF_LAT,
    REF_LON,
    REF_ALT
)

R_n2e = ned2ecef_matrix(
    REF_LAT,
    REF_LON
)

gps_time = tight["timestamp"].values

tx_i = np.interp(
    gps_time,
    gt["timestamp"],
    gt["tx"]
)

ty_i = np.interp(
    gps_time,
    gt["timestamp"],
    gt["ty"]
)

tz_i = np.interp(
    gps_time,
    gt["timestamp"],
    gt["tz"]
)

ned_pos = np.column_stack([
    tx_i,
    ty_i,
    tz_i
])

rx_ecef = (
    ecef_ref[None,:]
    +
    ned_pos @ R_n2e.T
)

# ==========================================================
# FIGURE
# ==========================================================

fig = plt.figure(figsize=(14,12))
ax = fig.add_subplot(
    111,
    projection='3d'
)

# ==========================================================
# ANIMATION
# ==========================================================

def update(frame):

    ax.cla()

    ax.plot_surface(
        earth_x,
        earth_y,
        earth_z,
        alpha=0.25
    )

    # Receiver

    rx = rx_ecef[frame]

    ax.scatter(
        rx[0],
        rx[1],
        rx[2],
        s=120,
        color='red',
        label='Receiver'
    )

    # Receiver trajectory

    ax.plot(
        rx_ecef[:frame+1,0],
        rx_ecef[:frame+1,1],
        rx_ecef[:frame+1,2],
        linewidth=2,
        color='black'
    )

    # Satellites

    for sat in range(1,N_SATS+1):

        sx = tight[f"sat{sat}_x"].iloc[frame]
        sy = tight[f"sat{sat}_y"].iloc[frame]
        sz = tight[f"sat{sat}_z"].iloc[frame]

        elev = tight[
            f"sat{sat}_elev_deg"
        ].iloc[frame]

        color = (
            "green"
            if elev > 10
            else "red"
        )

        ax.scatter(
            sx,
            sy,
            sz,
            color=color,
            s=70
        )

        ax.text(
            sx,
            sy,
            sz,
            f"S{sat}",
            fontsize=8
        )

        ax.plot(
            [rx[0], sx],
            [rx[1], sy],
            [rx[2], sz],
            color=color,
            alpha=0.4
        )

    limit = 3e7

    ax.set_xlim(
        -limit,
         limit
    )

    ax.set_ylim(
        -limit,
         limit
    )

    ax.set_zlim(
        -limit,
         limit
    )

    ax.set_xlabel("ECEF X [m]")
    ax.set_ylabel("ECEF Y [m]")
    ax.set_zlabel("ECEF Z [m]")

    ax.set_title(
        f"GNSS Simulation\n"
        f"Frame={frame}"
    )

    ax.set_box_aspect(
        [1,1,1]
    )

    ax.view_init(
        elev=25,
        azim=frame*0.3
    )

# ==========================================================
# RUN
# ==========================================================

ani = FuncAnimation(
    fig,
    update,
    frames=len(tight),
    interval=50,
    repeat=True
)

plt.show()