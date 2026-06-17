import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation

# ==========================================================
# DATA
# ==========================================================

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

# ----------------------------------------------------------
# HIZLANDIRMA
# ----------------------------------------------------------

df = df.iloc[::20].reset_index(drop=True)

print("Frame sayısı:", len(df))

# ==========================================================
# FIGURE
# ==========================================================

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')

# ==========================================================
# WORLD FRAME
# ==========================================================

world_scale = 1.0

for axis,name in zip(np.eye(3),['Xw','Yw','Zw']):

    ax.quiver(
        0,0,0,
        axis[0]*world_scale,
        axis[1]*world_scale,
        axis[2]*world_scale,
        color='black',
        linewidth=3
    )

    ax.text(
        axis[0]*1.1,
        axis[1]*1.1,
        axis[2]*1.1,
        name,
        fontsize=14,
        fontweight='bold'
    )

# ==========================================================
# TRAJECTORY
# ==========================================================

ax.plot(
    df["tx"],
    df["ty"],
    df["tz"],
    color='gray',
    alpha=0.3,
    linewidth=1
)

# ==========================================================
# LIMITS
# ==========================================================

margin = 1

ax.set_xlim(
    df["tx"].min()-margin,
    df["tx"].max()+margin
)

ax.set_ylim(
    df["ty"].min()-margin,
    df["ty"].max()+margin
)

ax.set_zlim(
    df["tz"].min()-margin,
    df["tz"].max()+margin
)

ax.set_box_aspect([1,1,1])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# ==========================================================
# BODY FRAME HOLDERS
# ==========================================================

body_quivers = []
body_labels = []

drone_point = None

# ==========================================================
# INIT
# ==========================================================

def init():

    global drone_point

    drone_point = ax.scatter(
        [0],[0],[0],
        s=80
    )

    return []




# ==========================================================
# UPDATE
# ==========================================================

def update(frame):

    global drone_point

    for q in body_quivers:
        q.remove()

    for t in body_labels:
        t.remove()

    body_quivers.clear()
    body_labels.clear()

    row = df.iloc[frame]

    tx = row["tx"]
    ty = row["ty"]
    tz = row["tz"]

    qx = row["qx"]
    qy = row["qy"]
    qz = row["qz"]
    qw = row["qw"]

    rot = R.from_quat([qx,qy,qz,qw])

    roll,pitch,yaw = rot.as_euler(
        'xyz',
        degrees=True
    )

    drone_point.remove()

    drone_point = ax.scatter(
        [tx],[ty],[tz],
        s=100
    )

    colors = ['red','green','blue']
    names = ['Xb','Yb','Zb']

    scale = 0.5

    for axis,color,name in zip(
        np.eye(3),
        colors,
        names
    ):

        v = rot.apply(axis)

        qv = ax.quiver(
            tx,
            ty,
            tz,
            v[0]*scale,
            v[1]*scale,
            v[2]*scale,
            color=color,
            linewidth=3
        )

        body_quivers.append(qv)

        txt = ax.text(
            tx + v[0]*scale*1.2,
            ty + v[1]*scale*1.2,
            tz + v[2]*scale*1.2,
            name,
            color=color,
            fontsize=12
        )

        body_labels.append(txt)

    ax.set_title(
        f"Frame {frame}/{len(df)}\n"
        f"Roll={roll:.1f}°   "
        f"Pitch={pitch:.1f}°   "
        f"Yaw={yaw:.1f}°"
    )

    return body_quivers

# ==========================================================
# ANIMATION
# ==========================================================

ani = FuncAnimation(
    fig,
    update,
    frames=len(df),
    init_func=init,
    interval=20,
    blit=False
)

plt.show()