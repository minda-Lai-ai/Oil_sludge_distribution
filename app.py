import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import io
import matplotlib.font_manager as fm

# 中文字型配置（多備一個總會有一種中！）
fontname = 'Noto Sans CJK TC'  # 或你系統有的字型
plt.rcParams['font.sans-serif'] = [fontname, 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 在 streamlit 主頁加入 ---
st.markdown("""
### 說明
本程式可視化油槽內部油泥量之3D分布結果。支援 Excel 上傳與手動輸入座標，並可自訂油槽參數、柱狀圖顏色與透明度。  
操作方式：
- 設定左側參數與數據
- 點擊「執行」可動態出圖，右側可下載 PNG 圖檔與原始數據

### 作者聯絡
- 作者：Minda
- 若此工具對你有幫助或遇到任何問題，歡迎聯絡/反饋！
""")

# 頁面參數
st.set_page_config(page_title="油槽內油泥分布圖", layout="wide")
st.sidebar.header("油槽參數設定")
tank_name = st.sidebar.text_input("油槽名稱", value="油槽油泥分布圖")
radius = st.sidebar.number_input("油槽半徑(公尺)", value=45.73, min_value=0.1)
height = st.sidebar.number_input("油槽高度(公尺)", value=3.0, min_value=0.1)
grid_points = st.sidebar.slider("解析度(個點數)", min_value=10, max_value=100, value=50)
elev = st.sidebar.slider("仰角(度)", min_value=0, max_value=90, value=30)
azim = st.sidebar.slider("方位角(度)", min_value=0, max_value=360, value=180)
show_labels = st.sidebar.checkbox("顯示數據標籤", value=True)
frame_alpha = st.sidebar.slider("油槽骨架透明度", min_value=0.0, max_value=1.0, value=0.5)
bar_edgecolor = st.sidebar.selectbox("柱狀線條顏色", options=['gray', 'black', 'red', 'blue'], index=0)
run_btn = st.sidebar.button("執行 → 產生 3D 油槽圖")

st.title("油槽內油泥分布圖")

plt.rcParams['font.family'] = 'Microsoft JhengHei'  # 請根據自己的系統字型調整

upload_opt = st.radio("數據輸入方式：", ["上傳 EXCEL 檔", "手動輸入數據"])
data = None
if upload_opt == "上傳 EXCEL 檔":
    uploaded_file = st.file_uploader("請上傳 oil_sludge_measurements.xlsx", type=["xlsx"])
    if uploaded_file:
        try:
            data = pd.read_excel(uploaded_file)
            st.success("檔案讀取成功！")
        except Exception as e:
            st.error(f"檔案讀取失敗：{str(e)}")
else:
    st.text("手動輸入 24 組 X, Y, Z 資料。欄數不足可上下拉動補齊。")
    input_data = st.data_editor(pd.DataFrame({"X": [0.0]*24, "Y": [0.0]*24, "Z": [0.0]*24}), num_rows="fixed")
    data = input_data

if (data is not None) and run_btn:
    x = data['X'].values
    y = data['Y'].values
    z = data['Z'].values

    # XY視覺比例加寬
    scale_xy = 4.0
    scaled_radius = radius * scale_xy
    scaled_x = x * scale_xy
    scaled_y = y * scale_xy

    grid_x, grid_y = np.meshgrid(
        np.linspace(-scaled_radius, scaled_radius, grid_points),
        np.linspace(-scaled_radius, scaled_radius, grid_points)
    )
    grid_z = griddata((scaled_x, scaled_y), z, (grid_x, grid_y), method='cubic')
    grid_z[grid_x ** 2 + grid_y ** 2 > scaled_radius ** 2] = np.nan

    colors = ['darkblue', 'deepskyblue', 'yellow', 'red']
    positions = [0, 0.2, 0.8, 1]
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(positions, colors)), N=256)
    z_min = np.nanmin(z)
    z_max = np.nanmax(z)
    dx = dy = (2 * scaled_radius) / grid_points

    # ===== 圖表區域加寬空間，左main圖右配色bar =====
    fig = plt.figure(figsize=(17, 10))
    ax = fig.add_subplot(121, projection='3d')

    # 柱狀分布
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            if not np.isnan(grid_z[i, j]):
                x_pos = grid_x[i, j]
                y_pos = grid_y[i, j]
                z_pos = 0
                dz = grid_z[i, j]
                if x_pos**2 + y_pos**2 <= scaled_radius**2:
                    ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz,
                             color=custom_cmap((dz - z_min) / (z_max - z_min)),
                             edgecolor=bar_edgecolor, linewidth=0.5)
    # 油槽骨架（透明度可調）
    n_cylinder = 100
    theta = np.linspace(0, 2*np.pi, n_cylinder)
    cx = scaled_radius * np.cos(theta)
    cy = scaled_radius * np.sin(theta)
    ax.plot(cx, cy, [height] * n_cylinder, color='black', linewidth=2, alpha=frame_alpha)
    ax.plot(cx, cy, [0] * n_cylinder, color='black', linewidth=2, alpha=frame_alpha)
    for t_deg in np.arange(0, 360, 45):
        t_rad = np.radians(t_deg)
        ax.plot([scaled_radius * np.cos(t_rad)]*2, [scaled_radius * np.sin(t_rad)]*2,
                [0, height], color='gray', linewidth=1.2, alpha=frame_alpha)

    # 東南西北
    compass = {'北':'北', '東':'東', '南':'南', '西':'西'}
    a1_angle_rad = np.arctan2(scaled_y[0], scaled_x[0]) if len(scaled_x) > 0 else 0
    base_deg = np.degrees(a1_angle_rad) % 360
    offsets = {'北':0, '東':90, '南':180, '西':270}
    for key, ch in compass.items():
        deg = base_deg + offsets[key]
        rad = np.radians(deg)
        x_top = (scaled_radius + 1.1)*np.cos(rad)
        y_top = (scaled_radius + 1.1)*np.sin(rad)
        z_top = height
        ax.plot([scaled_radius*np.cos(rad), x_top], [scaled_radius*np.sin(rad), y_top],
                [height, height+0.1], color='darkred', linewidth=2, alpha=0.9)
        ax.text(x_top, y_top, height+0.2, ch,
                fontsize=18, color='darkred', fontweight='bold', ha='center', va='bottom')

    # 數據標籤
    if show_labels:
        data_labels = [f'A{i+1}' for i in range(len(x))]
        for idx, (x_val, y_val, z_val, label) in enumerate(zip(scaled_x, scaled_y, z, data_labels)):
            if x_val**2 + y_val**2 <= scaled_radius**2:
                ax.text(x_val, y_val, z_val+0.2, label,
                        color='black', fontsize=10, fontweight='bold', ha='center', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'))

    ax.set_xlabel('X (公尺)', fontsize=14, fontname=fontname)
    ax.set_ylabel('Y (公尺)', fontsize=14, fontname=fontname)
    ax.set_zlabel('高度 Z (公尺)', fontsize=14, fontname=fontname)
    ax.set_title(tank_name, fontsize=22, fontweight='bold', loc='center',fontfamily=fontname)
    ax.set_xlim(scaled_radius, -scaled_radius)
    ax.set_ylim(scaled_radius, -scaled_radius)
    ax.set_zlim(0, height)
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()

    # ===== colorbar區（用等高Bar色顯示）=====
    ax2 = fig.add_subplot(122)
    norm = plt.Normalize(z_min, z_max)
    cb1 = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap),
        cax=ax2, orientation='vertical', shrink=0.7, aspect=7
    )
    cb1.set_label('高度 (公尺)', fontsize=14, fontfamily='fontfamily=fontname')
    tick_values = np.linspace(z_min, z_max, 8)
    cb1.set_ticks(tick_values)
    cb1.set_ticklabels([f"{v:.1f}" for v in tick_values])
    ax2.set_visible(False) # 隱藏bar右的坐標軸，僅顯示色階本身

    st.subheader("油槽3D油泥分布圖")
    st.pyplot(fig)

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", dpi=300)
    img_buf.seek(0)
    st.download_button("下載圖片 (PNG)", img_buf, "oil_sludge_Label.png", mime="image/png")
    st.download_button("下載 Excel (原始數據)", data.to_csv(index=False).encode("utf-8-sig"), "oil_sludge_data.csv")
    st.caption("Designed by Minda (油槽/可調參數/色階)")
else:
    st.info("請輸入數據、參數並按左側『執行』。")




