import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="SULFATE-LIFE AI", page_icon="🏗️", layout="wide")

st.markdown("""
<style>
.main {background: #f6f8fb;}
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
.hero {padding: 1.5rem 1.7rem; border-radius: 18px; background: linear-gradient(120deg,#102a43,#176b87); color: white; margin-bottom: 1rem;}
.hero h1 {color:white; margin:0; font-size:2.15rem;}
.hero p {color:#e6f4f7; margin:.5rem 0 0 0;}
.small-note {font-size: .88rem; color:#586574;}
div[data-testid="stMetric"] {background:white; border:1px solid #e5eaf0; padding:12px; border-radius:12px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🏗️ SULFATE-LIFE AI</h1>
<p>Ứng dụng học máy hỗ trợ phân tích độ bền bê tông và nghiên cứu tuổi thọ kết cấu BTCT trong môi trường sulfat</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Điều hướng")
    page = st.radio("Chọn chức năng", ["Tổng quan", "Dự đoán từ mô hình", "Huấn luyện & đánh giá", "Tạo mẫu dữ liệu"])
    st.divider()
    st.caption("Phiên bản đồ án nghiên cứu")
    st.caption("Kết quả chỉ phục vụ tham khảo học thuật, không thay thế thiết kế/kiểm định kết cấu.")

st.warning("Lưu ý khoa học: phần mềm không tự tạo ra độ chính xác hay tuổi thọ giả. Muốn dự đoán định lượng, hãy tải dữ liệu thí nghiệm thực và huấn luyện mô hình.")

FEATURE_HELP = {
    "silica_fume_pct": "Hàm lượng silica fume (% khối lượng chất kết dính; phải thống nhất cách định nghĩa)",
    "water_binder_ratio": "Tỷ lệ nước/chất kết dính (w/b)",
    "initial_strength_mpa": "Cường độ nén ban đầu (MPa)",
    "sulfate_concentration_pct": "Nồng độ dung dịch sulfat (%) — chỉ so sánh dữ liệu dùng cùng định nghĩa",
    "temperature_c": "Nhiệt độ môi trường/thí nghiệm (°C)",
    "exposure_days": "Thời gian tiếp xúc (ngày)",
    "wet_dry_cycles": "Số chu kỳ ướt–khô",
    "curing_days": "Thời gian bảo dưỡng (ngày)",
    "relative_humidity_pct": "Độ ẩm tương đối (%)",
}
FEATURES = list(FEATURE_HELP.keys())
TARGET_OPTIONS = {
    "remaining_strength_mpa": "Cường độ nén còn lại (MPa)",
    "strength_loss_pct": "Độ giảm cường độ (%)",
    "expansion_pct": "Độ trương nở (%)",
    "mass_change_pct": "Thay đổi khối lượng (%)",
}

def template_df():
    return pd.DataFrame([
        [0, .40, 40, 5, 25, 30, 0, 28, 75, 37],
        [5, .40, 42, 5, 25, 30, 0, 28, 75, 40],
        [10, .40, 43, 5, 25, 30, 0, 28, 75, 41],
        [0, .40, 40, 5, 25, 60, 5, 28, 75, 34],
        [10, .40, 43, 5, 25, 60, 5, 28, 75, 38],
    ], columns=FEATURES + ["remaining_strength_mpa"])

def load_df(upload):
    if upload is None:
        return None
    try:
        if upload.name.lower().endswith(".xlsx"):
            return pd.read_excel(upload)
        return pd.read_csv(upload)
    except Exception as e:
        st.error(f"Không đọc được tệp: {e}")
        return None

def clean_numeric(df, columns):
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

if page == "Tổng quan":
    st.subheader("Bảng điều khiển nghiên cứu")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mục tiêu", "Độ bền sulfat")
    c2.metric("Thuật toán", "Random Forest / Extra Trees / SVR")
    c3.metric("Dữ liệu", "Do người dùng cung cấp")
    st.markdown("### Quy trình sử dụng")
    steps = [
        ("01", "Chuẩn bị dữ liệu", "CSV hoặc Excel; mỗi hàng là một mẫu/điều kiện thí nghiệm."),
        ("02", "Huấn luyện", "Chọn biến đầu vào và biến cần dự đoán; kiểm tra chất lượng dữ liệu."),
        ("03", "Đánh giá", "Xem MAE, RMSE, R² trên tập kiểm tra độc lập."),
        ("04", "Dự đoán", "Nhập điều kiện mới để mô hình đã huấn luyện dự đoán đầu ra."),
        ("05", "Báo cáo", "Tải kết quả dự đoán và bảng đánh giá xuống CSV."),
    ]
    for n, title, desc in steps:
        with st.container(border=True):
            st.markdown(f"**{n} · {title}**")
            st.write(desc)
    st.markdown("### Các biến dữ liệu được hỗ trợ")
    st.dataframe(pd.DataFrame({"Tên cột": FEATURES, "Ý nghĩa": [FEATURE_HELP[x] for x in FEATURES]}), use_container_width=True, hide_index=True)
    st.info("Cột mục tiêu có thể là remaining_strength_mpa, strength_loss_pct, expansion_pct hoặc mass_change_pct. Chỉ chọn một mục tiêu có dữ liệu thực.")

elif page == "Tạo mẫu dữ liệu":
    st.subheader("Tạo tệp mẫu để điền dữ liệu")
    st.write("Tệp mẫu chỉ minh họa cấu trúc cột. Các số trong tệp được đánh dấu là dữ liệu minh họa, không phải kết quả thực nghiệm và không nên dùng để kết luận khoa học.")
    df = template_df()
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Tải mẫu CSV", df.to_csv(index=False).encode("utf-8-sig"), "mau_du_lieu_sulfate_life_ai.csv", "text/csv")
    st.caption("Hãy thay các hàng minh họa bằng dữ liệu thí nghiệm có nguồn gốc rõ ràng trước khi huấn luyện mô hình.")

elif page == "Huấn luyện & đánh giá":
    st.subheader("Huấn luyện mô hình từ dữ liệu")
    upload = st.file_uploader("Tải dữ liệu CSV hoặc Excel", type=["csv", "xlsx"], key="train_upload")
    if upload:
        df = load_df(upload)
        if df is not None:
            st.write("Xem trước dữ liệu")
            st.dataframe(df.head(20), use_container_width=True)
            available_features = [c for c in FEATURES if c in df.columns]
            if len(available_features) < 2:
                st.error("Cần ít nhất 2 cột biến đầu vào được hỗ trợ. Hãy dùng tên cột trong trang 'Tạo mẫu dữ liệu'.")
            else:
                target = st.selectbox("Chọn biến cần dự đoán", [c for c in TARGET_OPTIONS if c in df.columns] or list(TARGET_OPTIONS.keys()))
                if target not in df.columns:
                    st.error(f"Không tìm thấy cột mục tiêu '{target}' trong dữ liệu.")
                else:
                    features = st.multiselect("Chọn biến đầu vào", available_features, default=available_features)
                    test_size = st.slider("Tỷ lệ dữ liệu kiểm tra", 0.15, 0.40, 0.20, 0.05)
                    model_name = st.selectbox("Thuật toán", ["Random Forest", "Extra Trees", "SVR"])
                    if st.button("Huấn luyện và đánh giá", type="primary"):
                        if not features:
                            st.error("Hãy chọn ít nhất một biến đầu vào.")
                        else:
                            data = clean_numeric(df, features + [target]).dropna(subset=[target])
                            data = data.replace([np.inf, -np.inf], np.nan)
                            data = data.dropna(subset=[target])
                            if len(data) < 12:
                                st.error("Cần tối thiểu 12 dòng dữ liệu có mục tiêu để chia tập và đánh giá sơ bộ; nhiều dữ liệu hơn sẽ đáng tin cậy hơn.")
                            else:
                                X = data[features]
                                y = data[target]
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=test_size, random_state=42
                                )
                                if model_name == "Random Forest":
                                    reg = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2)
                                elif model_name == "Extra Trees":
                                    reg = ExtraTreesRegressor(n_estimators=300, random_state=42, min_samples_leaf=2)
                                else:
                                    reg = Pipeline([("imputer", SimpleImputer(strategy="median")),
                                                    ("scale", StandardScaler()),
                                                    ("model", SVR(C=10, epsilon=0.1))])
                                if model_name in ["Random Forest", "Extra Trees"]:
                                    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", reg)])
                                else:
                                    model = reg
                                model.fit(X_train, y_train)
                                pred = model.predict(X_test)
                                mae = mean_absolute_error(y_test, pred)
                                rmse = np.sqrt(mean_squared_error(y_test, pred))
                                r2 = r2_score(y_test, pred) if len(y_test) > 1 else float("nan")
                                st.session_state["sulfate_model"] = model
                                st.session_state["sulfate_features"] = features
                                st.session_state["sulfate_target"] = target
                                st.session_state["sulfate_model_name"] = model_name
                                st.session_state["sulfate_data_range"] = {c: (float(data[c].min()), float(data[c].max())) for c in features if data[c].notna().any()}
                                st.session_state["sulfate_metrics"] = {"MAE": mae, "RMSE": rmse, "R²": r2, "n_train": len(X_train), "n_test": len(X_test)}
                                result = pd.DataFrame({"Gia_tri_thuc": y_test.to_numpy(), "Gia_tri_du_doan": pred})
                                st.session_state["sulfate_test_results"] = result
                                a,b,c = st.columns(3)
                                a.metric("MAE", f"{mae:.4g}")
                                b.metric("RMSE", f"{rmse:.4g}")
                                c.metric("R²", f"{r2:.4g}" if np.isfinite(r2) else "Không xác định")
                                st.caption(f"Mô hình: {model_name} · Huấn luyện: {len(X_train)} mẫu · Kiểm tra: {len(X_test)} mẫu")
                                fig, ax = plt.subplots()
                                ax.scatter(y_test, pred)
                                lo = min(float(np.min(y_test)), float(np.min(pred)))
                                hi = max(float(np.max(y_test)), float(np.max(pred)))
                                ax.plot([lo, hi], [lo, hi], linestyle="--")
                                ax.set_xlabel("Giá trị thực")
                                ax.set_ylabel("Giá trị dự đoán")
                                ax.set_title("Thực tế so với dự đoán (tập kiểm tra)")
                                st.pyplot(fig)
                                st.download_button("⬇️ Tải kết quả kiểm tra", result.to_csv(index=False).encode("utf-8-sig"), "ket_qua_danh_gia.csv", "text/csv")
                                st.success("Đã huấn luyện trong phiên hiện tại. Chuyển sang trang Dự đoán từ mô hình để thử đầu vào mới.")
    else:
        st.info("Tải dữ liệu CSV/Excel của bạn để bắt đầu. Không có dữ liệu thật thì phần mềm không đưa ra điểm số đánh giá.")

elif page == "Dự đoán từ mô hình":
    st.subheader("Dự đoán từ mô hình đã huấn luyện")
    if "sulfate_model" not in st.session_state:
        st.info("Bạn chưa huấn luyện mô hình trong phiên này. Vào 'Huấn luyện & đánh giá', tải dữ liệu và huấn luyện trước.")
    else:
        model = st.session_state["sulfate_model"]
        features = st.session_state["sulfate_features"]
        target = st.session_state["sulfate_target"]
        ranges = st.session_state.get("sulfate_data_range", {})
        st.write(f"**Mô hình:** {st.session_state['sulfate_model_name']}  ·  **Biến đầu ra:** {TARGET_OPTIONS.get(target, target)}")
        st.caption("Nên nhập trong phạm vi dữ liệu huấn luyện. Ngoại suy ngoài phạm vi này có thể không đáng tin cậy.")
        values = {}
        cols = st.columns(2)
        for i, feature in enumerate(features):
            default = (ranges[feature][0] + ranges[feature][1]) / 2 if feature in ranges else 0.0
            if feature in ranges:
                low, high = ranges[feature]
                if low == high:
                    value = cols[i % 2].number_input(FEATURE_HELP.get(feature, feature), value=float(default), key=feature)
                else:
                    value = cols[i % 2].number_input(FEATURE_HELP.get(feature, feature), value=float(default), min_value=float(low), max_value=float(high), key=feature)
            else:
                value = cols[i % 2].number_input(FEATURE_HELP.get(feature, feature), value=float(default), key=feature)
            values[feature] = value
        if st.button("Dự đoán", type="primary"):
            input_df = pd.DataFrame([values], columns=features)
            result = float(model.predict(input_df)[0])
            st.metric("Giá trị dự đoán", f"{result:.4g}", TARGET_OPTIONS.get(target, target))
            st.caption("Đây là dự đoán của mô hình trên dữ liệu đã cung cấp, không phải xác nhận tuổi thọ hoặc an toàn công trình.")
            out = input_df.copy()
            out["target_du_doan"] = result
            st.download_button("⬇️ Tải dự đoán CSV", out.to_csv(index=False).encode("utf-8-sig"), "du_doan_sulfate_life_ai.csv", "text/csv")
        metrics = st.session_state.get("sulfate_metrics")
        if metrics:
            with st.expander("Xem kết quả đánh giá mô hình"):
                st.json(metrics)
