"""
Streamlit Web UI - AIC2026 Multimedia Retrieval System
Giao diện người dùng tìm kiếm video đa phương tiện trực quan (TRAKE Ready).
"""

import os
import requests
import streamlit as st

st.set_page_config(
    page_title="AIC 2026 Multimedia Retrieval System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 AIC 2026 Multimedia Retrieval System")
st.markdown("### Hệ thống truy xuất Video đa phương tiện độ trễ Mili-giây (Local-First Architecture)")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Cấu hình Trọng số & Tìm kiếm")
    image_weight = st.slider("Trọng số Hình ảnh (SigLIP 2)", 0.0, 1.0, 0.60, 0.05)
    metadata_weight = st.slider("Trọng số Metadata (OCR/Objects)", 0.0, 1.0, 0.30, 0.05)
    audio_weight = st.slider("Trọng số Âm thanh (PhoWhisper)", 0.0, 1.0, 0.10, 0.05)
    
    top_k = st.number_input("Top K Clips trả về", min_value=5, max_value=200, value=30)
    backend_url = st.text_input("FastAPI Backend URL", "http://localhost:8000")

# --- MAIN SEARCH INTERFACE ---
query = st.text_input("🔍 Nhập từ khóa hoặc mô tả cảnh quay (Ví dụ: 'Người đi bộ trên phố cà phê Hà Nội'):", "")

if st.button("🚀 Tìm kiếm Video", type="primary") and query:
    st.info(f"Đang gửi yêu cầu tìm kiếm cho: '{query}'...")

    payload = {
        "query": query,
        "top_k": top_k,
        "weights": {
            "image": image_weight,
            "audio": audio_weight,
            "meta": metadata_weight
        }
    }

    try:
        api_endpoint = f"{backend_url}/api/search"
        response = requests.post(api_endpoint, json=payload, timeout=10)

        if response.status_code == 200:
            clips = response.json()
            st.success(f"Đã tìm thấy {len(clips)} Event Clips phù hợp!")

            # RENDER DANH SÁCH CARDS
            for idx, clip in enumerate(clips, start=1):
                with st.container():
                    st.markdown(f"---")
                    st.subheader(f"#{idx} | Clip ID: `{clip.get('clip_id')}` (Điểm số: `{clip.get('score')}`)")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"**📹 Video:** `{clip.get('video_id')}`")
                        st.markdown(f"**⏱️ Thời gian:** `{clip.get('start_time_str')}` ➔ `{clip.get('end_time_str')}` ({clip.get('duration_sec')}s)")
                        st.markdown(f"**🎯 Shots:** `{clip.get('start_shot_id')}` đến `{clip.get('end_shot_id')}`")
                        st.markdown(f"**🖼️ Keyframes count:** `{clip.get('keyframes_count')}`")
                        
                    with col2:
                        st.markdown(f"**📝 Bối cảnh 30s (Context Summary):**")
                        st.caption(clip.get("context_summary", "Không có bối cảnh."))

                    # RENDER KEYFRAMES IMAGES
                    keyframes = clip.get("keyframes", [])
                    if keyframes:
                        st.markdown("**🖼️ Các Khung hình Keyframes tiêu biểu trong Clip:**")
                        cols = st.columns(min(len(keyframes), 4))
                        for k_idx, kf in enumerate(keyframes):
                            col_target = cols[k_idx % len(cols)]
                            with col_target:
                                img_p = kf.get("frame_path", "")
                                if os.path.exists(img_p):
                                    st.image(img_p, caption=f"Shot {kf.get('shot_id')} ({kf.get('frame_type')})\n{kf.get('timestamp_str')}")
                                else:
                                    st.info(f"Frame: {kf.get('frame_type')}\n{kf.get('timestamp_str')}")
                                st.caption(f"Objs: {kf.get('objects', '')}\nOCR: {kf.get('ocr_text', '')}")

        else:
            st.error(f"Lỗi API Server (Code {response.status_code}): {response.text}")

    except Exception as e:
        st.error(f"Không thể kết nối tới FastAPI Backend tại '{backend_url}': {e}")
