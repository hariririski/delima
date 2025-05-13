import streamlit as st
import os, json, cv2
import numpy as np
import torch
import pyiqa

# Konfigurasi pipeline
PIPELINE_RGB = "P10"
PIPELINE_GRAY = "P1"

PRESET_PATH_RGB = os.path.join("out_rgb", f"preset_{PIPELINE_RGB}.json")
PRESET_PATH_GRAY = os.path.join("out", f"preset_{PIPELINE_GRAY}.json")

niqe = pyiqa.create_metric("niqe")
piqe = pyiqa.create_metric("piqe")

# ==== Evaluasi Metrik ====
def eval_q_rgb(img):
    t = torch.tensor(img / 255.).permute(2, 0, 1).unsqueeze(0).float()
    cf = float(np.std(img[..., 0] - img[..., 1]) + 0.3 * np.std(img[..., 2] - img.mean(axis=2)))
    return {"niqe": round(niqe(t).item(), 2), "piqe": round(piqe(t).item(), 2), "cf": round(cf, 2)}

def eval_q_gray(img):
    t = torch.tensor(img / 255.).unsqueeze(0).unsqueeze(0).float()
    return {"niqe": round(niqe(t).item(), 2), "piqe": round(piqe(t).item(), 2)}

def resize_same_height(imgs, target_height=300):
    resized = []
    for im in imgs:
        h, w = im.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        resized.append(cv2.resize(im, (new_w, target_height), interpolation=cv2.INTER_AREA))
    return resized

# ==== Transformasi RGB ====
def denoise_rgb(im, method, p):
    if method == "bilateral":
        return cv2.bilateralFilter(im, p["d"], p["sigmaColor"], p["sigmaSpace"])
    if method == "fast_nlm":
        return cv2.fastNlMeansDenoisingColored(im, None, p["h"], p["h"], p["template"], p["search"])
    return im

def contrast_rgb(imf, method, p):
    if method == "clahe":
        lab = cv2.cvtColor((imf * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cla = cv2.createCLAHE(clipLimit=p["clip"], tileGridSize=tuple(p["tile"]))
        cl = cla.apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB) / 255.
    if method == "adaptive_gamma":
        return np.power(imf, 0.5 + p["mult"] * (1 - imf.mean()))
    if method == "percentile":
        lo, hi = np.percentile(imf, (p["lo"] * 100, p["hi"] * 100))
        return np.clip((imf - lo) / (hi - lo), 0, 1)
    return imf

def sharpen_rgb(imf, method, p):
    im8 = np.clip(imf * 255, 0, 255).astype(np.uint8)
    if method == "unsharp":
        blur = cv2.GaussianBlur(im8, (0, 0), p["sigma"])
        return cv2.addWeighted(im8, 1 + p["alpha"], blur, -p["alpha"], 0)
    if method == "guided":
        return cv2.edgePreservingFilter(im8, flags=1, sigma_s=p["radius"] * 10, sigma_r=p["eps"])
    return im8

# ==== Transformasi Grayscale ====
def denoise_gray(im, method, p):
    if method == "median":
        return cv2.medianBlur(im, p["ksize"])
    if method == "fast_nlm":
        im8 = (im // 256).astype(np.uint8)
        out = cv2.fastNlMeansDenoising(im8, None, p["h"], p["template"], p["search"])
        return out.astype(np.uint16) * 256
    return im

def contrast_gray(f, method, p):
    if method == "clahe":
        cla = cv2.createCLAHE(clipLimit=p["clip"], tileGridSize=tuple(p["tile"]))
        return cla.apply((f * 255).astype(np.uint8)) / 255.
    if method == "adaptive_gamma":
        return np.power(f, 0.5 + p["mult"] * (1 - f.mean()))
    if method == "percentile":
        lo, hi = np.percentile(f, (p["lo"], p["hi"]))
        return np.clip((f - lo) / (hi - lo), 0, 1)
    return f

def sharpen_gray(f, method, p):
    im8 = np.clip(f * 255, 0, 255).astype(np.uint8)
    if method == "unsharp":
        blur = cv2.GaussianBlur(im8, (0, 0), p["sigma"])
        return cv2.addWeighted(im8, 1 + p["alpha"], blur, -p["alpha"], 0)
    if method == "guided":
        return cv2.edgePreservingFilter(im8, flags=1, sigma_s=p["radius"] * 10, sigma_r=p["eps"])
    return im8

# ==== UI STREAMLIT ====
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    .element-container h6 {
        font-size: 22px !important;
        font-weight: 800 !important;
        color: #000 !important;
        text-align: center !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎯 Visualisasi Pipeline Preset RGB & Grayscale")

# === RGB PIPELINE ===
st.subheader(f"🎨 Pipeline RGB ({PIPELINE_RGB})")
img_rgb = st.file_uploader("Upload gambar RGB (TIFF/PNG/JPG)", type=["tiff", "tif", "png", "jpg", "jpeg"], key="rgb")

if img_rgb:
    preset_rgb = json.load(open(PRESET_PATH_RGB))
    img_arr = cv2.imdecode(np.frombuffer(img_rgb.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

    if img.ndim != 3 or img.shape[2] != 3:
        st.error("❌ Gambar RGB harus memiliki 3 channel warna (R, G, B).")
    else:
        q0 = eval_q_rgb(img)
        d = denoise_rgb(img, preset_rgb["denoise"], preset_rgb["den_kw"])
        df_ = d.astype(np.float32) / 255.
        c = contrast_rgb(df_, preset_rgb["contrast"], preset_rgb["con_kw"])
        f = sharpen_rgb(c, preset_rgb["sharpen"], preset_rgb["shr_kw"])
        q1 = eval_q_rgb(f)

        titles = [
            f"<b>Original</b><br>NIQE {q0['niqe']} | PIQE {q0['piqe']} | CF {q0['cf']}",
            f"<b>Denoised</b><br>Method: {preset_rgb['denoise']}<br>Param: {preset_rgb['den_kw']}<br>NIQE {eval_q_rgb(d)['niqe']} | PIQE {eval_q_rgb(d)['piqe']} | CF {eval_q_rgb(d)['cf']}",
            f"<b>Contrast</b><br>Method: {preset_rgb['contrast']}<br>Param: {preset_rgb['con_kw']}<br>NIQE {eval_q_rgb((c*255).astype(np.uint8))['niqe']} | PIQE {eval_q_rgb((c*255).astype(np.uint8))['piqe']} | CF {eval_q_rgb((c*255).astype(np.uint8))['cf']}",
            f"<b>Sharpened</b><br>Method: {preset_rgb['sharpen']}<br>Param: {preset_rgb['shr_kw']}<br>NIQE {q1['niqe']} | PIQE {q1['piqe']} | CF {q1['cf']}"
        ]
        images = resize_same_height([img, d, (c*255).astype(np.uint8), f])
        cols = st.columns(4)
        for col, title, im in zip(cols, titles, images):
            col.image(im, use_column_width=True)
            col.markdown(f"<div style='font-size:18px; font-weight:700; color:#000;'>{title}</div>", unsafe_allow_html=True)

# === GRAYSCALE PIPELINE ===
st.subheader(f"🌑 Pipeline Grayscale ({PIPELINE_GRAY})")
img_gray = st.file_uploader("Upload gambar Grayscale (TIFF)", type=["tiff", "tif"], key="gray")

if img_gray:
    preset_gray_all = json.load(open(PRESET_PATH_GRAY))
    preset_gray = preset_gray_all if "den" in preset_gray_all else preset_gray_all[PIPELINE_GRAY]

    im = cv2.imdecode(np.frombuffer(img_gray.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    if im.ndim != 2:
        st.error("❌ Gambar grayscale harus 1 channel (hitam-putih).")
    else:
        orig8 = (im // 256).astype(np.uint8)
        q0 = eval_q_gray(orig8)

        d16 = denoise_gray(im, preset_gray["den"], preset_gray["dkw"])
        cf = contrast_gray(d16.astype(np.float32) / 65535, preset_gray["con"], preset_gray["ckw"])
        fin = sharpen_gray(cf, preset_gray["shr"], preset_gray["skw"])
        q1 = eval_q_gray(fin)

        titles = [
            f"<b>Original</b><br>NIQE {q0['niqe']} | PIQE {q0['piqe']}",
            f"<b>Denoised</b><br>Method: {preset_gray['den']}<br>Param: {preset_gray['dkw']}<br>NIQE {eval_q_gray((d16//256).astype(np.uint8))['niqe']} | PIQE {eval_q_gray((d16//256).astype(np.uint8))['piqe']}",
            f"<b>Contrast</b><br>Method: {preset_gray['con']}<br>Param: {preset_gray['ckw']}<br>NIQE {eval_q_gray((cf*255).astype(np.uint8))['niqe']} | PIQE {eval_q_gray((cf*255).astype(np.uint8))['piqe']}",
            f"<b>Sharpened</b><br>Method: {preset_gray['shr']}<br>Param: {preset_gray['skw']}<br>NIQE {q1['niqe']} | PIQE {q1['piqe']}"
        ]
        images = resize_same_height([orig8, (d16//256).astype(np.uint8), (cf*255).astype(np.uint8), fin])
        cols = st.columns(4)
        for col, title, im in zip(cols, titles, images):
            col.image(im, use_column_width=True)
            col.markdown(f"<div style='font-size:18px; font-weight:700; color:#000;'>{title}</div>", unsafe_allow_html=True)