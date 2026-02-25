"""
Streamlit dashboard for BDD object detection dataset analysis.

Sections: overview metrics, class distribution (train & val),
anomalies, interesting samples per class, raw data tables.

Run: streamlit run data_analysis/dashboard.py
"""
from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from data.config import DEFAULT_BDD_ROOT, DETECTION_CLASSES
from data_analysis.analysis import run_full_analysis

HIGHLIGHT_COLOR = (0, 255, 0)       # green for the selected class
OTHER_COLOR = (160, 160, 160)       # grey for other classes
HIGHLIGHT_WIDTH = 3
OTHER_WIDTH = 1


def draw_boxes_on_image(
    path: Path, boxes: list[dict], highlight_category: str | None = None
) -> Image.Image:
    """Draw all bounding boxes on image with category labels.

    Boxes matching highlight_category are drawn in green with thicker lines;
    other classes are drawn in grey so the user has full context.
    """
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Draw non-highlighted boxes first so highlighted ones are on top
    for is_highlight in (False, True):
        for b in boxes:
            cat = b.get("category", "")
            match = (cat == highlight_category)
            if match != is_highlight:
                continue
            x1, y1 = int(b["x1"]), int(b["y1"])
            x2, y2 = int(b["x2"]), int(b["y2"])
            color = HIGHLIGHT_COLOR if match else OTHER_COLOR
            width = HIGHLIGHT_WIDTH if match else OTHER_WIDTH
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            label_y = max(y1 - 16, 0)
            draw.text((x1, label_y), cat, fill=color, font=font)
    return img


st.set_page_config(page_title="BDD Object Detection Analysis", layout="wide")

# Path input
bdd_root = st.sidebar.text_input(
    "BDD dataset root",
    value=str(DEFAULT_BDD_ROOT),
    help="Path to BDD folder (e.g. /data when using Docker with volume mount).",
)
bdd_path = Path(bdd_root)
if not bdd_path.exists():
    st.error(f"Path does not exist: {bdd_path}")
    st.stop()

# Full dataset by default (first load can be slow; subsequent loads use Streamlit cache).
st.sidebar.caption("Data scope")
use_sample = st.sidebar.checkbox(
    "Quick preview (sample only)",
    value=False,
    help="Limit to 1,000 frames per split for a faster load. Uncheck for full dataset.",
)
max_frames_per_split: int | None = 1000 if use_sample else None


@st.cache_data(show_spinner="Loading BDD labels and running analysis...")
def load_analysis(root: Path, max_frames_per_split: int | None) -> dict:
    """Run analysis; result is cached by (root, max_frames_per_split)."""
    return run_full_analysis(root, max_frames_per_split=max_frames_per_split)


with st.spinner("Loading dataset..."):
    results = load_analysis(bdd_path, max_frames_per_split)

records = results["records"]
image_stats = results["image_stats"]
class_dist = results["class_distribution"]
train_val = results["train_val_analysis"]
anomalies = results["anomalies"]
interesting = results["interesting_samples"]

st.title("BDD100K Object Detection – Dataset Analysis")
st.caption(f"Data root: {results['bdd_root']}")
if max_frames_per_split is not None:
    st.caption(f"Quick preview: stats from a sample (up to {max_frames_per_split} frames per split). Uncheck **Quick preview** in the sidebar for full dataset.")

# --- Overview ---
st.header("1. Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Train images", train_val["n_train_images"])
c2.metric("Val images", train_val["n_val_images"])
c3.metric("Train instances", train_val["n_train_instances"])
c4.metric("Val instances", train_val["n_val_instances"])
st.metric("Images with zero detections", anomalies.get("n_empty_images", 0))

# --- Class distribution ---
st.header("2. Class distribution (train & val)")
dist_pivot = class_dist.pivot(index="category", columns="split", values="instance_count").reindex(
    DETECTION_CLASSES
)
st.bar_chart(dist_pivot)
st.dataframe(class_dist, use_container_width=True, hide_index=True)

# --- Size & aspect ratio ---
st.header("3. Size and aspect ratio by class")
st.subheader("Area (px²) per class")
size_stats = results["size_stats"]
st.dataframe(size_stats, use_container_width=True, hide_index=True)
st.subheader("Aspect ratio (width / height) per class")
aspect_stats = results["aspect_ratio_stats"]
st.dataframe(aspect_stats, use_container_width=True, hide_index=True)

# --- Anomalies & patterns ---
st.header("4. Anomalies & patterns")
st.subheader("Occlusion and truncation by class (%)")
occ_df = pd.DataFrame(anomalies.get("occlusion_truncation_rates", {})).T
if not occ_df.empty:
    occ_df = occ_df.reset_index().rename(columns={"index": "class"})
    st.dataframe(occ_df, use_container_width=True, hide_index=True)
st.subheader("Tiny boxes (area < 100 px²)")
st.write(anomalies.get("tiny_boxes", {}))
st.subheader("Very large boxes (> 50% frame area)")
st.write(anomalies.get("huge_boxes", {}))
st.subheader("Sample empty images (no detection labels)")
empty_list = anomalies.get("empty_images", [])[:20]
if empty_list:
    st.dataframe(pd.DataFrame(empty_list), use_container_width=True, hide_index=True)
else:
    st.info("No empty images in this sample.")

# --- Interesting samples ---
st.header("5. Interesting samples by class")
class_sel = st.selectbox("Select class", DETECTION_CLASSES, key="class_sel")
samples = interesting.get(class_sel, [])
if samples:
    for s in samples:
        with st.expander(f"{s['image_name']} ({s['split']}) – {s['reason']}"):
            # Resolve path from current BDD root so images show when cache was built elsewhere (e.g. Docker vs local)
            img_path = bdd_path / "images" / s["split"] / f"{s['image_name']}.jpg"
            if not img_path.exists():
                img_path = bdd_path / "images" / s["split"] / f"{s['image_name']}.jpeg"
            if img_path.exists():
                boxes = s.get("boxes", [])
                if boxes:
                    img = draw_boxes_on_image(img_path, boxes, highlight_category=class_sel)
                    st.image(img, use_container_width=True)
                else:
                    st.image(str(img_path), use_container_width=True)
            else:
                st.warning(f"Image not found: {img_path}")
                st.caption("Ensure the BDD root in the sidebar points to a directory that contains an `images/` folder with train/ and val/ subfolders.")
            st.json({k: v for k, v in s.items() if k not in ("image_path", "boxes")})
else:
    st.info(f"No samples found for class '{class_sel}'.")

# --- Raw tables (optional) ---
st.header("6. Raw data")
tab1, tab2 = st.tabs(["Records (sample)", "Image stats (sample)"])
with tab1:
    st.dataframe(records.head(500), use_container_width=True, hide_index=True)
with tab2:
    st.dataframe(image_stats.head(500), use_container_width=True, hide_index=True)
