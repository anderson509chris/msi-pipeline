"""GUI for the MSI target-extraction pipeline.

Run with:
    streamlit run app.py
"""
import io
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from msi_io import DEFAULT_TARGETS
from run_pipeline import clean_out_dir, discover_runs, stage_commands

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="MSI Pipeline", layout="wide")
st.title("MALDI-MSI target extraction pipeline")

if "targets_df" not in st.session_state:
    st.session_state.targets_df = pd.DataFrame(DEFAULT_TARGETS["targets"])
if "params" not in st.session_state:
    st.session_state.params = dict(DEFAULT_TARGETS["params"])
if "path_str" not in st.session_state:
    st.session_state.path_str = ""


def browse_folder():
    """Opens a native folder picker in a separate process (folder_dialog.py).
    Only works for local use (server and browser on the same machine)."""
    try:
        result = subprocess.run(
            [sys.executable, str(HERE / "folder_dialog.py")],
            capture_output=True, text=True, timeout=300,
        )
        folder = result.stdout.strip()
        return folder or None
    except Exception as e:
        st.warning(f"Couldn't open a native folder dialog ({e}). Type the path instead.")
        return None


# --- data folder ---------------------------------------------------------
st.subheader("1. Data folder")
col1, col2 = st.columns([5, 1])
with col1:
    path_str = st.text_input(
        "A single run folder, or a parent folder containing several run folders",
        value=st.session_state.path_str,
        placeholder="/path/to/data",
    )
with col2:
    st.write("")
    if st.button("Browse…", use_container_width=True):
        chosen = browse_folder()
        if chosen:
            path_str = chosen
st.session_state.path_str = path_str

runs = []
base_path = None
if path_str:
    base_path = Path(path_str).expanduser()
    if not base_path.is_dir():
        st.error(f"Not a folder: {base_path}")
    else:
        try:
            runs = discover_runs(base_path)
        except SystemExit as e:
            st.error(str(e))

selected_runs = []
if runs:
    st.caption(f"Found {len(runs)} run folder(s) (each needs a 'Profile Mode' and 'Centroid Mode' subfolder):")
    for r in runs:
        label = r.name if r != base_path else r.name
        checked = st.checkbox(label, value=True, key=f"sel_{r}")
        if checked:
            selected_runs.append(r)

# --- targets ---------------------------------------------------------
st.subheader("2. Target m/z list")
st.caption("Edit, add, or remove rows. Names/formulas are only used for figure labels and the output CSV.")
st.session_state.targets_df = st.data_editor(
    st.session_state.targets_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "mz": st.column_config.NumberColumn("m/z", format="%.4f", required=True),
        "name": st.column_config.TextColumn("Name"),
        "formula_plain": st.column_config.TextColumn("Formula"),
    },
)

with st.expander("Advanced parameters"):
    c1, c2, c3, c4 = st.columns(4)
    st.session_state.params["ntop"] = c1.number_input("Top-N pixels", min_value=1, value=int(st.session_state.params["ntop"]), step=10)
    st.session_state.params["halfwin"] = c2.number_input("Half window (Da)", value=float(st.session_state.params["halfwin"]), format="%.4f")
    st.session_state.params["grid"] = c3.number_input("Grid spacing (Da)", value=float(st.session_state.params["grid"]), format="%.6f")
    st.session_state.params["ppm"] = c4.number_input("PPM tolerance (pass1)", value=float(st.session_state.params["ppm"]))

show_metrics_box = st.checkbox(
    "Show metrics box (centroid, FWHM, R, peak count) on graphs",
    value=st.session_state.get("show_metrics_box", True),
)
st.session_state.show_metrics_box = show_metrics_box

# --- run ---------------------------------------------------------
st.subheader("3. Run")
run_clicked = st.button("Run pipeline", type="primary", disabled=not selected_runs)

if run_clicked:
    targets_df = st.session_state.targets_df.dropna(subset=["mz"])
    if targets_df.empty:
        st.error("Add at least one target m/z first.")
        st.stop()

    targets_payload = {
        "sample_name": base_path.name,
        "instrument_desc": "",
        "targets": targets_df.fillna("").to_dict("records"),
        "params": st.session_state.params,
    }
    targets_path = Path(tempfile.mkdtemp()) / "targets.json"
    targets_path.write_text(json.dumps(targets_payload, indent=2))

    total_stages = len(selected_runs) * len(stage_commands(selected_runs[0], None, targets_path, show_metrics_box))
    done = 0
    progress = st.progress(0.0)
    failures = []

    for run_dir in selected_runs:
        st.markdown(f"**{run_dir.name}**")
        clean_out_dir(run_dir, None)
        log_box = st.empty()
        lines = []
        run_failed = False
        for label, cmd in stage_commands(run_dir, None, targets_path, show_metrics_box):
            lines.append(f"── {label} ──")
            log_box.code("\n".join(lines[-60:]))
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                lines.append(line.rstrip())
                log_box.code("\n".join(lines[-60:]))
            proc.wait()
            done += 1
            progress.progress(done / total_stages)
            if proc.returncode != 0:
                st.error(f"{run_dir.name}: '{label}' failed (exit {proc.returncode})")
                run_failed = True
                failures.append(run_dir)
                break
        if not run_failed:
            st.success(f"{run_dir.name}: done")

    if failures:
        st.error(f"{len(failures)} of {len(selected_runs)} run(s) failed.")
    else:
        st.success("All runs completed.")

    # persist so the preview/download section below survives reruns
    # triggered by clicking a download button
    st.session_state.completed_runs = [str(r) for r in selected_runs if r not in failures]

# --- preview + download ---------------------------------------------------------
completed = [Path(p) for p in st.session_state.get("completed_runs", [])]
if completed:
    st.subheader("4. Results")
    for run_dir in completed:
        out_dir = run_dir / "output"
        if not out_dir.is_dir():
            continue
        st.markdown(f"**{run_dir.name}**")
        st.caption(f"Saved to: `{out_dir}`")

        pngs = sorted(out_dir.glob("spectrum_mz*.png"))
        if pngs:
            cols = st.columns(min(4, len(pngs)))
            for i, png in enumerate(pngs):
                col = cols[i % len(cols)]
                col.image(str(png), caption=png.stem.replace("spectrum_mz", "m/z "))
                col.download_button(
                    "Download PNG", data=png.read_bytes(), file_name=png.name,
                    mime="image/png", key=f"dl_{run_dir}_{png.name}",
                )

        csv_path = out_dir / "peak_metrics.csv"
        if csv_path.is_file():
            st.dataframe(pd.read_csv(csv_path), use_container_width=True)

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in out_dir.iterdir():
                if f.is_file():
                    zf.write(f, arcname=f.name)
        st.download_button(
            f"Download all outputs for {run_dir.name} (.zip)",
            data=zip_buf.getvalue(),
            file_name=f"{run_dir.name}_output.zip",
            mime="application/zip",
            key=f"dlzip_{run_dir}",
        )
        st.divider()
