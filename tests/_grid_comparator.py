"""grid_comparator.py

Interactive SOFA measurement-grid comparator.

Usage
-----
    python tests/grid_comparator.py [sofa_file ...]

Drag and drop additional .sofa files onto the window at any time.

Panels
------
  A (left)   — toggle: Raw Grid  |  Original Weighting (bare ConvexHull)
  B (right)  — New Weighting (ghost-node ConvexHull, production method)

Controls
--------
  Sync rotation     — rotating either panel rotates the other
  Reset view        — snap both panels back to the default viewpoint
  Shared color scale — both weighted panels share one viridis range
  Export PNGs…      — write raw-grid + both weighting PNGs to a folder
"""

import json
import os
import sys

# ── path setup so imports work regardless of CWD ──────────────────────────────
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

# PySide6 must be imported before matplotlib's Qt backend to avoid a DLL
# resolution failure for PySide6.QtGui on Windows.
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QListWidget, QPushButton, QLabel, QComboBox, QCheckBox,
    QDoubleSpinBox, QFileDialog, QSizePolicy, QFrame,
)
from PySide6.QtCore import Qt

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers '3d' projection

from compare_weighting_grids import (
    spherical_to_cartesian,
    _clean_source_pos,
    geometric_weights_original,
    geometric_weights_new,
    geometric_weights_uniform,
    weight_cv,
    weights_for_mode,
    NODE_SIZE,
    plot_weighted_grid,
)

try:
    import sofar as sf
except ImportError:
    print("[ERROR] sofar not installed.  Run: pip install sofar")
    sys.exit(1)

try:
    from core.sofa_utils import get_sofa_metadata
except ImportError:
    def get_sofa_metadata(path):           # lightweight fallback
        try:
            s = sf.read_sofa(path)
            return {
                'samplerate': float(s.Data_SamplingRate),
                'length':     s.Data_IR.shape[2],
                'points':     s.Data_IR.shape[0],
            }
        except Exception:
            return None

# ─────────────────────────── scale-defaults loader ───────────────────────────

_DEFAULTS_PATH = os.path.join(_REPO_ROOT, 'core', 'default_settings.json')


def _load_scale_defaults():
    """Read weighted_grid_scale_* from default_settings.json (read-only).

    Returns (mode, low, high) — falls back to ('auto', 0.001, 0.003) on any
    error so the comparator still works even when run outside the repo.
    """
    try:
        with open(_DEFAULTS_PATH, 'r', encoding='utf-8') as fh:
            adv = json.load(fh)['dfhrtf']['advanced']
        return (
            str(adv.get('weighted_grid_scale_mode', 'auto')).lower(),
            float(adv.get('weighted_grid_scale_low', 0.0001)),
            float(adv.get('weighted_grid_scale_high', 0.0025)),
        )
    except Exception:
        return ('auto', 0.0001, 0.0025)


# ─────────────────────────── constants ───────────────────────────────────────

DEFAULT_AZIM = -60
DEFAULT_ELEV = 20

RAW_KEPT_COLOR = '#4A90D9'   # blue — cleaned, weighted nodes
RAW_EXCL_COLOR = '#E74C3C'   # red  — excluded by the cleaning step

PANEL_A_MODES = ["Raw Grid", "Original Weighting", "Uniform"]
PANEL_B_MODES = ["Auto", "Geometric)", "Uniform"]
BALANCE_CV    = 0.12   # matches generate_dfhrtf.py default


# ─────────────────────────── per-file cache ───────────────────────────────────

class _SofaCache:
    """Lazy-loads SOFA files and caches all derived data per path."""

    def __init__(self):
        self._store: dict = {}

    def get(self, path: str) -> dict | None:
        if path not in self._store:
            self._store[path] = self._load(path)
        return self._store[path]

    @staticmethod
    def _load(path: str) -> dict | None:
        try:
            sofa = sf.read_sofa(path)
        except Exception as e:
            print(f"[ERROR] Could not load {os.path.basename(path)}: {e}")
            return None

        pos = sofa.SourcePosition                    # (M, 3) — az, el, r
        cart_all = spherical_to_cartesian(1.0, pos[:, 0], pos[:, 1])

        _cart_clean, final_idx = _clean_source_pos(pos)

        all_idx  = np.arange(len(pos))
        excl_mask = np.ones(len(pos), dtype=bool)
        excl_mask[final_idx] = False
        excl_idx = all_idx[excl_mask]

        # Dominant radius (same logic as cleaning prologue)
        radii = np.round(pos[:, 2], decimals=3)
        vals, counts = np.unique(radii, return_counts=True)
        dominant_r = float(vals[np.argmax(counts)])

        w_orig    = geometric_weights_original(pos)
        w_new     = geometric_weights_new(pos)
        w_uniform = geometric_weights_uniform(pos)
        geo_cv    = weight_cv(w_new)
        w_auto, auto_verdict, _ = weights_for_mode(pos, 'auto', BALANCE_CV)

        return {
            'pos':          pos,
            'cart_all':     cart_all,
            'final_idx':    final_idx,
            'excl_idx':     excl_idx,
            'w_orig':       w_orig,
            'w_new':        w_new,
            'w_uniform':    w_uniform,
            'w_auto':       w_auto,
            'auto_verdict': auto_verdict,   # 'geometric' or 'uniform'
            'geo_weight_cv': geo_cv,
            'dominant_r':   dominant_r,
            'n_total':      len(pos),
            'n_kept':       len(final_idx),
            'n_excl':       len(excl_idx),
        }


# ─────────────────────────── embedded canvas frame ────────────────────────────

class _GridPanel(QFrame):
    """Thin QFrame wrapper: a header widget + a matplotlib 3-D canvas."""

    def __init__(self, header: QWidget, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(header)

        self.fig    = plt.figure()
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self.canvas)

        self.ax = None   # created on first render

    # ── helpers ───────────────────────────────────────────────────────────────

    def get_or_create_ax(self):
        if self.ax is None:
            self.ax = self.fig.add_subplot(111, projection='3d')
        return self.ax

    def clear(self):
        self.fig.clear()
        self.ax = None

    def reset_view(self):
        if self.ax is not None:
            self.ax.view_init(elev=DEFAULT_ELEV, azim=DEFAULT_AZIM)
            self.canvas.draw_idle()


# ─────────────────────────── main window ──────────────────────────────────────

class GridComparator(QMainWindow):

    def __init__(self, preload_paths=None):
        super().__init__()
        self.setWindowTitle("SOFA Grid Comparator")
        self.resize(1440, 780)
        self.setAcceptDrops(True)

        self._cache        = _SofaCache()
        self._current_data: dict | None = None
        self._sync_cid_a   = None
        self._sync_cid_b   = None

        self._build_ui()

        for p in (preload_paths or []):
            self._add_file(p)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root_widget = QWidget()
        self.setCentralWidget(root_widget)
        root = QVBoxLayout(root_widget)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # Toolbar row — fixed height so it stays slim
        toolbar_widget = QWidget()
        toolbar_widget.setFixedHeight(28)
        toolbar = QHBoxLayout(toolbar_widget)
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(6)

        self.chk_sync = QCheckBox("Sync rotation")
        self.chk_sync.setChecked(True)
        toolbar.addWidget(self.chk_sync)

        btn_reset = QPushButton("Reset view")
        btn_reset.setFixedHeight(22)
        btn_reset.clicked.connect(self._reset_view)
        toolbar.addWidget(btn_reset)

        self.chk_shared = QCheckBox("Shared color scale")
        self.chk_shared.setChecked(True)
        self.chk_shared.stateChanged.connect(self._rerender)
        toolbar.addWidget(self.chk_shared)

        # Scale mode: Auto (percentile) vs Manual (fixed low/high from JSON)
        _scale_mode, _scale_low, _scale_high = _load_scale_defaults()
        toolbar.addWidget(QLabel("Scale:"))
        self.combo_scale = QComboBox()
        self.combo_scale.setFixedHeight(22)
        self.combo_scale.addItems(["Auto", "Manual"])
        self.combo_scale.setCurrentText("Manual" if _scale_mode == "manual" else "Auto")
        self.combo_scale.currentTextChanged.connect(self._on_scale_mode_changed)
        toolbar.addWidget(self.combo_scale)

        lbl_low = QLabel("Low:")
        toolbar.addWidget(lbl_low)
        self.spin_low = QDoubleSpinBox()
        self.spin_low.setFixedHeight(22)
        self.spin_low.setFixedWidth(100)
        self.spin_low.setDecimals(5)
        self.spin_low.setRange(0.0, 1.0)
        self.spin_low.setSingleStep(0.0001)
        self.spin_low.setValue(_scale_low)
        self.spin_low.valueChanged.connect(self._rerender)
        toolbar.addWidget(self.spin_low)

        lbl_high = QLabel("High:")
        toolbar.addWidget(lbl_high)
        self.spin_high = QDoubleSpinBox()
        self.spin_high.setFixedHeight(22)
        self.spin_high.setFixedWidth(100)
        self.spin_high.setDecimals(5)
        self.spin_high.setRange(0.0, 1.0)
        self.spin_high.setSingleStep(0.0001)
        self.spin_high.setValue(_scale_high)
        self.spin_high.valueChanged.connect(self._rerender)
        toolbar.addWidget(self.spin_high)

        # Set initial enabled state
        self._apply_scale_mode_state()

        btn_export = QPushButton("Export PNGs…")
        btn_export.setFixedHeight(22)
        btn_export.clicked.connect(self._export_pngs)
        toolbar.addWidget(btn_export)

        toolbar.addStretch()

        self.lbl_meta = QLabel("No file selected")
        self.lbl_meta.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        toolbar.addWidget(self.lbl_meta)

        root.addWidget(toolbar_widget)

        # Main splitter: file list | panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left: file list ───────────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(210)
        left.setMaximumWidth(310)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(4)

        ll.addWidget(QLabel("SOFA Files  (drag & drop)"))

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.file_list.currentRowChanged.connect(self._on_file_selected)
        ll.addWidget(self.file_list)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add Files…")
        btn_add.clicked.connect(self._add_files_dialog)
        btn_row.addWidget(btn_add)

        btn_rm = QPushButton("Remove")
        btn_rm.clicked.connect(self._remove_selected)
        btn_row.addWidget(btn_rm)

        btn_clr = QPushButton("Clear")
        btn_clr.clicked.connect(self.file_list.clear)
        btn_row.addWidget(btn_clr)

        ll.addLayout(btn_row)
        splitter.addWidget(left)

        # ── Right: two panels ─────────────────────────────────────────────────
        right = QWidget()
        rl = QHBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)

        # Panel A — toggleable
        self.combo_a = QComboBox()
        self.combo_a.addItems(PANEL_A_MODES)
        self.combo_a.currentIndexChanged.connect(self._on_combo_changed)
        self.panel_a = _GridPanel(header=self.combo_a)
        rl.addWidget(self.panel_a)

        # Panel B — toggleable
        self.combo_b = QComboBox()
        self.combo_b.addItems(PANEL_B_MODES)
        self.combo_b.currentIndexChanged.connect(self._on_combo_changed)
        self.panel_b = _GridPanel(header=self.combo_b)
        rl.addWidget(self.panel_b)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # ── File management ───────────────────────────────────────────────────────

    def _add_file(self, path: str):
        path = os.path.normpath(os.path.abspath(path))
        if not path.lower().endswith('.sofa') or not os.path.isfile(path):
            return
        existing = {self.file_list.item(i).text()
                    for i in range(self.file_list.count())}
        if path in existing:
            return
        self.file_list.addItem(path)
        if self.file_list.count() == 1:
            self.file_list.setCurrentRow(0)

    def _add_files_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select SOFA files", "", "SOFA Files (*.sofa *.SOFA)"
        )
        for f in files:
            self._add_file(f)

    def _remove_selected(self):
        row = self.file_list.currentRow()
        if row >= 0:
            self.file_list.takeItem(row)

    # ── Drag and drop ─────────────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        mime = event.mimeData()
        if mime.hasUrls() and any(
            u.toLocalFile().lower().endswith('.sofa') for u in mime.urls()
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith('.sofa'):
                self._add_file(path)
        event.acceptProposedAction()

    # ── Selection → render ────────────────────────────────────────────────────

    def _on_file_selected(self, row: int):
        if row < 0:
            self._current_data = None
            self._clear_panels()
            self.lbl_meta.setText("No file selected")
            return

        path = self.file_list.item(row).text()
        self.lbl_meta.setText(f"Loading {os.path.basename(path)}…")
        QApplication.processEvents()

        data = self._cache.get(path)
        if data is None:
            self.lbl_meta.setText(f"Error loading: {os.path.basename(path)}")
            return

        self._current_data = data
        meta = get_sofa_metadata(path)
        sr_str = f"{int(meta['samplerate'])} Hz" if meta else "?"
        cv_pct = data['geo_weight_cv'] * 100
        verdict = data['auto_verdict']
        self.lbl_meta.setText(
            f"{os.path.basename(path)}  ·  "
            f"M={data['n_total']}  SR={sr_str}  r={data['dominant_r']:g} m  "
            f"kept={data['n_kept']}  excl={data['n_excl']}  "
            f"geo-CV={cv_pct:.1f}%  auto→{verdict}"
        )
        self._rerender()

    def _on_combo_changed(self, _idx: int):
        self._rerender()

    def _clear_panels(self):
        self._disconnect_sync()
        for panel in (self.panel_a, self.panel_b):
            panel.clear()
            panel.canvas.draw_idle()

    def _rerender(self):
        data = self._current_data
        if data is None:
            return
        self._disconnect_sync()
        self._render_a(data)
        self._render_b(data)
        self._connect_sync()

    # ── Scale-mode helpers ────────────────────────────────────────────────────

    def _apply_scale_mode_state(self):
        """Enable/disable spinboxes and the Shared checkbox based on scale mode."""
        manual = self.combo_scale.currentText() == "Manual"
        self.spin_low.setEnabled(manual)
        self.spin_high.setEnabled(manual)
        # In Manual mode the range is already fixed/shared; the Shared checkbox
        # only applies to Auto mode so disable it when Manual is active.
        self.chk_shared.setEnabled(not manual)

    def _on_scale_mode_changed(self, _text: str):
        self._apply_scale_mode_state()
        self._rerender()

    def _manual_clim(self) -> tuple[float, float]:
        return float(self.spin_low.value()), float(self.spin_high.value())

    # ── Color-scale helpers ───────────────────────────────────────────────────

    def _shared_clim(self, w1: np.ndarray, w2: np.ndarray) -> tuple[float, float]:
        """Combined 1st–99th percentile across two weight arrays."""
        combined = np.concatenate([w1, w2])
        combined = combined[combined > 0]
        if len(combined) == 0:
            return 0.0, 1.0
        return float(np.percentile(combined, 1)), float(np.percentile(combined, 99))

    def _auto_clim(self, w: np.ndarray) -> tuple[float, float]:
        w_pos = w[w > 0]
        if len(w_pos) == 0:
            return 0.0, 1.0
        return float(np.percentile(w_pos, 1)), float(np.percentile(w_pos, 99))

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _init_ax(self, ax):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=DEFAULT_ELEV, azim=DEFAULT_AZIM)

    def _render_a(self, data: dict):
        mode  = self.combo_a.currentText()
        panel = self.panel_a
        panel.clear()
        ax = panel.get_or_create_ax()

        if mode == "Raw Grid":
            cart     = data['cart_all']
            kept_idx = data['final_idx']
            excl_idx = data['excl_idx']

            ax.scatter(
                cart[kept_idx, 0], cart[kept_idx, 1], cart[kept_idx, 2],
                c=RAW_KEPT_COLOR, s=NODE_SIZE, alpha=0.8,
                label=f"Kept ({len(kept_idx)})"
            )
            if len(excl_idx) > 0:
                ax.scatter(
                    cart[excl_idx, 0], cart[excl_idx, 1], cart[excl_idx, 2],
                    c=RAW_EXCL_COLOR, s=NODE_SIZE * 1.5, alpha=0.95,
                    label=f"Excluded ({len(excl_idx)})"
                )
            ax.legend(loc='upper left', fontsize=8)
            ax.set_title(f"Raw Grid — {data['n_total']} nodes")

        else:
            w_a  = data['w_orig'] if mode == "Original Weighting" else data['w_uniform']
            w_b  = self._current_b_weights(data)   # for shared scale
            mask = w_a > 0
            if self.combo_scale.currentText() == "Manual":
                vmin, vmax = self._manual_clim()
            else:
                vmin, vmax = (
                    self._shared_clim(w_a, w_b) if self.chk_shared.isChecked() and w_b is not None
                    else self._auto_clim(w_a)
                )
            sc = ax.scatter(
                data['cart_all'][mask, 0],
                data['cart_all'][mask, 1],
                data['cart_all'][mask, 2],
                c=w_a[mask], s=NODE_SIZE, cmap='viridis', alpha=0.85,
                vmin=vmin, vmax=vmax
            )
            panel.fig.colorbar(sc, ax=ax, label='Solid-angle weight',
                               shrink=0.6, pad=0.1)
            label = "Original Weighting" if mode == "Original Weighting" else "Uniform (1/N)"
            ax.set_title(f"{label} — {int(mask.sum())} nodes")

        self._init_ax(ax)
        panel.canvas.draw_idle()

    def _current_b_weights(self, data: dict):
        """Return the weight array Panel B is currently showing, for shared scale."""
        mode = self.combo_b.currentText()
        if mode == "New (Geometric)":
            return data['w_new']
        if mode == "Uniform":
            return data['w_uniform']
        # Auto
        return data['w_auto']

    def _current_a_weights(self, data: dict):
        """Return the weight array Panel A is currently showing (None for Raw Grid)."""
        mode = self.combo_a.currentText()
        if mode == "Raw Grid":
            return None
        if mode == "Original Weighting":
            return data['w_orig']
        return data['w_uniform']

    def _render_b(self, data: dict):
        mode  = self.combo_b.currentText()
        panel = self.panel_b
        panel.clear()
        ax = panel.get_or_create_ax()

        if mode == "New (Geometric)":
            w, title = data['w_new'], "New (Geometric)"
        elif mode == "Uniform":
            w, title = data['w_uniform'], "Uniform (1/N)"
        else:  # "Auto"
            w = data['w_auto']
            resolved = data['auto_verdict']
            cv_pct = data['geo_weight_cv'] * 100
            title = f"Auto → {resolved}  (geo-CV={cv_pct:.1f}%)"

        mask = w > 0
        w_a  = self._current_a_weights(data)
        if self.combo_scale.currentText() == "Manual":
            vmin, vmax = self._manual_clim()
        else:
            vmin, vmax = (
                self._shared_clim(w, w_a) if self.chk_shared.isChecked() and w_a is not None
                else self._auto_clim(w)
            )

        sc = ax.scatter(
            data['cart_all'][mask, 0],
            data['cart_all'][mask, 1],
            data['cart_all'][mask, 2],
            c=w[mask], s=NODE_SIZE, cmap='viridis', alpha=0.85,
            vmin=vmin, vmax=vmax
        )
        panel.fig.colorbar(sc, ax=ax, label='Solid-angle weight',
                           shrink=0.6, pad=0.1)
        ax.set_title(f"{title} — {int(mask.sum())} nodes")

        self._init_ax(ax)
        panel.canvas.draw_idle()

    # ── Synced rotation ───────────────────────────────────────────────────────

    def _disconnect_sync(self):
        pairs = [('_sync_cid_a', self.panel_a), ('_sync_cid_b', self.panel_b)]
        for attr, panel in pairs:
            cid = getattr(self, attr, None)
            if cid is not None:
                try:
                    panel.canvas.mpl_disconnect(cid)
                except Exception:
                    pass
                setattr(self, attr, None)

    def _connect_sync(self):
        self._disconnect_sync()
        self._sync_cid_a = self.panel_a.canvas.mpl_connect(
            'draw_event',
            lambda _e: self._sync_view(self.panel_a, self.panel_b)
        )
        self._sync_cid_b = self.panel_b.canvas.mpl_connect(
            'draw_event',
            lambda _e: self._sync_view(self.panel_b, self.panel_a)
        )

    def _sync_view(self, src: _GridPanel, dst: _GridPanel):
        """Copy src view → dst after a draw, only when the views differ."""
        if not self.chk_sync.isChecked():
            return
        if src.ax is None or dst.ax is None:
            return

        src_azim, src_elev = src.ax.azim, src.ax.elev
        dst_azim, dst_elev = dst.ax.azim, dst.ax.elev

        # Skip if already in sync (prevents feedback loop)
        if abs(src_azim - dst_azim) < 0.1 and abs(src_elev - dst_elev) < 0.1:
            return

        dst.ax.view_init(elev=src_elev, azim=src_azim)
        dst.canvas.draw_idle()

    # ── Controls ──────────────────────────────────────────────────────────────

    def _reset_view(self):
        for panel in (self.panel_a, self.panel_b):
            panel.reset_view()

    def _export_pngs(self):
        data = self._current_data
        row  = self.file_list.currentRow()
        if data is None or row < 0:
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Select export folder")
        if not out_dir:
            return

        path = self.file_list.item(row).text()
        stem = os.path.splitext(os.path.basename(path))[0]

        print(f"\nExporting PNGs for {stem} → {out_dir}")

        # Original weighting
        plot_weighted_grid(
            data['pos'], data['w_orig'],
            title="Original weighting (bare ConvexHull)",
            out_path=os.path.join(out_dir, f"{stem}_weighting_original.png"),
        )

        # New weighting
        plot_weighted_grid(
            data['pos'], data['w_new'],
            title="New weighting (ghost-node ConvexHull)",
            out_path=os.path.join(out_dir, f"{stem}_weighting_new.png"),
        )

        # Raw grid (two-color: kept / excluded)
        cart     = data['cart_all']
        kept_idx = data['final_idx']
        excl_idx = data['excl_idx']

        fig = plt.figure(figsize=(8, 8))
        ax  = fig.add_subplot(111, projection='3d')
        ax.scatter(
            cart[kept_idx, 0], cart[kept_idx, 1], cart[kept_idx, 2],
            c=RAW_KEPT_COLOR, s=NODE_SIZE, alpha=0.8,
            label=f"Kept ({len(kept_idx)})"
        )
        if len(excl_idx) > 0:
            ax.scatter(
                cart[excl_idx, 0], cart[excl_idx, 1], cart[excl_idx, 2],
                c=RAW_EXCL_COLOR, s=NODE_SIZE * 1.5, alpha=0.95,
                label=f"Excluded ({len(excl_idx)})"
            )
        ax.legend(fontsize=8)
        ax.set_title(f"Raw Grid — {data['n_total']} nodes")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_box_aspect([1, 1, 1])
        raw_path = os.path.join(out_dir, f"{stem}_raw_grid.png")
        fig.savefig(raw_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"   [+] Saved: {os.path.basename(raw_path)}")

        print("Export complete.")


# ─────────────────────────── entry point ──────────────────────────────────────

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    preload = [
        p for p in sys.argv[1:]
        if p.lower().endswith('.sofa') and os.path.isfile(p)
    ]
    win = GridComparator(preload_paths=preload or None)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
