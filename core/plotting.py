"""
Shared plotting helpers for SOFA Buddy.

These generate matplotlib figures in-process (e.g. from the averaging post-step
running inside the worker thread). The Agg backend is used to avoid touching the
Qt main thread.
"""


def smoothing_display_name(smoothing, window=None):
    """Render a smoothing setting the way the GUI dropdown shows it.

    Accepts the dropdown value ('Variable', '1/24') as well as the backend
    CLI form ('variable', '24'), so both graph paths label identically.
    With a *window* ('gaussian' / 'flat') it is appended: 'Variable (Gaussian)'.
    """
    s = str(smoothing).strip()
    name = "Variable" if s.lower() == "variable" else f"1/{s.split('/')[-1]}"
    if window is not None:
        name += f" ({str(window).strip().capitalize()})"
    return name


def graph_subtitle(tilt, smoothing=None, window=None):
    """Second line of a DFHRTF graph title: tilt, then smoothing (and its
    window) if known."""
    line = f"Spectral Tilt: {tilt} dB/oct"
    if smoothing is not None:
        line += f"; Smoothing: {smoothing_display_name(smoothing, window)}"
    return line


def plot_diffuse_field(freqs, val_l, val_r, out_path, tilt,
                       title="Diffuse Field HRTF (Averaged)",
                       label_l="Left Ear (avg)",
                       label_r="Right Ear (avg)",
                       smoothing=None, smoothing_window=None):
    """
    Render a diffuse-field HRTF graph with Left (solid) and Right (dashed) channels
    and save it to *out_path*.

    Styling is intentionally identical to the per-SOFA 'df' graph produced by
    core/scripts/generate_dfhrtf.py so averaged and per-SOFA graphs look consistent.
    """
    import matplotlib
    matplotlib.use("Agg")          # must precede pyplot import; safe in worker thread
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.semilogx(freqs, val_l, label=label_l, linewidth=2, alpha=0.8)
    plt.semilogx(freqs, val_r, label=label_r, linewidth=2, alpha=0.8, linestyle='--')
    plt.title(f"{title}\n{graph_subtitle(tilt, smoothing, smoothing_window)}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.ylim(-25, 25)
    plt.xlim(20, 20000)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.savefig(out_path, dpi=150)
    plt.close()
