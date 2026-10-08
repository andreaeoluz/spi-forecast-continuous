#!/usr/bin/env python3
"""generate_study_area_map.py - Elegant study-area map for both drought
forecasting papers, built from the IBGE state-level (BR_UF_2023) and South
America (Lim_america_do_sul_2021) shapefiles. Replaces the ArcGIS Layout2.pdf
mockup previously used in Paper.Regression/images and Paper.Binary/images.

Design choices, each addressing a specific critique of the original ArcGIS map:
  - Region palette reuses REGION_COLORS from analyze_pq_sensitivity.py, so
    the map shares a visual identity with every result figure in both papers.
  - English region names matching the papers' own terminology exactly
    ("Center-West", not "Central West").
  - No relief/bathymetry basemap: flat neutral background so the region
    colors (the actual data) are not competing with texture, and no
    on-image basemap-provider attribution.
  - Two-tier line-weight hierarchy: thin white state borders (context) vs.
    a bold dark region outline (the actual analysis unit).
  - Legend placed in the empty Atlantic space, not overlapping any polygon.
  - Geodesically-computed scale bar (pyproj.Geod at the map's central
    latitude) with English thousand-separators, instead of ArcGIS's
    period-separated "1.180".
  - Solid cone-shaped north arrow, matching the original layout's icon.

Usage: python generate_study_area_map.py
Output: study_area_map.pdf/.png written into both papers' images/ folders.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyproj import Geod

IBGE_DATA_DIR = Path(r"C:\Users\User\Desktop\projects_old\study_area\MapBin\BaseDados")
OUT_DIRS = [
    Path(r"C:\Users\User\Desktop\projects_tese\drought_forecast_regression\Paper.Regression\images"),
    Path(r"C:\Users\User\Desktop\projects_tese\drought_forecast_binary\Paper.Binary\images"),
]

REGION_COLORS = {
    "South": "#2E86AB",
    "Southeast": "#A23B72",
    "Northeast": "#F18F01",
    "Center-West": "#73AB84",
    "North": "#C73E1D",
}
PT_TO_EN = {
    "Sul": "South", "Sudeste": "Southeast", "Nordeste": "Northeast",
    "Centro-oeste": "Center-West", "Norte": "North",
}
REGION_ORDER = ["North", "Northeast", "Center-West", "Southeast", "South"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 400,
})


def build_figure():
    uf = gpd.read_file(IBGE_DATA_DIR / "BR_UF_2023.shp")
    uf["region_en"] = uf["NM_REGIAO"].map(PT_TO_EN)

    # Dissolve from the raw (unsimplified) geometry first, so neighboring
    # states' shared edges cancel out cleanly; only then simplify the
    # dissolved region outlines. This also smooths away the coastal
    # micro-island fragments (Maranhão alone has 61 separate polygon parts)
    # that otherwise render as a "dotted" border at country scale.
    regions = uf.dissolve(by="region_en", as_index=False)
    regions["geometry"] = regions["geometry"].simplify(0.025, preserve_topology=True)
    uf["geometry"] = uf["geometry"].simplify(0.025, preserve_topology=True)

    sa = gpd.read_file(IBGE_DATA_DIR / "Lim_america_do_sul_2021.shp")
    sa["geometry"] = sa["geometry"].simplify(0.02, preserve_topology=True)
    neighbors = sa[sa["nome"] != "Brasil"]

    fig, ax = plt.subplots(figsize=(8.5, 8.5))

    # Neutral context: neighboring countries, no basemap texture.
    neighbors.plot(ax=ax, facecolor="#EDEBE6", edgecolor="#BDB9B0", linewidth=0.5, zorder=1)

    # States: filled by region color, thin light borders (context tier).
    for region_en in REGION_ORDER:
        uf[uf["region_en"] == region_en].plot(
            ax=ax, facecolor=REGION_COLORS[region_en], edgecolor="white",
            linewidth=0.5, zorder=2,
        )

    # Bold region-level outline (the actual analysis unit).
    regions.boundary.plot(ax=ax, edgecolor="#2A2A2A", linewidth=1.4, zorder=3)

    minx, miny, maxx, maxy = -76, -34.5, -28, 6.5
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect(1.0)  # matches the plain lat/lon (unprojected) framing of the original
    ax.set_facecolor("#F7FBFF")  # flat, very light "ocean" tint instead of bathymetry

    # Degree tick labels, same convention as the original.
    xt = range(-70, -25, 5)
    yt = range(-30, 10, 5)
    ax.set_xticks(list(xt))
    ax.set_xticklabels([f"{abs(v)}°W" for v in xt], fontsize=9)
    ax.set_yticks(list(yt))
    ax.set_yticklabels([f"{v}°N" if v > 0 else (f"{abs(v)}°S" if v < 0 else "0°") for v in yt], fontsize=9)
    ax.tick_params(length=3, color="#666666")
    for spine in ax.spines.values():
        spine.set_edgecolor("#666666")
        spine.set_linewidth(0.8)

    # --- Legend, placed over the empty Atlantic, not overlapping any polygon ---
    handles = [mpatches.Patch(facecolor=REGION_COLORS[r], edgecolor="#2A2A2A", linewidth=0.8, label=r)
               for r in REGION_ORDER]
    legend = ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.99, 0.02),
                        frameon=True, framealpha=0.95, edgecolor="#999999",
                        title="Region", fontsize=9.5, title_fontsize=10.5,
                        handlelength=1.1, handleheight=1.1, borderpad=0.7,
                        labelspacing=0.5)
    legend.get_frame().set_linewidth(0.7)

    # --- North arrow: solid cone/triangle, matching the original ArcGIS icon ---
    cone_x, cone_base_y, cone_tip_y, cone_half_w = -30.0, 3.1, 5.3, 0.38
    cone = mpatches.Polygon(
        [(cone_x, cone_tip_y), (cone_x - cone_half_w, cone_base_y), (cone_x + cone_half_w, cone_base_y)],
        closed=True, facecolor="#2A2A2A", edgecolor="#2A2A2A", linewidth=0.5, zorder=5,
    )
    ax.add_patch(cone)
    ax.text(cone_x, cone_tip_y + 0.35, "N", ha="center", va="bottom", fontsize=13,
            fontweight="bold", color="#2A2A2A")

    # --- Geodesically accurate scale bar (computed, not eyeballed) ---
    geod = Geod(ellps="WGS84")
    center_lat = -15.0
    bar_lon0 = -73.5
    lon2, _, _ = geod.fwd(bar_lon0, center_lat, 90, 500_000)
    deg_per_500km = lon2 - bar_lon0
    bar_y = -32.5
    # two segments: 0-500 (black), 500-1000 (white), both outlined
    seg1 = mpatches.Rectangle((bar_lon0, bar_y), deg_per_500km, 0.35,
                               facecolor="#2A2A2A", edgecolor="#2A2A2A", linewidth=0.7, zorder=4)
    seg2 = mpatches.Rectangle((bar_lon0 + deg_per_500km, bar_y), deg_per_500km, 0.35,
                               facecolor="white", edgecolor="#2A2A2A", linewidth=0.7, zorder=4)
    ax.add_patch(seg1)
    ax.add_patch(seg2)
    for x, lab in [(bar_lon0, "0"), (bar_lon0 + deg_per_500km, "500"),
                   (bar_lon0 + 2 * deg_per_500km, "1,000 km")]:
        ax.text(x, bar_y - 0.9, lab, ha="center", fontsize=8.5, color="#2A2A2A")

    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def main():
    fig = build_figure()
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "study_area_map"
        fig.savefig(f"{out}.pdf", bbox_inches="tight", pad_inches=0.15)
        fig.savefig(f"{out}.png", bbox_inches="tight", pad_inches=0.15)
        print(f"Saved: {out}.pdf / .png")
    plt.close(fig)


if __name__ == "__main__":
    main()
