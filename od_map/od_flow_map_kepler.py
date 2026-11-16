import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import geopandas as gpd
import pandas as pd
from keplergl import KeplerGl

# Interactive kepler.gl version of plot_od_flow_map.py (run with .venv311, which has keplergl)
OD_CSV = "od_kabupaten_monthly.csv"
KAP_WKT = "../release/kapanewon_wkt.csv"
# Base map: CARTO styles need no token -- "dark-matter" (dark), "positron" (light), "voyager".
# The Mapbox styles ("dark", "light", "muted", "muted_night", "satellite") need a Mapbox token
# and render as a blank background without one. Override with MAP_STYLE=positron etc.
MAP_STYLE = os.environ.get("MAP_STYLE", "dark-matter")
OUTPUT_HTML = f"od_flow_map_kabupaten_kepler_{MAP_STYLE}.html"

# Layer colours per base map: royalblue / lightblue as in the matplotlib figures on light maps,
# brighter blues and dim boundaries on dark maps so the flows stand out
DARK_STYLES = {"dark-matter", "dark-matter-nolabels", "dark", "muted_night", "satellite"}
if MAP_STYLE in DARK_STYLES:
    THEME = {"area_fill": [70, 70, 70], "area_opacity": 0.25, "area_stroke": [150, 150, 150],
             "arc": [100, 149, 237], "arc_target": [173, 216, 230],
             "point": [173, 216, 230], "point_stroke": [100, 149, 237]}
else:
    THEME = {"area_fill": [240, 240, 240], "area_opacity": 0.3, "area_stroke": [128, 128, 128],
             "arc": [65, 105, 225], "arc_target": [65, 105, 225],
             "point": [173, 216, 230], "point_stroke": [65, 105, 225]}

# Kabupaten/kota polygons and label points (WGS84)
kap = pd.read_csv(KAP_WKT, dtype={"KAB_CODE": str, "KEC_CODE": str})
kab = gpd.GeoDataFrame(kap, geometry=gpd.GeoSeries.from_wkt(kap["wkt"]), crs="EPSG:4326").to_crs("EPSG:32749")
kab = kab.dissolve(by=["KAB_CODE", "KAB_KOTA"], as_index=False)[["KAB_CODE", "KAB_KOTA", "geometry"]]
kab["geometry"] = kab.buffer(5).buffer(-5)  # close sliver gaps between dissolved kalurahan (metres)
pts = kab.set_index("KAB_CODE").geometry.representative_point().to_crs("EPSG:4326")
kab = kab.to_crs("EPSG:4326")
kab["name"] = kab["KAB_KOTA"].str.title().replace({"Yogyakarta": "Kota Yogyakarta"})
name = kab.set_index("KAB_CODE")["name"]

# Whole-period flows (device-months)
od = pd.read_csv(OD_CSV, dtype={"origin_kab_code": str, "dest_kab_code": str})
flows = od.groupby(["origin_kab_code", "dest_kab_code"])["n_maids"].sum()

# One arc per pair, with both directions in the tooltip
codes = sorted(name.index)
arcs = pd.DataFrame([{
    "kabupaten_a": name[a], "kabupaten_b": name[b],
    "a_lat": pts[a].y, "a_lon": pts[a].x, "b_lat": pts[b].y, "b_lon": pts[b].x,
    "flow_a_to_b": int(flows[(a, b)]), "flow_b_to_a": int(flows[(b, a)]),
    "flow_total": int(flows[(a, b)] + flows[(b, a)]),
} for i, a in enumerate(codes) for b in codes[i + 1:]])

within = pd.DataFrame([{"kabupaten": name[c], "lat": pts[c].y, "lon": pts[c].x,
                        "flow_within": int(flows[(c, c)])} for c in codes])

# Float copies for sizing: keplergl 0.3.7 rendered integer size fields in reverse order (largest flow thinnest)
arcs["size_total"] = arcs["flow_total"].astype(float)
within["size_within"] = within["flow_within"].astype(float)

areas = kab[["name", "geometry"]].to_json()

def tooltip(fields):
    return [{"name": f, "format": None} for f in fields]

config = {
    "version": "v1",
    "config": {
        "visState": {
            "layers": [
                {"id": "kabupaten", "type": "geojson",
                 "config": {"dataId": "kabupaten", "label": "Kabupaten/kota", "color": THEME["area_fill"],
                            "columns": {"geojson": "_geojson"}, "isVisible": True,
                            "visConfig": {"opacity": THEME["area_opacity"], "strokeOpacity": 0.9, "thickness": 1,
                                          "strokeColor": THEME["area_stroke"], "filled": True, "stroked": True}}},
                {"id": "within", "type": "point",
                 "config": {"dataId": "within", "label": "Flow within kabupaten (device-months)",
                            "color": THEME["point"], "columns": {"lat": "lat", "lng": "lon", "altitude": None},
                            "isVisible": True,
                            "visConfig": {"radius": 30, "opacity": 0.6, "outline": True, "thickness": 1,
                                          "strokeColor": THEME["point_stroke"], "radiusRange": [15, 70], "filled": True}},
                 "visualChannels": {"sizeField": {"name": "size_within", "type": "real"},
                                    "sizeScale": "sqrt"}},
                {"id": "arcs", "type": "arc",
                 "config": {"dataId": "flows", "label": "Flow between kabupaten, both directions (device-months)",
                            "color": THEME["arc"],
                            "columns": {"lat0": "a_lat", "lng0": "a_lon", "lat1": "b_lat", "lng1": "b_lon"},
                            "isVisible": True,
                            "visConfig": {"opacity": 0.8, "thickness": 1, "sizeRange": [1, 10],
                                          "targetColor": THEME["arc_target"]}},
                 "visualChannels": {"sizeField": {"name": "size_total", "type": "real"},
                                    "sizeScale": "linear"}},
            ],
            "interactionConfig": {"tooltip": {"enabled": True, "fieldsToShow": {
                "flows": tooltip(["kabupaten_a", "kabupaten_b", "flow_a_to_b", "flow_b_to_a", "flow_total"]),
                "within": tooltip(["kabupaten", "flow_within"]),
                "kabupaten": tooltip(["name"]),
            }}},
        },
        "mapState": {"latitude": -7.87, "longitude": 110.42, "zoom": 9.3, "pitch": 30, "bearing": 0,
                     "dragRotate": True},
        "mapStyle": {"styleType": MAP_STYLE},
    },
}

kepler_map = KeplerGl(height=700, data={"kabupaten": areas, "within": within, "flows": arcs}, config=config)
kepler_map.save_to_html(file_name=OUTPUT_HTML)
print(arcs[["kabupaten_a", "kabupaten_b", "flow_a_to_b", "flow_b_to_a", "flow_total"]]
      .sort_values("flow_total", ascending=False).to_string(index=False))
print(f"Saved map to {OUTPUT_HTML}")
