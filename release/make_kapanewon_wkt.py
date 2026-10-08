import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import geopandas as gpd

BOUNDARY_SHP = (
    "/Users/widyawan/Library/CloudStorage/GoogleDrive-widyawan@ugm.ac.id/"
    "My Drive/mypaper/2026 Scientific Data/data/batas wilayah DIY/diy_kalurahan_2023.shp"
)
OUTPUT_CSV = "kapanewon_wkt.csv"

# Dissolve 438 kalurahan (Geoportal DIY 2023) into 78 kapanewon polygons, WGS84, as WKT for Spark/Sedona
gdf = gpd.read_file(BOUNDARY_SHP).to_crs("EPSG:4326")
kap = gdf.dissolve(by=["KAB_CODE", "KAB_KOTA", "KEC_CODE", "KEC_KAPN"], as_index=False)[
    ["KAB_CODE", "KAB_KOTA", "KEC_CODE", "KEC_KAPN", "geometry"]]
kap["wkt"] = kap.geometry.to_wkt()

kap.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False)
print(f"Tersimpan: {OUTPUT_CSV} ({len(kap)} kapanewon, {os.path.getsize(OUTPUT_CSV) / 1e6:.1f} MB)")
