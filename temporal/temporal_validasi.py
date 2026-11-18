import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # resolve relative paths from this folder

import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "release"))
from paths import fig_path  # noqa: E402
from source_data import get_spark, raw_pings  # noqa: E402
from pyspark.sql import functions as F
from pyspark.sql import Window

spark = get_spark("temporal_validasi")  # Spark Connect or local; session time zone checked (release/source_data.py)

df_filtered = raw_pings(spark)
df_geo = df_filtered.withColumn("geom", F.expr("ST_Point(longitude, latitude)"))

# Cache versi df_filtered yang sudah ditambah kolom turunan, karena dipakai berkali-kali di bawah
# Hanya simpan kolom yang dipakai (maid, date, dow, hour) supaya cache jauh lebih kecil
# Catatan: datetime = jam lokal WIB (timestamp epoch + 7 jam), session timezone server = UTC,
#          jadi to_date/hour di bawah sudah dalam WIB. Jangan ubah spark.sql.session.timeZone.
df_temporal = df_filtered.withColumn("date", F.to_date("datetime")) \
    .withColumn("dow", F.dayofweek("datetime")) \
    .withColumn("hour", F.hour("datetime")) \
    .select("maid", "date", "dow", "hour") \
    .cache()

# Trigger materialization cache sekali di awal (biar count/groupby berikutnya baca dari cache, bukan re-scan+re-parse)
n_rows = df_temporal.count()
print(f"df_temporal cached, jumlah baris: {n_rows:,}")

# ============================================================
# 1. POLA HARIAN — tren lintas 9 bulan (analog Fig. 6 YJMob)
# ============================================================
daily_stats = (
    df_temporal.groupBy("date")
    .agg(
        F.count("*").alias("n_pings"),
        F.countDistinct("maid").alias("n_unique_users")
    )
    .orderBy("date")
    .cache()
)

daily_stats.show(30, truncate=False)
daily_pd = daily_stats.toPandas()  # aman, cuma ~270 baris (9 bulan)

# ============================================================
# 2. POLA HARI-DALAM-MINGGU — weekday vs weekend
#    Spark dayofweek: 1=Minggu, 2=Senin, ..., 7=Sabtu
# ============================================================
dow_stats = (
    df_temporal.withColumn(
        "day_name",
        F.when(F.col("dow") == 1, "Minggu")
         .when(F.col("dow") == 2, "Senin")
         .when(F.col("dow") == 3, "Selasa")
         .when(F.col("dow") == 4, "Rabu")
         .when(F.col("dow") == 5, "Kamis")
         .when(F.col("dow") == 6, "Jumat")
         .otherwise("Sabtu")
    )
    .groupBy("dow", "day_name")
    .agg(
        F.count("*").alias("n_pings"),
        F.countDistinct("maid").alias("n_unique_users")
    )
    .orderBy("dow")
)

dow_stats.show()
dow_pd = dow_stats.toPandas()

# ============================================================
# 3. POLA JAM-DALAM-HARI — agregasi lintas semua hari (analog Fig. 7 YJMob)
# ============================================================
hourly_stats = (
    df_temporal.groupBy("hour")
    .agg(
        F.count("*").alias("n_pings"),
        F.countDistinct("maid").alias("n_unique_users")
    )
    .orderBy("hour")
)

hourly_stats.show(24)
hourly_pd = hourly_stats.toPandas()

# ============================================================
# 4. DETEKSI HARI ANOMALI — bandingkan tiap hari vs rolling average 7 hari
#    (analog "day 27 anomaly" YJMob)
#    Pakai daily_stats yang sudah di-cache di langkah 1, jangan scan df_temporal lagi
# ============================================================
w7 = Window.orderBy("date").rowsBetween(-3, 3)

daily_anomaly = daily_stats.withColumn(
    "rolling_avg_pings", F.avg("n_pings").over(w7)
).withColumn(
    "pct_dev", (F.col("n_pings") - F.col("rolling_avg_pings")) / F.col("rolling_avg_pings") * 100
)

print("Hari dengan penurunan ekstrem (kandidat anomali/gangguan):")
daily_anomaly.orderBy("pct_dev").show(10, truncate=False)

print("Hari dengan lonjakan ekstrem (kandidat event khusus):")
daily_anomaly.orderBy(F.desc("pct_dev")).show(10, truncate=False)

daily_anomaly_pd = daily_anomaly.toPandas()

# ============================================================
# Simpan hasil ke CSV (di folder temporal/)
# ============================================================
outputs = {
    "temporal_daily.csv": daily_pd,
    "temporal_dow.csv": dow_pd,
    "temporal_hourly.csv": hourly_pd,
    "temporal_daily_anomaly.csv": daily_anomaly_pd,
}
for filename, pdf in outputs.items():
    pdf.to_csv(fig_path(filename), index=False)
    print(f"Tersimpan: {filename} ({len(pdf)} baris)")

# ============================================================
# Bersihkan cache setelah selesai dipakai (hemat memori executor)
# ============================================================
df_temporal.unpersist()
daily_stats.unpersist()
