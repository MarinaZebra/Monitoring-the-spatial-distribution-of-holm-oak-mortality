# -*- coding: utf-8 -*-
import os, math, glob, tempfile, numpy as np
from qgis.core import (
    QgsProject, QgsRasterLayer, QgsCoordinateReferenceSystem,
    QgsCoordinateTransform, QgsPointXY, QgsProcessingFeedback,
    QgsVectorLayer, QgsFeature, QgsGeometry, QgsField
)
from qgis import processing
from PyQt5.QtCore import QVariant
from osgeo import gdal
gdal.UseExceptions()

# ============== AJUSTA ESTAS RUTAS ==============
INPUT_DIR   = r"E:/TFM/Cadena_de_codigo/Varias_imagenes"  # Carpeta con ortofotos *.tif
OVERLAY_SHP = r"E:/TFM/Mascara/Corine_land_cover_2018_2012/CLC_2012_2018.shp"
OUT_DIR     = r"E:/TFM/Cadena_de_codigo/Varias_imagenes/Resultados"    # Carpeta de resultados
TARGET_RES  = 0.5                     # m/px para el resample
OVERWRITE   = True                    # Cambia a False para saltar outputs existentes
# ================================================

# ================= AJUSTES SEGMENTACIÓN =================
OTSU_OFFSET         = 0.0
SIEVE_MIN_AREA_M2   = 9.0
MAX_HOLES_M2        = 2.0
CLOSING_ENABLE      = True
CLOSING_DILATE_PX   = 1.5
DIAG_SAMPLE_PX      = 1024
SAVE_DEBUG          = False

# ================= UTILIDADES =================
def _ensure_raster(path):
    rl = QgsRasterLayer(path, os.path.basename(path))
    if not rl.isValid():
        raise RuntimeError(f"No se pudo cargar ráster: {path}")
    return rl

def _gsd_meters_from_raster(rl, fallback=0.25):
    try:
        px = abs(rl.rasterUnitsPerPixelX()); py = abs(rl.rasterUnitsPerPixelY())
        if px == 0 or py == 0: return float(fallback)
        crs = rl.crs()
        if crs.isGeographic():
            ext = rl.extent(); c = ext.center()
            to84 = QgsCoordinateTransform(crs, QgsCoordinateReferenceSystem("EPSG:4326"),
                                          QgsProject.instance().transformContext())
            ll = to84.transform(QgsPointXY(c.x(), c.y()))
            lat = float(ll.y())
            m_per_deg_lat = 111319.49
            m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(lat))
            return float(max(px*m_per_deg_lon, py*m_per_deg_lat))
        else:
            return float(max(px, py))
    except Exception:
        return float(fallback)

def _read_arr(path):
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"GDAL no pudo abrir: {path}")
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr, ds

def _write_arr(path, arr, ref_ds, gdal_type, nodata=None, compress=True):
    drv = gdal.GetDriverByName('GTiff')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    opts = ['COMPRESS=LZW'] if compress else []
    ds_out = drv.Create(path, ref_ds.RasterXSize, ref_ds.RasterYSize, 1, gdal_type, options=opts)
    ds_out.SetGeoTransform(ref_ds.GetGeoTransform())
    ds_out.SetProjection(ref_ds.GetProjection())
    rb = ds_out.GetRasterBand(1)
    if nodata is not None:
        rb.SetNoDataValue(nodata)
    rb.WriteArray(arr)
    rb.FlushCache(); ds_out.FlushCache()
    del rb; del ds_out

def _otsu_threshold(values, bins=256):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmin == vmax:
        return vmin
    hist, edges = np.histogram(arr, bins=bins, range=(vmin, vmax))
    p = hist.astype(float) / max(hist.sum(), 1.0)
    mids = (edges[:-1] + edges[1:]) / 2.0
    omega = np.cumsum(p); mu = np.cumsum(p * mids); mu_t = mu[-1]
    denom = omega * (1.0 - omega); denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    idx = np.nanargmax(sigma_b2)
    return float(mids[idx])

def _closing_binary(bin_arr, radius_px):
    r = int(round(float(radius_px)))
    if r <= 0:
        return bin_arr.copy()
    k = 2 * r + 1
    def _box_sum(a, r):
        p = np.pad(a, pad_width=r, mode='constant', constant_values=0)
        s = p.cumsum(axis=0).cumsum(axis=1)
        return s[k:, k:] - s[:-k, k:] - s[k:, :-k] + s[:-k, :-k]
    dil = (_box_sum(bin_arr.astype(np.uint8), r) > 0).astype(np.uint8)
    ero = (_box_sum(dil, r) == (k * k)).astype(np.uint8)
    return ero

# ================= ETAPA 3: SEGMENTACIÓN COPAS =================
def segment_crowns(in_path, mask_path, out_path):
    print("\n=== [3] Delimitación de copas (RGB+Máscara) ===")
    fb = QgsProcessingFeedback()
    tmpdir = tempfile.mkdtemp(prefix="obia_mask_")
    print("Temp:", tmpdir)

    def _chk(path):
        if (not os.path.exists(path)) or (not _ensure_raster(path).isValid()):
            raise RuntimeError(f"Intermedio faltante o inválido: {path}")
        return path

    in_rl = _ensure_raster(in_path)
    mask_rl_src = _ensure_raster(mask_path)
    px = abs(in_rl.rasterUnitsPerPixelX()); py = abs(in_rl.rasterUnitsPerPixelY())
    ext = in_rl.extent()
    xmin, xmax, ymin, ymax = ext.xMinimum(), ext.xMaximum(), ext.yMinimum(), ext.yMaximum()
    tgt_crs_auth = in_rl.crs().authid()
    src_crs_auth = mask_rl_src.crs().authid() if mask_rl_src.crs().isValid() else None

    print("→ Alineando máscara al grid de la ortofoto …")
    mask_aligned = os.path.join(tmpdir, "mask_aligned.tif")
    processing.run("gdal:warpreproject", {
        'INPUT': mask_path,
        'SOURCE_CRS': src_crs_auth,
        'TARGET_CRS': tgt_crs_auth,
        'RESAMPLING': 0, 'NODATA': 0, 'TARGET_RESOLUTION': None, 'OPTIONS': '',
        'DATA_TYPE': 1,
        'TARGET_EXTENT': f"{xmin},{xmax},{ymin},{ymax}",
        'TARGET_EXTENT_CRS': tgt_crs_auth,
        'MULTITHREADING': True,
        'EXTRA': f'-tr {px} {py}',
        'OUTPUT': mask_aligned
    }, feedback=fb)
    _chk(mask_aligned)

    if in_rl.bandCount() < 3:
        raise RuntimeError(f"La ortofoto tiene {in_rl.bandCount()} bandas; se necesitan 3 (RGB).")

    # 1) Extraer R/G/B
    r_raw = os.path.join(tmpdir, "R_raw.tif")
    g_raw = os.path.join(tmpdir, "G_raw.tif")
    b_raw = os.path.join(tmpdir, "B_raw.tif")
    processing.run("gdal:translate", { 'INPUT': in_path, 'NODATA': None, 'OPTIONS': '',
        'EXTRA': '-b 1 -co COMPRESS=LZW', 'DATA_TYPE': 0, 'TARGET_RESOLUTION': None, 'OUTPUT': r_raw }, feedback=fb)
    processing.run("gdal:translate", { 'INPUT': in_path, 'NODATA': None, 'OPTIONS': '',
        'EXTRA': '-b 2 -co COMPRESS=LZW', 'DATA_TYPE': 0, 'TARGET_RESOLUTION': None, 'OUTPUT': g_raw }, feedback=fb)
    processing.run("gdal:translate", { 'INPUT': in_path, 'NODATA': None, 'OPTIONS': '',
        'EXTRA': '-b 3 -co COMPRESS=LZW', 'DATA_TYPE': 0, 'TARGET_RESOLUTION': None, 'OUTPUT': b_raw }, feedback=fb)
    _chk(r_raw); _chk(g_raw); _chk(b_raw)

    # 1b) Remuestrear R/G/B exacto a la máscara
    m_rl = _ensure_raster(mask_aligned)
    m_px = abs(m_rl.rasterUnitsPerPixelX()); m_py = abs(m_rl.rasterUnitsPerPixelY())
    m_ext = m_rl.extent()
    mxmin, mxmax, mymin, mymax = m_ext.xMinimum(), m_ext.xMaximum(), m_ext.yMinimum(), m_ext.yMaximum()
    r_tif = os.path.join(tmpdir, "R_aligned.tif")
    g_tif = os.path.join(tmpdir, "G_aligned.tif")
    b_tif = os.path.join(tmpdir, "B_aligned.tif")
    for src, dst in [(r_raw, r_tif), (g_raw, g_tif), (b_raw, b_tif)]:
        processing.run("gdal:warpreproject", {
            'INPUT': src, 'SOURCE_CRS': tgt_crs_auth, 'TARGET_CRS': tgt_crs_auth,
            'RESAMPLING': 0, 'NODATA': 0, 'TARGET_RESOLUTION': None, 'OPTIONS': '',
            'DATA_TYPE': 0, 'TARGET_EXTENT': f"{mxmin},{mxmax},{mymin},{mymax}",
            'TARGET_EXTENT_CRS': tgt_crs_auth, 'MULTITHREADING': True,
            'EXTRA': f'-tr {m_px} {m_py}', 'OUTPUT': dst
        }, feedback=fb)
        _chk(dst)

    print("→ ExGR (NumPy) + Otsu")
    G_arr, ref_ds = _read_arr(g_tif)
    R_arr, _ = _read_arr(r_tif)
    B_arr, _ = _read_arr(b_tif)
    M_arr, _ = _read_arr(mask_aligned)

    inside = M_arr > 0
    exgr_arr = np.zeros_like(G_arr, dtype=np.float32)
    exgr_arr[inside] = (3.0*G_arr[inside] - 2.4*R_arr[inside] - 1.0*B_arr[inside]).astype(np.float32)

    vals = exgr_arr[inside].astype(np.float64)
    if vals.size == 0:
        raise RuntimeError("Otsu: no hay píxeles dentro de máscara.")
    max_samples = DIAG_SAMPLE_PX*DIAG_SAMPLE_PX
    vals_s = vals if vals.size <= max_samples else vals[np.random.choice(vals.size, size=max_samples, replace=False)]

    T = _otsu_threshold(vals_s); T = float(np.median(vals_s)) if T is None else T
    T_adj = T + float(OTSU_OFFSET)
    print(f"   Umbral Otsu: {T:.4f} | Ajustado: {T_adj:.4f}")

    bin_arr = np.zeros(M_arr.shape, dtype=np.uint8)
    bin_arr[inside] = (exgr_arr[inside] >= T_adj).astype(np.uint8)

    # Sieve
    gsd = _gsd_meters_from_raster(in_rl, fallback=0.25)
    min_pix_sieve = max(1, int(round(SIEVE_MIN_AREA_M2 / (gsd * gsd))))
    print(f"→ Sieve (min {SIEVE_MIN_AREA_M2} m² ≈ {min_pix_sieve} px)")
    tmp_bin0 = os.path.join(tmpdir, "bin0.tif")
    _write_arr(tmp_bin0, bin_arr, ref_ds, gdal.GDT_Byte, nodata=None, compress=True)
    sieve1_path = os.path.join(tmpdir, "sieve1.tif")
    processing.run("gdal:sieve", {
        'INPUT': tmp_bin0, 'THRESHOLD': min_pix_sieve,
        'EIGHT_CONNECTED': True, 'NO_MASK': True, 'MASK_LAYER': None,
        'OUTPUT': sieve1_path
    }, feedback=fb)
    bin_arr, ref_ds2 = _read_arr(sieve1_path)
    bin_arr = (bin_arr > 0).astype(np.uint8)

    # Cierre morfológico
    if CLOSING_ENABLE and CLOSING_DILATE_PX > 0:
        print(f"→ Cierre morfológico (radio {CLOSING_DILATE_PX} px)")
        bin_arr = _closing_binary(bin_arr, CLOSING_DILATE_PX)

    # Relleno de agujeros
    if MAX_HOLES_M2 and MAX_HOLES_M2 > 0:
        min_pix_holes = max(1, int(round(MAX_HOLES_M2 / (gsd * gsd))))
        print(f"→ Rellenando agujeros < {MAX_HOLES_M2} m² (≈ {min_pix_holes} px)")
        inv_in = os.path.join(tmpdir, "inv_in.tif")
        _write_arr(inv_in, (1 - bin_arr).astype(np.uint8), ref_ds2, gdal.GDT_Byte, nodata=None, compress=True)
        inv_sieved = os.path.join(tmpdir, "inv_sieved.tif")
        processing.run("gdal:sieve", {
            'INPUT': inv_in, 'THRESHOLD': min_pix_holes,
            'EIGHT_CONNECTED': True, 'NO_MASK': True, 'MASK_LAYER': None,
            'OUTPUT': inv_sieved
        }, feedback=fb)
        inv_arr, _ = _read_arr(inv_sieved)
        bin_arr = (1 - (inv_arr > 0).astype(np.uint8)).astype(np.uint8)

    # Guardado final
    if SAVE_DEBUG:
        dbg_bin = os.path.join(os.path.dirname(out_path), f"{os.path.splitext(os.path.basename(out_path))[0]}_debug.tif")
        _write_arr(dbg_bin, bin_arr, ref_ds2, gdal.GDT_Byte, nodata=0, compress=True)
        print(f"✓ DEBUG guardado: {dbg_bin}")

    print(f"→ Guardando raster final (1 = encina, NODATA = 0): {out_path}")
    _write_arr(out_path, bin_arr, ref_ds2, gdal.GDT_Byte, nodata=0, compress=True)
    print("✓ Terminado:", out_path)

# ================= ETAPA 1+2: RESAMPLE + MÁSCARA =================
def run_resample_and_mask(input_raster, overlay_shp, resample_path, mask_path, target_res=0.5):
    print("=== [1] Warp/Resample ===")
    fb = QgsProcessingFeedback()

    if (not os.path.exists(resample_path)) or OVERWRITE:
        processing.run("gdal:warpreproject", {
            'INPUT': input_raster, 'TARGET_CRS': None, 'RESAMPLING': 0,
            'NODATA': None, 'TARGET_RESOLUTION': target_res, 'OPTIONS': '',
            'DATA_TYPE': 0, 'OUTPUT': resample_path
        }, feedback=fb)
        print(f"→ Resample guardado: {resample_path}")
    else:
        print(f"→ Resample ya existe, se omite: {resample_path}")

    print("\n=== [2] Generar máscara booleana alineada ===")
    raster_layer = _ensure_raster(resample_path)
    crs = raster_layer.crs()
    extent = raster_layer.extent()
    px_x = abs(raster_layer.rasterUnitsPerPixelX())
    px_y = abs(raster_layer.rasterUnitsPerPixelY())

    # Overlay
    if not os.path.exists(overlay_shp):
        raise IOError(f"No se encontró shapefile overlay: {overlay_shp}")
    layer_ov = QgsVectorLayer(overlay_shp, 'overlay_disk', 'ogr')
    if not layer_ov.isValid():
        raise IOError(f"Capa overlay inválida: {overlay_shp}")
    if layer_ov.crs() != crs:
        print("→ Reproyectando overlay al CRS del raster...")
        layer_ov = processing.run('native:reprojectlayer',
                                  {'INPUT': layer_ov, 'TARGET_CRS': crs, 'OUTPUT': 'memory:'})['OUTPUT']

    # Polígono de extensión
    pts = [
        QgsPointXY(extent.xMinimum(), extent.yMinimum()),
        QgsPointXY(extent.xMaximum(), extent.yMinimum()),
        QgsPointXY(extent.xMaximum(), extent.yMaximum()),
        QgsPointXY(extent.xMinimum(), extent.yMaximum()),
        QgsPointXY(extent.xMinimum(), extent.yMinimum())
    ]
    geom = QgsGeometry.fromPolygonXY([pts])
    uri_ext = f"Polygon?crs={crs.authid()}"
    layer_ext = QgsVectorLayer(uri_ext, 'extent_mem', 'memory')
    pr_ext = layer_ext.dataProvider()
    pr_ext.addAttributes([QgsField('ID', QVariant.Int)])
    layer_ext.updateFields()
    feat_ext = QgsFeature(); feat_ext.setGeometry(geom); feat_ext.setAttributes([1]); pr_ext.addFeature(feat_ext)

    # Intersección
    inter = processing.run('native:intersection', {
        'INPUT': layer_ext, 'OVERLAY': layer_ov,
        'INPUT_FIELDS': [], 'OVERLAY_FIELDS': [], 'OUTPUT': 'memory:'
    })['OUTPUT']

    # Rasterizar
    xmin, xmax = extent.xMinimum(), extent.xMaximum()
    ymin, ymax = extent.yMinimum(), extent.yMaximum()
    extent_str = f"{xmin},{xmax},{ymin},{ymax}"

    if (not os.path.exists(mask_path)) or OVERWRITE:
        processing.run('gdal:rasterize', {
            'INPUT': inter, 'FIELD': '', 'BURN': 1, 'USE_Z': False, 'UNITS': 1,
            'WIDTH': px_x, 'HEIGHT': px_y, 'EXTENT': extent_str,
            'NODATA': 0, 'DATA_TYPE': 1, 'INIT': 0, 'OUTPUT': mask_path
        })
        print(f"→ Máscara guardada: {mask_path}")
    else:
        print(f"→ Máscara ya existe, se omite: {mask_path}")

# ================= ETAPA 4: CLUMPS (GRASS) =================
def label_clumps(in_raster, out_raster, eight_connected=True):
    print("\n=== [4] Etiquetado de componentes (GRASS r.clump) ===")
    fb = QgsProcessingFeedback()
    try:
        res = processing.run(
            "grass7:r.clump",
            {
                'input': in_raster,
                'title': 'Encinas clumps',
                '-d': bool(eight_connected),
                'threshold': 0,
                'output': out_raster,
                'GRASS_REGION_PARAMETER': None,
                'GRASS_REGION_CELLSIZE_PARAMETER': 0
            },
            feedback=fb
        )
        print("→ Clumps OK:", res['output'])
    except Exception as e:
        print(f"(Aviso) r.clump falló ({e}); reintento creando NULL explícitos...")
        tmpdir = tempfile.mkdtemp(prefix="cl_")
        mask_null = os.path.join(tmpdir, "mask_null.tif")
        res_mask = processing.run(
            "grass7:r.mapcalc.simple",
            {
                'a': in_raster,
                'expression': 'if(A==1,1,null())',
                'output': mask_null,
                'GRASS_REGION_PARAMETER': None,
                'GRASS_REGION_CELLSIZE_PARAMETER': 0
            },
            feedback=fb
        )
        mask = res_mask['output']
        res = processing.run(
            "grass7:r.clump",
            {
                'input': mask,
                'title': 'Encinas clumps',
                '-d': bool(eight_connected),
                'threshold': 0,
                'output': out_raster,
                'GRASS_REGION_PARAMETER': None,
                'GRASS_REGION_CELLSIZE_PARAMETER': 0
            },
            feedback=fb
        )
        print("→ Clumps OK (fallback):", res['output'])

# ================= BUCLE POR CARPETA =================
def process_folder(input_dir, overlay_shp, out_dir, target_res=0.5):
    tif_list = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    if not tif_list:
        print(f"No se encontraron TIF en {input_dir}")
        return

    # Subcarpetas ordenadas
    paths = {
        "resample": os.path.join(out_dir, "resample"),
        "mask":     os.path.join(out_dir, "mask"),
        "crowns":   os.path.join(out_dir, "crowns"),
        "clumps":   os.path.join(out_dir, "clumps"),
    }
    for p in paths.values(): os.makedirs(p, exist_ok=True)

    for i, in_raster in enumerate(tif_list, 1):
        base = os.path.splitext(os.path.basename(in_raster))[0]
        print("\n" + "="*80)
        print(f"[{i}/{len(tif_list)}] Procesando: {base}")
        try:
            resample_path = os.path.join(paths["resample"], f"{base}_resample.tif")
            mask_path     = os.path.join(paths["mask"],     f"{base}_mask.tif")
            crowns_path   = os.path.join(paths["crowns"],   f"{base}_encina_mask.tif")
            clumps_path   = os.path.join(paths["clumps"],   f"{base}_clumps.tif")

            run_resample_and_mask(in_raster, overlay_shp, resample_path, mask_path, target_res=target_res)
            if (not os.path.exists(crowns_path)) or OVERWRITE:
                segment_crowns(resample_path, mask_path, crowns_path)
            else:
                print(f"→ Copas ya existen, se omite: {crowns_path}")

            if (not os.path.exists(clumps_path)) or OVERWRITE:
                label_clumps(crowns_path, clumps_path, eight_connected=True)
            else:
                print(f"→ Clumps ya existen, se omite: {clumps_path}")

            print(f"✓ FIN {base}")
        except Exception as e:
            print(f"✗ ERROR en {base}: {e}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    process_folder(INPUT_DIR, OVERLAY_SHP, OUT_DIR, target_res=TARGET_RES)
    print("\n=== FIN DEL PROCESO POR CARPETA ===")

main()
