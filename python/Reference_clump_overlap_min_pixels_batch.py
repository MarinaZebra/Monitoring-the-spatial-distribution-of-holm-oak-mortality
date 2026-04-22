# -*- coding: utf-8 -*-
import os
import sys
from osgeo import gdal, osr
import numpy as np
from glob import glob

gdal.UseExceptions()

# ==== RUTAS (EJEMPLO) ====
A_PATH = r'E:\TFM\PNOA_SERIE_ENTERA_JUNTA_050m_newmask_noclosing\PNOA_SERIE_ENTERA_JUNTA_050m_newmask_noclosing\Clump_10\PNOA_ANUAL_2010_OF_ETRS89_HU30_h25_0624-1_clumps.tif'
B_DIR  = r'E:\TFM\PNOA_SERIE_ENTERA_JUNTA_050m_newmask_noclosing\PNOA_SERIE_ENTERA_JUNTA_050m_newmask_noclosing\clumps'  # carpeta con B
OUT_DIR = r'E:\TFM\PNOA_SERIE_ENTERA_JUNTA_050m_newmask_noclosing\Solape_20_pixeles'                    # carpeta salida

# ==== PARÁMETROS ====
MIN_PIXELS = 20
# Si quieres filtrar extensiones:
B_EXTS = ('.tif', '.tiff')

def info_ds(ds, name):
    gt = ds.GetGeoTransform()
    resx, resy = gt[1], abs(gt[5])
    originx, originy = gt[0], gt[3]
    print(f"{name}: size=({ds.RasterXSize}, {ds.RasterYSize})  res=({resx}, {resy})  origin=({originx}, {originy})")

def align_B_to_A(dsA, dsB):
    """
    Devuelve un dataset VRT de B alineado a la malla/extensión/CRS de A (vecino más cercano).
    Usa -tap (targetAlignedPixels=True) y -tr (xRes/yRes), además de -te (outputBounds).
    """
    gtA = dsA.GetGeoTransform()
    minx = gtA[0]
    maxy = gtA[3]
    maxx = minx + dsA.RasterXSize * gtA[1]
    miny = maxy + dsA.RasterYSize * gtA[5]  # gtA[5] suele ser negativo

    xres = gtA[1]
    yres = abs(gtA[5])  # positivo

    # Intentamos conservar NoData si existe en B
    bandB = dsB.GetRasterBand(1)
    nodataB = bandB.GetNoDataValue()

    warp_options = {
        'format': 'VRT',  # salida en memoria
        'dstSRS': dsA.GetProjection(),
        'outputBounds': (minx, miny, maxx, maxy),
        'xRes': xres, 'yRes': yres,
        'resampleAlg': gdal.GRA_NearestNeighbour,
        'targetAlignedPixels': True
    }

    if nodataB is not None:
        warp_options['srcNodata'] = nodataB
        warp_options['dstNodata'] = nodataB

    alignedB = gdal.Warp('', dsB, **warp_options)
    return alignedB

def prepare_A(A_path):
    dsA = gdal.Open(A_path, gdal.GA_ReadOnly)
    if dsA is None:
        raise RuntimeError(f"No se pudo abrir A: {A_path}")

    print("== Diagnóstico A ==")
    info_ds(dsA, "A")

    bandA   = dsA.GetRasterBand(1)
    nodataA = bandA.GetNoDataValue()
    A       = bandA.ReadAsArray()

    # Máscara válida de A
    validA = (A != 0) if nodataA is None else (A != nodataA)

    # Guardamos objetos/props para reusar en cada B
    metaA = {
        'ds': dsA,
        'A': A,
        'validA': validA,
        'nodataA': nodataA,
        'gt': dsA.GetGeoTransform(),
        'proj': dsA.GetProjection(),
        'xsize': dsA.RasterXSize,
        'ysize': dsA.RasterYSize,
        'dtype': bandA.DataType
    }
    return metaA

def process_pair(metaA, B_path, out_dir, min_pixels=10):
    """
    Procesa una imagen B contra A y escribe un GeoTIFF con:
      - BANDA 1: máscara (1 = A válido; 0 = IDs de A con solape >= min_pix; nodata=255)
      - BANDA 2: copia de A
    El nombre de salida se deriva del nombre base de B.
    """
    A = metaA['A']
    validA = metaA['validA']
    nodataA = metaA['nodataA']
    dsA = metaA['ds']

    # Abrir B
    dsB_raw = gdal.Open(B_path, gdal.GA_ReadOnly)
    if dsB_raw is None:
        raise RuntimeError(f"No se pudo abrir B: {B_path}")

    # Si CRS/GeoTransform/Tamaño no coinciden, alineamos B a A
    srA = osr.SpatialReference(wkt=metaA['proj'])
    srB = osr.SpatialReference(wkt=dsB_raw.GetProjection() or '')
    same_crs = (srA.IsSame(srB) == 1)

    gtA = metaA['gt']
    gtB = dsB_raw.GetGeoTransform()
    geotrans_same = (gtB is not None and np.allclose(gtA, gtB, atol=1e-9))
    size_same = (metaA['xsize'] == dsB_raw.RasterXSize) and (metaA['ysize'] == dsB_raw.RasterYSize)

    if not (same_crs and geotrans_same and size_same):
        dsB = align_B_to_A(dsA, dsB_raw)
        # Libera el original
        dsB_raw = None
    else:
        dsB = dsB_raw
        dsB_raw = None

    # Diagnóstico mínimo tras alineación (opcional)
    # info_ds(dsB, "B (alineado)")

    # Leer B y máscara válida de B
    bandB = dsB.GetRasterBand(1)
    nodataB = bandB.GetNoDataValue()
    B = bandB.ReadAsArray()
    validB = (B != 0) if nodataB is None else (B != nodataB)

    # Solape (misma celda) y conteo por ID de A
    overlap_mask = validA & validB

    ids_all, counts_all = np.unique(A[overlap_mask], return_counts=True)

    # Filtramos 0 / nodata del listado de IDs de A
    if nodataA is None:
        valid_idx = (ids_all != 0)
    else:
        valid_idx = (ids_all != 0) & (ids_all != nodataA)

    ids_all    = ids_all[valid_idx]
    counts_all = counts_all[valid_idx]

    touching_ids = ids_all[counts_all >= min_pixels]

    # Construir máscara de salida (BANDA 1)
    out_nodata = 255
    out_mask = np.full(A.shape, out_nodata, dtype=np.uint8)
    out_mask[validA] = 1
    if touching_ids.size > 0:
        out_mask[np.isin(A, touching_ids) & validA] = 0

    # Preparar salida
    baseB = os.path.splitext(os.path.basename(B_path))[0]
    out_path = os.path.join(out_dir, f"{baseB}_vsA_min{min_pixels}px.tif")

    drv = gdal.GetDriverByName('GTiff')
    out_dtype = metaA['dtype']  # conservamos tipo de A para ambas bandas (simple y compatible)

    dst = drv.Create(
        out_path,
        metaA['xsize'],
        metaA['ysize'],
        2,
        out_dtype,
        options=[
            'COMPRESS=LZW',
            'TILED=YES',
            'BIGTIFF=IF_SAFER'
        ]
    )

    dst.SetGeoTransform(gtA)
    dst.SetProjection(metaA['proj'])

    # Banda 1: máscara (convertimos al tipo de salida si hace falta)
    b1 = dst.GetRasterBand(1)
    if gdal.GetDataTypeName(out_dtype).startswith('Float'):
        b1.WriteArray(out_mask.astype(np.float32))
        b1.SetNoDataValue(float(out_nodata))
    else:
        b1.WriteArray(out_mask.astype(np.uint16 if gdal.GetDataTypeName(out_dtype) in ('UInt16', 'Int16') else np.uint8))
        b1.SetNoDataValue(int(out_nodata))

    # Banda 2: copia de A
    b2 = dst.GetRasterBand(2)
    b2.WriteArray(A)
    if nodataA is not None:
        b2.SetNoDataValue(nodataA)

    b1.FlushCache(); b2.FlushCache()
    dst = None
    dsB = None

    print(f"OK -> {out_path} (IDs solapados ≥ {min_pixels} píxeles: {touching_ids.size})")
    return out_path, touching_ids.size

def main():
    # Preparar salida
    os.makedirs(OUT_DIR, exist_ok=True)

    # Cargar A una sola vez
    metaA = prepare_A(A_PATH)

    # Listar Bs
    candidates = []
    for ext in B_EXTS:
        candidates.extend(glob(os.path.join(B_DIR, f"*{ext}")))
    # Evitar que procese la propia A si está en la misma carpeta
    candidates = [p for p in candidates if os.path.abspath(p) != os.path.abspath(A_PATH)]

    if not candidates:
        print(f"No se encontraron imágenes en {B_DIR} con extensiones {B_EXTS}.")
        return

    total = len(candidates)
    print(f"== Encontrados {total} archivos en B. Procesando... ==")

    ok, fail = 0, 0
    for i, bpath in enumerate(sorted(candidates), 1):
        try:
            print(f"[{i}/{total}] Procesando: {bpath}")
            process_pair(metaA, bpath, OUT_DIR, min_pixels=MIN_PIXELS)
            ok += 1
        except Exception as e:
            # Registrar el error y continuar con el siguiente
            fail += 1
            print(f"ERROR con {bpath}: {e}", file=sys.stderr)

    print(f"== Terminado. Éxitos: {ok} | Fallos: {fail} ==")

# Funciona tanto en QGIS como en terminal
if __name__ in ("__main__", "__console__"):
    main()
