# -*- coding: utf-8 -*-
from osgeo import gdal, osr
import numpy as np
from pathlib import Path
import re
import sys
import traceback

gdal.UseExceptions()

# ========= RUTAS (ajusta estas tres) =========
A_PATH  = r'E:\TFM\PNOA_SERIE_ENTERA_JUNTA_050m_newmask_noclosing\PNOA_SERIE_ENTERA_JUNTA_050m_newmask_noclosing\Clump_10\PNOA_ANUAL_2010_OF_ETRS89_HU30_h25_0624-1_clumps.tif'
B_DIR   = r'E:\TFM\PNOA_SERIE_ENTERA_JUNTA_050m_newmask_noclosing\PNOA_SERIE_ENTERA_JUNTA_050m_newmask_noclosing\clumps'
OUT_DIR = r'E:\TFM\PNOA_SERIE_ENTERA_JUNTA_050m_newmask_noclosing\Solape_50_porciento'


# ==== PARÁMETRO: porcentaje mínimo de solape por ID de A con su mejor pareja en B (0–1) ====
MIN_PERCENT = 0.50  # 0.50 = 50%

# ========= UTILIDADES =========
def info_ds(ds, name):
    gt = ds.GetGeoTransform()
    resx, resy = gt[1], abs(gt[5])
    originx, originy = gt[0], gt[3]
    print(f"{name}: size=({ds.RasterXSize}, {ds.RasterYSize})  res=({resx}, {resy})  origin=({originx}, {originy})")

def align_B_to_A(dsA, dsB):
    """
    Devuelve un dataset VRT de B alineado a la malla/extensión/CRS de A (vecino más cercano).
    Usa -tap (targetAlignedPixels=True), -tr y -te.
    """
    gtA = dsA.GetGeoTransform()
    minx = gtA[0]
    maxy = gtA[3]
    maxx = minx + dsA.RasterXSize * gtA[1]
    miny = maxy + dsA.RasterYSize * gtA[5]  # gtA[5] suele ser negativo

    xres = gtA[1]
    yres = abs(gtA[5])  # positivo

    bandB = dsB.GetRasterBand(1)
    nodataB = bandB.GetNoDataValue()

    warp_options = {
        'format': 'VRT',
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

def guess_year(text: str):
    """Devuelve el año de 4 dígitos (1900–2099) si existe en el texto."""
    m = re.findall(r'(?<!\d)((?:19|20)\d{2})(?!\d)', text)
    return m[0] if m else None

def build_out_name(a_name: str, b_name: str, percent_int: int) -> str:
    """
    Nombre base del archivo de salida: incluye años (si los hay) y el stem de B
    para evitar colisiones entre archivos con el mismo año.
    """
    ay = guess_year(a_name)
    by = guess_year(b_name)
    bstem = Path(b_name).stem
    if ay and by:
        return f"{ay}_{by}_{bstem}_{percent_int}_por_ciento.tif"
    else:
        return f"{Path(a_name).stem}__{bstem}__{percent_int}pc.tif"

def ensure_unique_path(p: Path) -> Path:
    """Si el archivo existe, añade _1, _2, ... hasta encontrar uno libre."""
    i = 1
    candidate = p
    while candidate.exists():
        candidate = p.with_name(f"{p.stem}_{i}{p.suffix}")
        i += 1
    return candidate

# ========= NÚCLEO DE PROCESO =========
def process_pair(dsA, A_arr, validA, nodataA, B_path: Path, OUT_path: Path, percent: float):
    dsB = gdal.Open(str(B_path), gdal.GA_ReadOnly)
    if dsB is None:
        raise RuntimeError(f"No se pudo abrir B: {B_path}")

    # Comprobaciones y alineación
    srA = osr.SpatialReference(wkt=dsA.GetProjection())
    srB = osr.SpatialReference(wkt=dsB.GetProjection())
    same_crs = (srA.IsSame(srB) == 1)
    gtA = dsA.GetGeoTransform()
    gtB = dsB.GetGeoTransform()
    geotrans_same = np.allclose(gtA, gtB, atol=1e-9)
    size_same = (dsA.RasterXSize == dsB.RasterXSize) and (dsA.RasterYSize == dsB.RasterYSize)

    if not (same_crs and geotrans_same and size_same):
        dsB = align_B_to_A(dsA, dsB)

    # Lectura banda B
    bandB = dsB.GetRasterBand(1)
    nodataB = bandB.GetNoDataValue()
    B = bandB.ReadAsArray()

    # Máscaras de validez
    validB = (B != 0) if nodataB is None else (B != nodataB)

    # IDs de A que solapan con B (umbral por porcentaje)
    overlap_mask = validA & validB

    # Totales por ID de A (considerando A válida)
    idsA_tot, cntA_tot = np.unique(A_arr[validA], return_counts=True)
    if nodataA is None:
        keepA = (idsA_tot != 0)
    else:
        keepA = (idsA_tot != 0) & (idsA_tot != nodataA)
    idsA_tot = idsA_tot[keepA]
    cntA_tot = cntA_tot[keepA]

    # Pares (A_id, B_id) donde ambos son válidos
    a_pairs = A_arr[overlap_mask]
    b_pairs = B[overlap_mask]

    if nodataA is None:
        valid_pairs_A = (a_pairs != 0)
    else:
        valid_pairs_A = (a_pairs != 0) & (a_pairs != nodataA)
    if nodataB is None:
        valid_pairs_B = (b_pairs != 0)
    else:
        valid_pairs_B = (b_pairs != 0) & (b_pairs != nodataB)

    vp = valid_pairs_A & valid_pairs_B
    a_pairs = a_pairs[vp]
    b_pairs = b_pairs[vp]

    if a_pairs.size == 0:
        touching_ids = np.array([], dtype=idsA_tot.dtype)
    else:
        # Contar píxeles por par (A_id, B_id)
        pairs = np.stack([a_pairs, b_pairs], axis=1)
        uniq_pairs, pair_counts = np.unique(pairs, axis=0, return_counts=True)
        a_ids = uniq_pairs[:, 0]

        # Mejor solape por A_id
        best_overlap = {}
        for aid, c in zip(a_ids, pair_counts):
            if aid in best_overlap:
                if c > best_overlap[aid]:
                    best_overlap[aid] = c
            else:
                best_overlap[aid] = c

        totA = dict(zip(idsA_tot.tolist(), cntA_tot.tolist()))

        touching_list = []
        for aid, ov in best_overlap.items():
            if aid in totA and (ov / float(totA[aid]) >= percent):
                touching_list.append(aid)

        touching_ids = np.array(touching_list, dtype=idsA_tot.dtype)

    print(f" - {Path(B_path).name}: IDs de A que cumplen ≥ {int(percent*100)}% = {touching_ids.size}")

    # Construir salida (banda 1 = máscara binaria; banda 2 = copia de A)
    out_nodata = 255
    out_mask = np.full(A_arr.shape, out_nodata, dtype=np.uint8)
    out_mask[validA] = 1
    if touching_ids.size > 0:
        out_mask[np.isin(A_arr, touching_ids) & validA] = 0

    # Escribir (mismo dtype que A para ambas bandas, como en tu diseño original)
    drv = gdal.GetDriverByName('GTiff')
    bandA = dsA.GetRasterBand(1)
    out_dtype = bandA.DataType

    dst = drv.Create(
        str(OUT_path),
        dsA.RasterXSize,
        dsA.RasterYSize,
        2,
        out_dtype,
        options=[
            'COMPRESS=LZW',
            'TILED=YES',
            'BIGTIFF=IF_SAFER'
        ]
    )
    dst.SetGeoTransform(dsA.GetGeoTransform())
    dst.SetProjection(dsA.GetProjection())

    b1 = dst.GetRasterBand(1)
    b1.WriteArray(out_mask)
    if gdal.GetDataTypeName(out_dtype).startswith('Float'):
        b1.SetNoDataValue(float(out_nodata))
    else:
        b1.SetNoDataValue(int(out_nodata))

    b2 = dst.GetRasterBand(2)
    b2.WriteArray(A_arr)
    if nodataA is not None:
        b2.SetNoDataValue(nodataA)

    b1.FlushCache(); b2.FlushCache()
    dst = None  # cierra

def main():
    A_path = Path(A_PATH).resolve()
    B_dir = Path(B_DIR).resolve()
    OUT_dir = Path(OUT_DIR).resolve()
    OUT_dir.mkdir(parents=True, exist_ok=True)

    # Abrimos A una sola vez y precargamos su array/máscara
    dsA = gdal.Open(str(A_path), gdal.GA_ReadOnly)
    if dsA is None:
        raise RuntimeError("No se pudo abrir el raster A. Revisa A_PATH.")
    print("== Diagnóstico A ==")
    info_ds(dsA, "A")

    bandA = dsA.GetRasterBand(1)
    nodataA = bandA.GetNoDataValue()
    A_arr = bandA.ReadAsArray()
    validA = (A_arr != 0) if nodataA is None else (A_arr != nodataA)

    # Listado de B: recursivo, extensiones comunes
    exts = ('*.tif', '*.tiff', '*.vrt')
    b_list = []
    for e in exts:
        b_list.extend(B_dir.rglob(e))
    # Únicos + ordenados
    b_list = sorted(set(p.resolve() for p in b_list))
    # Evitar procesar A como B si está en el mismo directorio
    b_list = [p for p in b_list if p != A_path]

    if not b_list:
        print(f"No se encontraron rasters en {B_dir}")
        return

    percent_int = int(MIN_PERCENT * 100)
    a_name = A_path.name
    print(f"Archivos B encontrados: {len(b_list)}")
    for i, bpath in enumerate(b_list, 1):
        try:
            out_name = build_out_name(a_name, bpath.name, percent_int)
            out_path = ensure_unique_path(OUT_dir / out_name)  # <- garantiza nombre único
            print(f"[{i}/{len(b_list)}] Procesando: {bpath.name} -> {out_path.name}")
            process_pair(dsA, A_arr, validA, nodataA, bpath, out_path, MIN_PERCENT)
            print(f"   OK -> {out_path}")
        except Exception as e:
            print(f"   ERROR con {bpath}: {e}")
            traceback.print_exc(file=sys.stdout)

    print("== Terminado ==")

# Ejecutar tanto en QGIS como en terminal
if __name__ in ("__main__", "__console__"):
    main()
