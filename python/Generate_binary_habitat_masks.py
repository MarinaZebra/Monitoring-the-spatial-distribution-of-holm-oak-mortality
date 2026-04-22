from qgis.core import (
    QgsRasterLayer,
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsField,
    QgsPointXY
)
from qgis import processing
from PyQt5.QtCore import QVariant
import os
import glob

# --- 0. Parámetros de entrada ---
raster_folder = r'D:\TFM\PNOA\PENOA_2022'  # carpeta con los .tif
overlay_shp   = r'D:\TFM\Comparacion_areas_clip\02.Habitat_6310\Habitat_6310_Caceres.shp'
output_folder = r'D:\TFM\Mascara_boleana\Mascara'
os.makedirs(output_folder, exist_ok=True)

# --- 1. Preparar capa overlay y reproyectar si hace falta ---
if not os.path.exists(overlay_shp):
    raise IOError(f"No se encontró shapefile overlay: {overlay_shp}")

layer_ov = QgsVectorLayer(overlay_shp, 'overlay_disk', 'ogr')
if not layer_ov.isValid():
    raise IOError(f"Capa overlay inválida: {overlay_shp}")

# Reproyectamos overlay al CRS del primer raster que procesemos
# para reutilizarlo en todos los demás
sample_raster = glob.glob(os.path.join(raster_folder, '*.tif'))[0]
sample_layer = QgsRasterLayer(sample_raster, 'sample')
if sample_layer.crs() != layer_ov.crs():
    reproj = processing.run(
        'native:reprojectlayer',
        {'INPUT': layer_ov, 'TARGET_CRS': sample_layer.crs(), 'OUTPUT': 'memory:'}
    )
    layer_ov = reproj['OUTPUT']

# --- 2. Recorrer todos los rasters en la carpeta ---
for raster_path in glob.glob(os.path.join(raster_folder, '*.tif')):
    print(f"\nProcesando: {raster_path}")

    # 2.1 Cargar raster
    raster_layer = QgsRasterLayer(raster_path, 'raster_layer')
    if not raster_layer.isValid():
        print(f"  ERROR: raster inválido: {raster_path}")
        continue

    crs    = raster_layer.crs()
    extent = raster_layer.extent()

    # 2.2 Crear polígono de la extensión en memoria
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
    feat_ext = QgsFeature()
    feat_ext.setGeometry(geom)
    feat_ext.setAttributes([1])
    pr_ext.addFeature(feat_ext)

    # 2.3 Intersección (extent ∩ overlay)
    inter = processing.run(
        'native:intersection',
        {
            'INPUT': layer_ext,
            'OVERLAY': layer_ov,
            'INPUT_FIELDS': [],
            'OVERLAY_FIELDS': [],
            'OUTPUT': 'memory:'
        }
    )['OUTPUT']

    # 2.4 Rasterizar la intersección
    basename = os.path.splitext(os.path.basename(raster_path))[0]
    out_raster = os.path.join(output_folder, f"{basename}_mask.tif")
    if os.path.exists(out_raster):
        os.remove(out_raster)

    xmin, xmax = extent.xMinimum(), extent.xMaximum()
    ymin, ymax = extent.yMinimum(), extent.yMaximum()
    extent_str = f"{xmin},{xmax},{ymin},{ymax}"

    params = {
        'INPUT': inter,
        'FIELD': '',
        'BURN': 1,
        'USE_Z': False,
        'UNITS': 1,
        'WIDTH': 0.25,
        'HEIGHT': 0.25,
        'EXTENT': extent_str,
        'NODATA': 0,
        'DATA_TYPE': 1,
        'OUTPUT': out_raster
    }
    processing.run('gdal:rasterize', params)
    print(f"  → Máscara guardada en: {out_raster}")

print("\n¡Proceso completado para todos los rasters!")
