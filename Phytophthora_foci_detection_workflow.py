# -----------------------------------------------------------
# Script completo:
# 1) Calcula espaciamiento típico (NN distance) con todas las encinas.
# 2) Agrupa encinas afectadas (clase = 1) en focos (clustering tipo DBSCAN)
#    usando R = R_MULTIPLIER * mean_NND, con mínimo 5 encinas por foco.
#    - Capa de salida: 'encinas_afectadas_clusters' (puntos)
#    - Capa de salida: 'focos_phytophthora' (polígonos)
# 3) Muestra MDT/DEM sobre encinas_afectadas_clusters y, para cada foco,
#    elige la encina afectada con mayor cota como "punto de inicio".
#    - Capa de salida: 'puntos_inicio_focos' (puntos)
# -----------------------------------------------------------

from qgis.core import (
    QgsProject,
    QgsPointXY,
    QgsGeometry,
    QgsRectangle,
    QgsVectorLayer,
    QgsFields,
    QgsField,
    QgsFeature,
    QgsSpatialIndex,
    QgsWkbTypes,
    QgsRasterLayer
)
from qgis.PyQt.QtCore import QVariant
import processing
import math

# ---------------- PARÁMETROS A EDITAR ----------------------

# Capa de TODAS las encinas (sanas + afectadas)
TREE_LAYER_NAME = 'Encinas_vivas_vs_muertas_2012'
CLASS_FIELD = 'Clase'     # campo que indica sana/afectada
INFECTED_VALUE = 1        # valor que indica encina afectada

# Clustering
R_MULTIPLIER = 1.5        # R = R_MULTIPLIER * distancia media al vecino más cercano
MIN_TREES_PER_FOCUS = 5   # mínimo de encinas afectadas por foco
BUFFER_DIST = 10          # buffer (m) alrededor del hull del foco (0 = sin buffer)

# MDT / DEM
USE_DEM = True
DEM_LAYER_NAME = 'Extend_MDT'  # <-- pon aquí el nombre EXACTO de tu MDT en QGIS
DEM_PREFIX = 'z_'              # prefijo para el campo de cota de rastersampling
ADD_DEM_SAMPLED_LAYER = False  # pon True si quieres ver también la capa con cota muestreada

# -----------------------------------------------------------
# 1. Cargar capa de encinas y comprobaciones básicas
# -----------------------------------------------------------

layers = QgsProject.instance().mapLayersByName(TREE_LAYER_NAME)
if not layers:
    raise Exception(f"No se ha encontrado la capa '{TREE_LAYER_NAME}' en el proyecto.")

trees_layer = layers[0]
print(f"[1] Usando capa de encinas: {trees_layer.name()}")

if trees_layer.geometryType() != QgsWkbTypes.PointGeometry:
    raise Exception(
        f"La capa '{TREE_LAYER_NAME}' no es de puntos "
        f"(tipo: {QgsWkbTypes.displayString(trees_layer.wkbType())})."
    )

crs = trees_layer.crs()
if crs.isGeographic():
    print("AVISO: La capa está en coordenadas geográficas (grados). "
          "Las distancias no estarán en metros. "
          "Sería mejor reproyectar a un CRS proyectado (UTM, etc.).")

field_names = [f.name() for f in trees_layer.fields()]
if CLASS_FIELD not in field_names:
    raise Exception(
        f"El campo '{CLASS_FIELD}' no existe en la capa '{TREE_LAYER_NAME}'.\n"
        f"Campos disponibles: {field_names}"
    )

class_idx = trees_layer.fields().indexFromName(CLASS_FIELD)

# -----------------------------------------------------------
# 2. Calcular distancias al vecino más cercano (todas encinas)
# -----------------------------------------------------------

print("[1] Construyendo índice espacial para TODAS las encinas...")
index_all = QgsSpatialIndex(trees_layer.getFeatures())

id_to_point = {}
for feat in trees_layer.getFeatures():
    geom = feat.geometry()
    if geom is None or geom.isEmpty():
        continue
    pt = geom.asPoint()
    id_to_point[feat.id()] = QgsPointXY(pt.x(), pt.y())

print(f"[1] Total encinas (todas): {len(id_to_point)}")

distances = []
for fid, pt in id_to_point.items():
    nn_ids = index_all.nearestNeighbor(pt, 2)  # incluye el propio punto
    if len(nn_ids) < 2:
        continue
    nn_fid = nn_ids[1]
    pt2 = id_to_point[nn_fid]
    d = math.hypot(pt2.x() - pt.x(), pt2.y() - pt.y())
    distances.append(d)

if not distances:
    raise Exception("No se han podido calcular distancias al vecino más cercano.")

mean_nnd = sum(distances) / len(distances)
R = R_MULTIPLIER * mean_nnd

print(f"[1] Distancia media al vecino más cercano: {mean_nnd:.2f} unidades de mapa")
print(f"[1] Radio de agrupación R = {R:.2f} (R_MULTIPLIER = {R_MULTIPLIER})")

# -----------------------------------------------------------
# 3. Extraer encinas afectadas (clase = INFECTED_VALUE)
# -----------------------------------------------------------

infected_fids = []
infected_coords = []

for feat in trees_layer.getFeatures():
    val = feat[class_idx]
    if val == INFECTED_VALUE:
        fid = feat.id()
        if fid not in id_to_point:
            continue
        pt = id_to_point[fid]
        infected_fids.append(fid)
        infected_coords.append((pt.x(), pt.y()))

n_infected = len(infected_fids)
print(f"[2] Encinas afectadas (clase = {INFECTED_VALUE}): {n_infected}")

if n_infected < MIN_TREES_PER_FOCUS:
    raise Exception(
        f"Hay menos de {MIN_TREES_PER_FOCUS} encinas afectadas. "
        "No se pueden formar focos con ese mínimo."
    )

fid_to_idx = {fid: i for i, fid in enumerate(infected_fids)}

# -----------------------------------------------------------
# 4. Clustering tipo DBSCAN (componentes conexas con radio R)
# -----------------------------------------------------------

print("[2] Iniciando clustering de encinas afectadas...")

cluster_ids = [-1] * n_infected
current_cluster = 0

for i in range(n_infected):
    if cluster_ids[i] != -1:
        continue

    stack = [i]
    cluster_ids[i] = current_cluster

    while stack:
        k = stack.pop()
        xk, yk = infected_coords[k]
        pt_k = QgsPointXY(xk, yk)

        rect = QgsRectangle(xk - R, yk - R, xk + R, yk + R)
        cand_fids = index_all.intersects(rect)

        for cand_fid in cand_fids:
            if cand_fid not in fid_to_idx:
                continue
            j = fid_to_idx[cand_fid]
            if cluster_ids[j] != -1:
                continue

            xj, yj = infected_coords[j]
            d = math.hypot(xj - xk, yj - yk)
            if d <= R:
                cluster_ids[j] = current_cluster
                stack.append(j)

    current_cluster += 1

print(f"[2] Clusters brutos encontrados (incluyendo pequeños): {current_cluster}")

cluster_sizes = {}
for cid in cluster_ids:
    if cid == -1:
        continue
    cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1

valid_cluster_map = {}
new_cluster_id = 0
for cid, size in cluster_sizes.items():
    if size >= MIN_TREES_PER_FOCUS:
        valid_cluster_map[cid] = new_cluster_id
        new_cluster_id += 1

for i, cid in enumerate(cluster_ids):
    if cid in valid_cluster_map:
        cluster_ids[i] = valid_cluster_map[cid]
    else:
        cluster_ids[i] = -1  # ruido / grupos < MIN_TREES_PER_FOCUS

n_foci = len(valid_cluster_map)
print(f"[2] Focos válidos (clusters con >= {MIN_TREES_PER_FOCUS} encinas): {n_foci}")

if n_foci == 0:
    print("AVISO: no se ha identificado ningún foco con el mínimo establecido.")
    # se seguirá ejecutando, pero no habrá polígonos ni puntos de inicio útiles

# -----------------------------------------------------------
# 5. Capa de puntos: encinas_afectadas_clusters
# -----------------------------------------------------------

pts_layer = QgsVectorLayer(f'Point?crs={crs.authid()}', 'encinas_afectadas_clusters', 'memory')
pts_pr = pts_layer.dataProvider()

pts_fields = QgsFields()
for f in trees_layer.fields():
    pts_fields.append(QgsField(f.name(), f.type()))
pts_fields.append(QgsField('cluster_id', QVariant.Int))
pts_fields.append(QgsField('mean_nnd', QVariant.Double))
pts_fields.append(QgsField('R', QVariant.Double))

pts_pr.addAttributes(pts_fields)
pts_layer.updateFields()

for i, fid in enumerate(infected_fids):
    cid = cluster_ids[i]
    feat_src = trees_layer.getFeature(fid)

    new_feat = QgsFeature(pts_layer.fields())
    new_feat.setGeometry(feat_src.geometry())

    for f in trees_layer.fields():
        new_feat[f.name()] = feat_src[f.name()]

    new_feat['cluster_id'] = cid
    new_feat['mean_nnd'] = mean_nnd
    new_feat['R'] = R

    pts_pr.addFeatures([new_feat])

pts_layer.updateExtents()
QgsProject.instance().addMapLayer(pts_layer)
print("[2] Capa de puntos 'encinas_afectadas_clusters' creada y añadida al proyecto.")

# -----------------------------------------------------------
# 6. Capa de polígonos: focos_phytophthora (convex hull + buffer)
# -----------------------------------------------------------

poly_layer = QgsVectorLayer(f'Polygon?crs={crs.authid()}', 'focos_phytophthora', 'memory')
poly_pr = poly_layer.dataProvider()

poly_fields = QgsFields()
poly_fields.append(QgsField('focus_id', QVariant.Int))
poly_fields.append(QgsField('cluster_id', QVariant.Int))
poly_fields.append(QgsField('n_trees', QVariant.Int))
poly_fields.append(QgsField('mean_nnd', QVariant.Double))
poly_fields.append(QgsField('R', QVariant.Double))

poly_pr.addAttributes(poly_fields)
poly_layer.updateFields()

cluster_to_indices = {}
for i, cid in enumerate(cluster_ids):
    if cid == -1:
        continue
    cluster_to_indices.setdefault(cid, []).append(i)

focus_id = 1
for cid, idx_list in cluster_to_indices.items():
    pts = []
    for i in idx_list:
        x, y = infected_coords[i]
        pts.append(QgsPointXY(x, y))

    if len(pts) < MIN_TREES_PER_FOCUS:
        continue

    multi_geom = QgsGeometry.fromMultiPointXY(pts)
    hull = multi_geom.convexHull()

    if BUFFER_DIST > 0:
        poly_geom = hull.buffer(BUFFER_DIST, 8)
    else:
        poly_geom = hull

    feat_poly = QgsFeature(poly_layer.fields())
    feat_poly.setGeometry(poly_geom)
    feat_poly['focus_id'] = focus_id
    feat_poly['cluster_id'] = cid
    feat_poly['n_trees'] = len(pts)
    feat_poly['mean_nnd'] = mean_nnd
    feat_poly['R'] = R

    poly_pr.addFeatures([feat_poly])
    focus_id += 1

poly_layer.updateExtents()
QgsProject.instance().addMapLayer(poly_layer)
print(f"[2] Capa de polígonos 'focos_phytophthora' creada con {focus_id - 1} focos y añadida al proyecto.")

# -----------------------------------------------------------
# 7. Muestrear MDT/DEM sobre encinas_afectadas_clusters
# -----------------------------------------------------------

points_layer = pts_layer
elev_field_name = None

if USE_DEM:
    dem_layers = QgsProject.instance().mapLayersByName(DEM_LAYER_NAME)
    if not dem_layers:
        raise Exception(f"No se ha encontrado la capa raster '{DEM_LAYER_NAME}' para el MDT/DEM.")

    dem_layer = dem_layers[0]
    if not isinstance(dem_layer, QgsRasterLayer):
        raise Exception(
            f"La capa '{DEM_LAYER_NAME}' no es un raster "
            f"(es {dem_layer.__class__.__name__})."
        )

    print(f"[3] Usando MDT/DEM: {dem_layer.name()} para muestrear cotas...")

    params_rs = {
        'INPUT': pts_layer,
        'RASTERCOPY': dem_layer,
        'COLUMN_PREFIX': DEM_PREFIX,
        'OUTPUT': 'TEMPORARY_OUTPUT'
    }

    res_rs = processing.run('native:rastersampling', params_rs)
    sampled = res_rs['OUTPUT']

    if isinstance(sampled, str):
        points_layer = QgsVectorLayer(sampled, 'encinas_afectadas_clusters_dem', 'ogr')
    else:
        points_layer = sampled

    if not points_layer or not points_layer.isValid():
        raise Exception("No se ha podido cargar la capa resultante de 'native:rastersampling'.")

    for f in points_layer.fields():
        if f.name().startswith(DEM_PREFIX):
            elev_field_name = f.name()
            break

    if elev_field_name is None:
        raise Exception(
            f"No se ha encontrado ningún campo de cota que empiece por '{DEM_PREFIX}' "
            "en la capa muestreada."
        )

    if ADD_DEM_SAMPLED_LAYER:
        QgsProject.instance().addMapLayer(points_layer)

    print(f"[3] Cota muestreada en el campo: {elev_field_name}")

# -----------------------------------------------------------
# 8. Agrupar por cluster_id en la capa con DEM
# -----------------------------------------------------------

cluster_idx_pts = points_layer.fields().indexFromName('cluster_id')
if cluster_idx_pts == -1:
    raise Exception("No se ha encontrado el campo 'cluster_id' en la capa de puntos muestreada.")

clusters = {}  # cluster_id -> lista de QgsFeature

for feat in points_layer.getFeatures():
    cid = feat[cluster_idx_pts]
    if cid is None:
        continue
    try:
        cid_val = int(cid)
    except:
        continue
    if cid_val < 0:
        continue
    clusters.setdefault(cid_val, []).append(feat)

print(f"[3] Clusters (focos) encontrados en la capa de puntos con DEM: {len(clusters)}")

if not clusters:
    raise Exception("No hay clusters con cluster_id >= 0 en la capa con DEM.")

# -----------------------------------------------------------
# 9. Funciones auxiliares para elegir el "punto de inicio"
# -----------------------------------------------------------

def choose_start_point_centroid(cluster_feats):
    coords = []
    for f in cluster_feats:
        geom = f.geometry()
        if geom is None or geom.isEmpty():
            continue
        pt = geom.asPoint()
        coords.append((f, pt.x(), pt.y()))

    if not coords:
        return None, None

    xs = [c[1] for c in coords]
    ys = [c[2] for c in coords]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    best_feat = None
    best_d2 = None
    for f, x, y in coords:
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_feat = f

    return best_feat, 'centroid'

def choose_start_point_dem(cluster_feats, elev_field):
    best_feat = None
    best_z = None

    for f in cluster_feats:
        z = f[elev_field]
        if z is None:
            continue
        try:
            z_val = float(z)
        except:
            continue
        if best_z is None or z_val > best_z:
            best_z = z_val
            best_feat = f

    if best_feat is not None:
        return best_feat, 'dem_max'
    else:
        return None, 'dem_max'

# -----------------------------------------------------------
# 10. Capa de salida: puntos_inicio_focos
# -----------------------------------------------------------

start_layer = QgsVectorLayer(f'Point?crs={crs.authid()}', 'puntos_inicio_focos', 'memory')
start_pr = start_layer.dataProvider()

start_fields = QgsFields()
start_fields.append(QgsField('focus_id', QVariant.Int))   # asumimos focus_id ≈ cluster_id + 1
start_fields.append(QgsField('cluster_id', QVariant.Int))
start_fields.append(QgsField('n_trees', QVariant.Int))
start_fields.append(QgsField('method', QVariant.String))
if USE_DEM:
    start_fields.append(QgsField('elev', QVariant.Double))

start_pr.addAttributes(start_fields)
start_layer.updateFields()

n_created = 0

for cid, feat_list in clusters.items():
    if not feat_list:
        continue

    chosen_feat = None
    method_used = None
    elev_value = None

    # Primero intentamos con MDT (cota máxima)
    if USE_DEM and elev_field_name is not None:
        chosen_feat, method_used = choose_start_point_dem(feat_list, elev_field_name)
        if chosen_feat is not None:
            z_raw = chosen_feat[elev_field_name]
            try:
                elev_value = float(z_raw) if z_raw is not None else None
            except:
                elev_value = None

    # Si falla MDT, usamos centroide
    if chosen_feat is None:
        chosen_feat, method_used = choose_start_point_centroid(feat_list)

    if chosen_feat is None:
        continue

    geom = chosen_feat.geometry()
    if geom is None or geom.isEmpty():
        continue

    out_feat = QgsFeature(start_layer.fields())
    out_feat.setGeometry(geom)
    out_feat['cluster_id'] = int(cid)
    out_feat['focus_id'] = int(cid) + 1   # consistente con scripts anteriores
    out_feat['n_trees'] = len(feat_list)
    out_feat['method'] = method_used
    if USE_DEM and elev_field_name is not None and elev_value is not None:
        out_feat['elev'] = elev_value

    start_pr.addFeatures([out_feat])
    n_created += 1

start_layer.updateExtents()
QgsProject.instance().addMapLayer(start_layer)

print(f"[3] Capa 'puntos_inicio_focos' creada con {n_created} puntos (uno por foco).")
print("Método: 'dem_max' = cota más alta; 'centroid' = más cercano al centro geométrico.")
