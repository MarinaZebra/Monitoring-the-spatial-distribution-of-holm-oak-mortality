# ================================
# PyQGIS VALIDATION WORKFLOW (QGIS Console)
# 1) MODE='generate' -> creates stratified sample points INSIDE MASK with 'pred' (0/1) and empty 'truth'.
# 2) MODE='compute'  -> after you fill 'truth' (0/1), prints confusion matrix + metrics.
# ================================

from qgis.core import (
    QgsProject, QgsGeometry, QgsPointXY, QgsVectorLayer, QgsField, QgsWkbTypes,
    QgsCoordinateTransform, QgsCoordinateReferenceSystem, QgsRaster, QgsFeature
)
from qgis.PyQt.QtCore import QVariant
import random, math, time

# -------- USER PARAMS ----------
MODE = 'compute'     # 'generate' or 'compute'

MASK_LAYER_NAME = 'Mascara_zona_piloto'                 # Polygon mask layer (AOI where the algorithm ran)
CLASSIF_RASTER_NAME = 'PNOA_ANUAL_2010_OF_ETRS89_HU30_h25_0624-1_encina_mask'    # Binary raster (1=canopy; 0/NULL=grass)
OUT_LAYER_NAME = 'points_validate'       # Output memory layer name (overwrites in-memory layer if exists)

# Sampling params (used when MODE='generate')
N_TOTAL = 200             # total desired validation points (e.g., ~200 for 95%±5%)
ALLOCATION = 'equal'      # 'equal' (recommended). (Proportional can be added if you give class proportions.)
MIN_DIST = 5.0            # minimum spacing between points (in layer CRS units; meters if projected)
CANDIDATE_MULT = 8        # generate ~N_TOTAL*CANDIDATE_MULT candidates before spacing filter
SEED = 42                 # random seed

# Class semantics
CANOPY_VALUE = 1                 # raster canopy value
GRASS_IS_NULL_OR_ZERO = True     # treat NULL or 0 as grass
RASTER_BAND = 1                  # band index to read from

# -------------------------------

def get_layer_by_name(name):
    ls = QgsProject.instance().mapLayersByName(name)
    return ls[0] if ls else None

def unary_union_geoms(layer):
    geoms = [f.geometry() for f in layer.getFeatures() if not f.geometry().isEmpty()]
    if not geoms:
        return None
    # robust union
    u = QgsGeometry.unaryUnion(geoms)
    if not u or u.isEmpty():
        # fallback: iterative combine
        u = geoms[0]
        for g in geoms[1:]:
            u = u.combine(g)
        u = u.makeValid()
    return u

def random_point_in_geom(geom: QgsGeometry):
    # bounding-box rejection (simple & reliable for large AOIs)
    bb = geom.boundingBox()
    for _ in range(3000):
        x = random.uniform(bb.xMinimum(), bb.xMaximum())
        y = random.uniform(bb.yMinimum(), bb.yMaximum())
        p = QgsPointXY(x, y)
        if geom.contains(QgsGeometry.fromPointXY(p)):
            return p
    return None

def transform_point(pt_xy, src_crs, dst_crs):
    if src_crs.authid() == dst_crs.authid():
        return pt_xy
    xform = QgsCoordinateTransform(src_crs, dst_crs, QgsProject.instance())
    p = xform.transform(pt_xy)
    return QgsPointXY(p.x(), p.y())

def sample_raster_value_at_point(rlayer, point_xy_in_raster_crs, band=1):
    prov = rlayer.dataProvider()
    # Identify returns a dict keyed by band
    res = prov.identify(point_xy_in_raster_crs, QgsRaster.IdentifyFormatValue)
    if not res.isValid():
        return None
    vals = res.results()
    if not vals:
        return None
    # band key can be 1-based (band index)
    v = vals.get(band, list(vals.values())[0])
    return v

def greedy_min_dist_select(features, min_dist):
    """Greedy spacing without relying on feature IDs or a spatial index."""
    if min_dist <= 0 or len(features) <= 1:
        return features[:]
    sel = []
    sel_geoms = []
    for f in features:
        g = f.geometry()
        keep = True
        for sg in sel_geoms:
            if g.distance(sg) < min_dist:
                keep = False
                break
        if keep:
            sel.append(f)
            sel_geoms.append(g)
    return sel

def make_memory_point_layer(crs, name):
    vl = QgsVectorLayer(f'Point?crs={crs.authid()}', name, 'memory')
    pr = vl.dataProvider()
    pr.addAttributes([
        QgsField('pred', QVariant.Int),     # predicted class (0/1)
        QgsField('truth', QVariant.Int),    # to be filled by you (0/1)
        QgsField('stratum', QVariant.String) # 'canopy'/'grass' by predicted class
    ])
    vl.updateFields()
    return vl

def compute_confusion(points_layer):
    TP=TN=FP=FN=0
    total=0
    for f in points_layer.getFeatures():
        truth = f['truth']
        pred  = f['pred']
        if truth is None or pred is None:
            continue
        try:
            truth = int(truth)
            pred  = int(pred)
        except:
            continue
        if truth not in (0,1) or pred not in (0,1):
            continue
        total += 1
        if   truth==1 and pred==1: TP += 1
        elif truth==0 and pred==0: TN += 1
        elif truth==0 and pred==1: FP += 1
        elif truth==1 and pred==0: FN += 1

    if total == 0:
        print("No samples with non-null truth. Fill 'truth' (0/1) and rerun.")
        return

    oa = (TP+TN)/total
    prod_can = TP/(TP+FN) if (TP+FN)>0 else float('nan')
    user_can = TP/(TP+FP) if (TP+FP)>0 else float('nan')
    prod_grs = TN/(TN+FP) if (TN+FP)>0 else float('nan')
    user_grs = TN/(TN+FN) if (TN+FN)>0 else float('nan')

    # Kappa
    row1 = TP+FN; row0 = FP+TN
    col1 = TP+FP; col0 = FN+TN
    pe = ((row1*col1) + (row0*col0)) / (total**2)
    kappa = (oa - pe) / (1 - pe) if (1-pe)!=0 else float('nan')

    # 95% CI for OA (Wilson)
    z = 1.96
    p = oa; N = total
    num = p + (z*z)/(2*N)
    rad = z*math.sqrt((p*(1-p) + (z*z)/(4*N))/N)
    den = 1 + (z*z)/N
    ci_low = (num - rad)/den
    ci_high = (num + rad)/den

    print("=== Confusion Matrix (truth rows, pred cols) ===")
    print(f"            pred=1   pred=0")
    print(f"truth=1     {TP:6d}   {FN:6d}")
    print(f"truth=0     {FP:6d}   {TN:6d}\n")
    print(f"N={total}")
    print(f"Overall Accuracy: {oa:.3f}  (95% CI {ci_low:.3f}–{ci_high:.3f})")
    print(f"Canopy  - Producer's (recall):  {prod_can:.3f}   User's (precision): {user_can:.3f}")
    print(f"Grass   - Producer's (recall):  {prod_grs:.3f}   User's (precision): {user_grs:.3f}")
    print(f"Kappa: {kappa:.3f}")

# ----------------- MAIN -----------------
random.seed(SEED)

if MODE.lower() == 'generate':
    mask = get_layer_by_name(MASK_LAYER_NAME)
    ras  = get_layer_by_name(CLASSIF_RASTER_NAME)
    if mask is None:
        raise Exception(f"Mask layer '{MASK_LAYER_NAME}' not found.")
    if ras is None:
        raise Exception(f"Raster layer '{CLASSIF_RASTER_NAME}' not found.")
    if mask.wkbType() not in (QgsWkbTypes.Polygon, QgsWkbTypes.MultiPolygon):
        raise Exception("Mask must be a polygon layer.")

    # unified mask geometry
    print("Preparing mask geometry…")
    geom_u = unary_union_geoms(mask)
    if not geom_u or geom_u.isEmpty():
        raise Exception("Mask has empty geometry.")

    # allocation
    if ALLOCATION.lower() == 'equal':
        n_can = N_TOTAL // 2
        n_grs = N_TOTAL - n_can
    else:
        # (add proportional allocation here if you provide class proportions)
        n_can = N_TOTAL // 2
        n_grs = N_TOTAL - n_can

    # create candidates and classify by raster
    pool_target = max(N_TOTAL * CANDIDATE_MULT, 1000)
    print(f"Generating ~{pool_target} candidate points inside mask…")
    pool_can = []
    pool_grs = []

    mask_crs = mask.crs()
    ras_crs = ras.crs()

    attempts = 0
    made = 0
    t0 = time.time()
    while made < pool_target and attempts < pool_target * 40:
        attempts += 1
        p_mask = random_point_in_geom(geom_u)
        if p_mask is None:
            continue
        # transform to raster CRS before sampling
        p_ras = transform_point(p_mask, mask_crs, ras_crs)
        v = sample_raster_value_at_point(ras, p_ras, band=RASTER_BAND)

        if v is None:
            pred = 0 if GRASS_IS_NULL_OR_ZERO else None
        else:
            try:
                vv = float(v)
                if math.isnan(vv):
                    pred = 0 if GRASS_IS_NULL_OR_ZERO else None
                else:
                    pred = 1 if int(round(vv)) == CANOPY_VALUE else 0
            except:
                pred = 0

        f = QgsFeature()
        f.setGeometry(QgsGeometry.fromPointXY(p_mask))  # store in MASK CRS
        f.setAttributes([pred, None, 'canopy' if pred==1 else 'grass'])

        if pred == 1:
            pool_can.append(f)
        else:
            pool_grs.append(f)
        made += 1

    print(f"Candidate pool: canopy={len(pool_can)}, grass={len(pool_grs)}")
    if len(pool_can) < n_can:
        print(f"WARNING: canopy candidates {len(pool_can)} < requested {n_can}.")
    if len(pool_grs) < n_grs:
        print(f"WARNING: grass candidates {len(pool_grs)} < requested {n_grs}.")

    # enforce spacing per stratum
    random.shuffle(pool_can)
    random.shuffle(pool_grs)
    sel_can = greedy_min_dist_select(pool_can, MIN_DIST)[:n_can]
    sel_grs = greedy_min_dist_select(pool_grs, MIN_DIST)[:n_grs]

    # build memory layer
    out = make_memory_point_layer(mask.crs(), OUT_LAYER_NAME)
    pr = out.dataProvider()
    pr.addFeatures(sel_can + sel_grs)
    out.updateExtents()

    # replace any in-memory layer with same name
    existing = get_layer_by_name(OUT_LAYER_NAME)
    if existing:
        QgsProject.instance().removeMapLayer(existing.id())
    QgsProject.instance().addMapLayer(out)

    print(f"Created '{OUT_LAYER_NAME}' with {out.featureCount()} points.")
    print("Fields:")
    print("  pred   -> predicted class from raster (0=grass, 1=canopy)")
    print("  truth  -> fill manually by visual inspection (0/1)")
    print("  stratum-> 'canopy' or 'grass' (by predicted class)")
    print("Next: toggle edit on this layer and set 'truth' (0/1) for each point, using the orthophoto for reference.")
    print("Then set MODE='compute' at the top and run again.")

elif MODE.lower() == 'compute':
    pts = get_layer_by_name(OUT_LAYER_NAME)
    if pts is None:
        raise Exception(f"Points layer '{OUT_LAYER_NAME}' not found. Run first in MODE='generate'.")
    # sanity check
    fld_names = [f.name() for f in pts.fields()]
    if not all(n in fld_names for n in ('pred','truth')):
        raise Exception("Points layer must contain 'pred' and 'truth' fields.")
    compute_confusion(pts)

else:
    raise Exception("MODE must be 'generate' or 'compute'.")
