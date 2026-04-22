Generate_binary_habitat_masks.py -> 
This script generates binary mask rasters for each input TIFF by intersecting a habitat polygon layer with the raster extent and rasterizing the result. It was used to delineated the dehesas habitat within the PNOA orthophotos. 
USE: PyQGIS 

Orthophoto_oak_detection_workflow.py -> 
This script performs a batch orthophoto processing workflow that resamples input rasters, creates aligned binary masks from a vector overlay, segments oak crowns using RGB-based ExGR and Otsu thresholding, and labels individual crown clusters as connected components.
USE: PyQGIS

Reference_clump_overlap_min_pixels_batch.py ->
This script batch-processes labeled clump rasters against a fixed reference raster, reprojects and aligns each candidate raster when necessary, computes overlap counts for reference object IDs, applies a minimum overlap threshold, and exports a two-band GeoTIFF with the resulting overlap mask and the original reference labels.
USE: PyQGIS

Reference_clump_overlap_min_percent_batch.py ->
Compares a reference labeled raster against all candidate clump rasters in a folder, aligns each candidate to the reference grid if needed, identifies reference IDs whose best overlap with a candidate reaches a minimum percentage of their area, and writes a two-band output with the overlap mask and the original reference labels.
USE: PyQGIS

Matrix_confusion.py ->
This PyQGIS workflow creates stratified random validation points within a mask based on predicted classes from a binary raster, allows manual assignment of reference labels, and then computes the confusion matrix, overall accuracy, class-specific producer’s and user’s accuracy, confidence intervals, and kappa statistic.
USE: PyQGIS

Phytophthora_focus_detection_workflow.py ->
This script estimates the typical tree spacing from all oak points, clusters infected trees into candidate Phytophthora foci using a distance-based DBSCAN-like approach, creates focus polygons from clustered trees, samples DEM elevation values, and assigns a start point to each focus using the highest-elevation infected tree or the cluster centroid as fallback.
USE: PyQGIS
