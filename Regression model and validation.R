############################################################
## REGRESIÓN LOGÍSTICA + BLOCK CV ESPACIAL + ESTABILIDAD
## LOGISTIC REGRESSION + SPATIAL BLOCK CROSS-VALIDATION + STABILITY
############################################################
# install.packages(c("terra","caret","pROC","ggplot2","dplyr","ggeffects","patchwork"))

library(terra)
library(caret)
library(pROC)       # requerido por twoClassSummary / required by twoClassSummary
library(ggplot2)    # para el gráfico CV(500) vs CV(1000) / for CV plot
library(dplyr)      # recode/mutate / tidy helpers
library(ggeffects)  # marginal effects
library(patchwork)  # multipanel plots

############################################
## 0. Ruta base / Base path
############################################
ruta_base <- "E:/TFM/REGRESION LOGISTICA FINAL"

outdir <- file.path(ruta_base, "salidas_spatial_block_cv")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

############################################
## 1. Carga de ráster estado + predictores
##    Oak raster + Predictor loading
############################################
encinas   <- rast(file.path(ruta_base, "Oaks Map.tif"))

aspecto    <- rast(file.path(ruta_base, "Aspect.tif"))
caminos    <- rast(file.path(ruta_base, "Distance to roads.tif"))
dist_foci  <- rast(file.path(ruta_base, "Distance to original foci .tif"))
pendiente  <- rast(file.path(ruta_base, "Slope %.tif"))
plan_curv  <- rast(file.path(ruta_base, "Plan curvature index.tif"))
twi        <- rast(file.path(ruta_base, "Topographic wetness index.tif"))
flow_acc   <- rast(file.path(ruta_base, "Proximity to the Water Network.tif"))

# Chequeo geometría / Geometry check
ok_geom <- compareGeom(encinas, aspecto, caminos, dist_foci, pendiente, plan_curv, twi, flow_acc, stopOnError = FALSE)
print(ok_geom)

############################################
## 2. Limpieza del ráster de estado / Removing value 255
############################################
estado <- encinas
estado[estado == 255] <- NA

############################################
## 3. Stack + extracción a data.frame / Stack + extract to data.frame
############################################
vars <- c(dist_foci, flow_acc, plan_curv, caminos, aspecto, twi, pendiente)
names(vars) <- c("dist_foci", "flow_acc", "plan_curv",
                 "dist_caminos", "aspect", "twi", "pendiente")

full_stack <- c(estado, vars)
names(full_stack)[1] <- "estado"

df_encinas <- as.data.frame(full_stack, xy = TRUE, na.rm = TRUE)
cat("\nN filas:", nrow(df_encinas), "\n")
print(table(df_encinas$estado))

############################################
## 4. Aspecto -> sin/cos  / Aspect > sin/cos
############################################
df_encinas$aspect_rad <- df_encinas$aspect * pi / 180
df_encinas$aspect_sin <- sin(df_encinas$aspect_rad)
df_encinas$aspect_cos <- cos(df_encinas$aspect_rad)

############################################
## 5. Correlación entre predictores (colinealidad)
##    Correlation among predictors (collinearity)
############################################
vars_modelo <- df_encinas[, c("dist_foci", "flow_acc", "plan_curv",
                              "dist_caminos", "twi", "pendiente",
                              "aspect_sin", "aspect_cos")]

cor_mat <- cor(vars_modelo, use = "pairwise.complete.obs", method = "pearson")
cat("\nMatriz de correlación (redondeada):\n")
print(round(cor_mat, 2))

############################################
## 6. Dataset de modelado (IMPORTANTE: incluye x,y)
##    Modelling dataset (IMPORTANT: includes x,y)
############################################
df_model <- df_encinas[, c("x","y","estado",
                           "dist_foci","flow_acc","plan_curv",
                           "dist_caminos","twi","pendiente",
                           "aspect_sin","aspect_cos")]

df_model$afectada <- df_model$estado

df_model$afectada_f <- factor(
  ifelse(df_model$afectada == 1, "afectada", "sana"),
  levels = c("afectada", "sana")  # "afectada" = clase positiva / positive class
)

preds <- c("dist_foci","flow_acc","plan_curv","dist_caminos",
           "twi","pendiente","aspect_sin","aspect_cos")

# Guardar medias/sd globales por si luego quieres escalar rásteres para mapear predicción
# Save global mean/sd (optional; useful for raster standardization later)
medias_global <- sapply(df_model[, preds], mean, na.rm = TRUE)
sds_global    <- sapply(df_model[, preds], sd,   na.rm = TRUE)

############################################
## 7. Modelo regresión logística original con CV aleatoria
##    Original logistic regression model with random cross-validation
############################################
set.seed(123)
ctrl_random <- trainControl(
  method          = "repeatedcv",
  number          = 5,
  repeats         = 3,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  sampling        = "up"
)

modelo_random <- train(
  afectada_f ~ dist_foci + flow_acc + plan_curv +
    dist_caminos + twi + pendiente +
    aspect_sin + aspect_cos,
  data      = df_model,
  method    = "glm",
  family    = binomial,
  metric    = "ROC",
  trControl = ctrl_random,
  preProcess = c("center","scale")
)

cat("\nModelo CV aleatoria:\n")
print(modelo_random)
print(varImp(modelo_random))

############################################
## 7bis. Marginal effects (TODAS las variables del modelo)
############################################

# Modelo "inferencial" (sin upsampling) para interpretar probabilidades
# Inference-style model (no upsampling) for interpretable probabilities
fit_infer <- glm(
  afectada ~ dist_foci + flow_acc + plan_curv +
    dist_caminos + twi + pendiente +
    aspect_sin + aspect_cos,
  data = df_model,
  family = binomial
)

summary(fit_infer)                 # imprime toda la salida
summary(fit_infer)$coefficients

# Carpeta de salida / Output folder
out_me <- file.path(outdir, "marginal_effects_ggeffects")
dir.create(out_me, showWarnings = FALSE, recursive = TRUE)

# 1) Predictores: TODOS los del modelo (automático)
key_vars <- attr(terms(fit_infer), "term.labels")

# 2) Etiquetas (si falta alguna, se usa el nombre de la variable)
label_x <- c(
  dist_foci     = "Distance to focus (m)",
  flow_acc      = "Flow accumulation",
  plan_curv     = "Plan curvature",
  dist_caminos  = "Distance to paths (m)",
  twi           = "TWI",
  pendiente     = "Slope (%)",
  aspect_sin    = "Aspect (sin)",
  aspect_cos    = "Aspect (cos)"
)

label_title <- c(
  dist_foci     = "Marginal effect: Distance to focus",
  flow_acc      = "Marginal effect: Flow accumulation",
  plan_curv     = "Marginal effect: Plan curvature",
  dist_caminos  = "Marginal effect: Distance to paths",
  twi           = "Marginal effect: TWI",
  pendiente     = "Marginal effect: Slope",
  aspect_sin    = "Marginal effect: Aspect (sin)",
  aspect_cos    = "Marginal effect: Aspect (cos)"
)

get_lab <- function(var, lab_vec) {
  if (!is.null(lab_vec[[var]]) && !is.na(lab_vec[[var]])) lab_vec[[var]] else var
}

plot_me <- function(var){
  
  eff <- ggpredict(fit_infer, terms = paste0(var, " [all]"))
  
  ggplot(eff, aes(x = x, y = predicted)) +
    geom_line(linewidth = 0.9) +
    geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.2) +
    labs(
      title = get_lab(var, label_title),
      x     = get_lab(var, label_x),
      y     = "Predicted probability of affected oaks"
    ) +
    theme_minimal(base_size = 12)
}

# Crear plots / Create plots (TODAS)
me_plots <- lapply(key_vars, plot_me)
names(me_plots) <- key_vars

# Guardar individuales / Save individual plots
for (v in key_vars) {
  ggsave(
    filename = file.path(out_me, paste0("ME_", v, ".png")),
    plot = me_plots[[v]],
    width = 7, height = 5, dpi = 300
  )
}

# Multipanel automático con TODOS / Auto multipanel for ALL predictors
ncol_mp <- 3
nrow_mp <- ceiling(length(key_vars) / ncol_mp)

p_multi_all <- wrap_plots(me_plots, ncol = ncol_mp)

ggsave(
  filename = file.path(out_me, "ME_multipanel_ALL_predictors.png"),
  plot = p_multi_all,
  width = 14,
  height = max(6, 3.5 * nrow_mp),
  dpi = 300
)

cat("\nEfectos marginales (todos) guardados en:\n", out_me, "\n")

############################################
## 8. Block CV espacial + estabilidad de coeficientes
##    Spatial block CV + coefficient stability
############################################

# Fórmulas / Formulas
form_caret <- afectada_f ~ dist_foci + flow_acc + plan_curv +
  dist_caminos + twi + pendiente + aspect_sin + aspect_cos

form_glm <- afectada ~ dist_foci + flow_acc + plan_curv +
  dist_caminos + twi + pendiente + aspect_sin + aspect_cos

# Referencia espacial para los bloques / Spatial reference for the blocks
xmin0 <- xmin(encinas)
ymin0 <- ymin(encinas)

run_block_cv <- function(block_size, k = 5, seed = 123) {
  
  # ---- Crear block_id (grid) / Create block_id (grid)
  bx <- floor((df_model$x - xmin0) / block_size)
  by <- floor((df_model$y - ymin0) / block_size)
  block_id <- interaction(bx, by, drop = TRUE)
  n_blocks <- nlevels(block_id)
  
  # ---- folds por bloque / folds by block
  set.seed(seed)
  index <- groupKFold(block_id, k = k)
  indexOut <- lapply(index, function(tr) setdiff(seq_len(nrow(df_model)), tr))
  
  cat("\n====================================\n")
  cat("Block size:", block_size, "m\n")
  cat("N bloques:", n_blocks, "\n")
  cat("N folds (index):", length(index), "\n")
  
  # ---- Prevalencia por fold (TEST) / Prevalence by fold (TEST)
  fold_prev <- lapply(seq_along(indexOut), function(i){
    te <- indexOut[[i]]
    tab <- prop.table(table(df_model$afectada_f[te]))
    data.frame(
      block_size = block_size,
      fold = paste0("Fold", i),
      prev_afectada = unname(tab["afectada"]),
      prev_sana = unname(tab["sana"])
    )
  })
  fold_prev <- do.call(rbind, fold_prev)
  print(fold_prev)
  
  # ---- Control caret (upsampling para rendimiento) / Caret control (upsampling for performance)
  ctrl <- trainControl(
    method = "cv",
    index = index,
    indexOut = indexOut,
    classProbs = TRUE,
    summaryFunction = twoClassSummary,
    savePredictions = "final",
    sampling = "up"
  )
  
  # ---- Modelo caret con escalado dentro de folds / Caret model with scaling within folds
  set.seed(seed)
  mod <- train(
    form_caret,
    data = df_model,
    method = "glm",
    family = binomial,
    metric = "ROC",
    trControl = ctrl,
    preProcess = c("center","scale")
  )
  
  # ---- Performance summary
  perf <- data.frame(
    block_size = block_size,
    n_samples = nrow(df_model),
    n_blocks  = n_blocks,
    ROC_mean  = mean(mod$resample$ROC),
    ROC_sd    = sd(mod$resample$ROC),
    Sens_mean = mean(mod$resample$Sens),
    Sens_sd   = sd(mod$resample$Sens),
    Spec_mean = mean(mod$resample$Spec),
    Spec_sd   = sd(mod$resample$Spec),
    prev_afectada_min = min(fold_prev$prev_afectada, na.rm = TRUE),
    prev_afectada_max = max(fold_prev$prev_afectada, na.rm = TRUE)
  )
  
  # ---- Estabilidad de coeficientes (sin upsampling, para interpretar efectos)
  #      Coefficient stability (no upsampling, for effect interpretation)
  sd_global <- sapply(df_model[, preds], sd, na.rm = TRUE)
  
  coef_list <- lapply(seq_along(index), function(i){
    tr <- index[[i]]
    fit <- glm(form_glm, data = df_model[tr, ], family = binomial)
    
    b <- coef(fit)
    b <- b[names(b) != "(Intercept)"]
    
    # estandarización "por 1 SD global" / standardization
    b_std <- b * sd_global[names(b)]
    b_std
  })
  
  coef_mat <- do.call(rbind, coef_list)
  
  stab <- t(apply(coef_mat, 2, function(v){
    m <- mean(v); s <- sd(v)
    c(media = m,
      sd = s,
      CV = s / abs(m),
      min = min(v),
      max = max(v))
  }))
  
  stab_df <- data.frame(
    block_size = block_size,
    predictor = rownames(stab),
    stab,
    row.names = NULL
  )
  
  folds_metrics <- mod$resample
  folds_metrics$block_size <- block_size
  
  list(
    model = mod,
    perf = perf,
    fold_prev = fold_prev,
    fold_metrics = folds_metrics,
    stability = stab_df
  )
}

# Ejecutar para 500 y 1000 / Run for 500 and 1000
block_sizes <- c(500, 1000)
results <- lapply(block_sizes, function(bs) run_block_cv(bs, k = 5, seed = 123))
names(results) <- paste0("block_", block_sizes)

perf_all <- do.call(rbind, lapply(results, `[[`, "perf"))
stab_all <- do.call(rbind, lapply(results, `[[`, "stability"))
fold_prev_all <- do.call(rbind, lapply(results, `[[`, "fold_prev"))
fold_metrics_all <- do.call(rbind, lapply(results, `[[`, "fold_metrics"))

cat("\nResumen performance por block_size:\n")
print(perf_all)

############################################
## 9. Exportar CSVs / Export CSVs
############################################
write.csv(perf_all,         file.path(outdir, "spatial_blockcv_performance_summary.csv"), row.names = FALSE)
write.csv(stab_all,         file.path(outdir, "spatial_blockcv_coeff_stability.csv"),     row.names = FALSE)
write.csv(fold_prev_all,    file.path(outdir, "spatial_blockcv_fold_prevalence.csv"),     row.names = FALSE)
write.csv(fold_metrics_all, file.path(outdir, "spatial_blockcv_fold_metrics.csv"),        row.names = FALSE)

saveRDS(results[["block_500"]]$model,  file.path(outdir, "modelo_caret_block500.rds"))
saveRDS(results[["block_1000"]]$model, file.path(outdir, "modelo_caret_block1000.rds"))

############################################
## 10. Gráfico: comparación CV(500) vs CV(1000) por predictor / Plotting
############################################

# 1) Diccionario: nombre original -> nombre que quieres mostrar
pred_labels <- c(
  twi          = "TWI (Topographic Wetness Index)",
  plan_curv    = "Plan curvature",
  pendiente    = "Slope (%)",
  flow_acc     = "Flow accumulation",
  dist_foci    = "Distance to focus (m)",
  dist_caminos = "Distance to paths (m)",
  aspect_sin   = "Aspect (sin)",
  aspect_cos   = "Aspect (cos)"
)

# 2) Crear etiqueta para plot
stab_plot <- stab_all %>%
  mutate(
    block_size_f = factor(block_size, levels = sort(unique(block_size))),
    predictor_label = recode(predictor, !!!pred_labels),
    # orden del eje (opcional): CV a 1000m de menor a mayor
    predictor_label = factor(predictor_label, levels = {
      tmp <- stab_all[stab_all$block_size == 1000, c("predictor","CV")]
      tmp <- tmp[order(tmp$CV), ]
      recode(tmp$predictor, !!!pred_labels)
    })
  )

# 3) Plot usando predictor_label
p_cv <- ggplot(stab_plot, aes(x = predictor_label, y = CV, color = block_size_f)) +
  geom_point(size = 2, position = position_dodge(width = 0.4)) +
  geom_line(aes(group = predictor_label), alpha = 0.3) +
  coord_flip() +
  labs(
    title = "Stability of coefficients across block sizes",
    subtitle = "CV = SD / |mean| of standardized coefficients by fold (lower = more stable)",
    x = "Predictor",
    y = "Coefficient of variation (CV)",
    color = "Block size (m)"
  ) +
  theme_minimal(base_size = 12)

print(p_cv)
ggsave(file.path(outdir, "CV_comparison_500_vs_1000.png"), p_cv, width = 10, height = 6, dpi = 300)

############################################
## 11. (Opcional) Guardar mapas de predictores (GeoTIFF + PNG)
##     (Optional) Save predictor maps (GeoTIFF + PNG)
############################################
out_pred <- file.path(ruta_base, "salidas_predictores")
dir.create(out_pred, showWarnings = FALSE, recursive = TRUE)

# Guardar cada predictor original / Save each original predictor
for (nm in names(vars)) {
  writeRaster(vars[[nm]],
              filename = file.path(out_pred, paste0(nm, ".tif")),
              overwrite = TRUE,
              gdal = c("COMPRESS=LZW"))
}

# Guardar sin/cos del aspecto como rásteres / Save aspect sin/cos rasters
aspect_rad_r <- aspecto * pi / 180
aspect_sin_r <- sin(aspect_rad_r); names(aspect_sin_r) <- "aspect_sin"
aspect_cos_r <- cos(aspect_rad_r); names(aspect_cos_r) <- "aspect_cos"

writeRaster(aspect_sin_r, file.path(out_pred, "aspect_sin.tif"), overwrite = TRUE, gdal = c("COMPRESS=LZW"))
writeRaster(aspect_cos_r, file.path(out_pred, "aspect_cos.tif"), overwrite = TRUE, gdal = c("COMPRESS=LZW"))

# Quicklooks en PNG / Quicklook PNGs
for (nm in names(vars)) {
  png(file.path(out_pred, paste0(nm, ".png")), width = 2000, height = 1600, res = 200)
  plot(vars[[nm]], main = nm)
  dev.off()
}
png(file.path(out_pred, "aspect_sin.png"), width = 2000, height = 1600, res = 200); plot(aspect_sin_r, main="aspect_sin"); dev.off()
png(file.path(out_pred, "aspect_cos.png"), width = 2000, height = 1600, res = 200); plot(aspect_cos_r, main="aspect_cos"); dev.off()

cat("\nListo. CSVs y figuras guardados en:\n", outdir, "\n")
cat("Mapas de predictores guardados en:\n", out_pred, "\n")
