required_packages <- c(
  "data.table", "dplyr", "future", "ggplot2", "mlr3",
  "mlr3extralearners", "mlr3pipelines", "mlr3proba", "recipes"
)
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop("Install required packages: ", paste(missing_packages, collapse = ", "))
}

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(ggplot2)
  library(mlr3)
  library(mlr3extralearners)
  library(mlr3pipelines)
  library(mlr3proba)
  library(recipes)
})

project_dir <- normalizePath(
  Sys.getenv("SAGITTAL_FD_PROJECT_DIR", unset = "."),
  mustWork = FALSE
)
processed_dir <- Sys.getenv(
  "SAGITTAL_FD_PROCESSED_DIR",
  unset = file.path(project_dir, "data", "processed")
)
output_dir <- Sys.getenv(
  "SAGITTAL_FD_OUTPUT_DIR",
  unset = file.path(project_dir, "results")
)

cohort_files <- c(
  training = Sys.getenv(
    "SAGITTAL_FD_TRAINING_DATA",
    unset = file.path(processed_dir, "training.csv")
  ),
  validation1 = Sys.getenv(
    "SAGITTAL_FD_VALIDATION1_DATA",
    unset = file.path(processed_dir, "validation1.csv")
  ),
  validation2 = Sys.getenv(
    "SAGITTAL_FD_VALIDATION2_DATA",
    unset = file.path(processed_dir, "validation2.csv")
  )
)

missing_files <- cohort_files[!file.exists(cohort_files)]
if (length(missing_files)) {
  stop(
    "Prepared cohort files are missing: ",
    paste(missing_files, collapse = ", "),
    ". Create them before running this analysis or set the SAGITTAL_FD_*_DATA variables."
  )
}

model_dir <- file.path(output_dir, "models")
prediction_dir <- file.path(output_dir, "predictions")
figure_dir <- file.path(output_dir, "figures")
dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(prediction_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

read_cohort <- function(path, cohort_name) {
  data <- as.data.frame(data.table::fread(path, data.table = FALSE))
  required <- c("OS", "status_os")
  missing <- setdiff(required, names(data))
  if (length(missing)) {
    stop(cohort_name, " is missing required columns: ", paste(missing, collapse = ", "))
  }
  data$OS <- as.numeric(data$OS)
  data$status_os <- as.integer(data$status_os)
  valid <- is.finite(data$OS) & data$OS > 0 & data$status_os %in% c(0L, 1L)
  if (!all(valid)) {
    stop(cohort_name, " contains invalid OS or status_os values.")
  }
  data
}

cohorts <- Map(read_cohort, cohort_files, names(cohort_files))
training_data <- cohorts$training

configured_predictors <- trimws(strsplit(
  Sys.getenv("SAGITTAL_FD_PREDICTORS", unset = ""), ",", fixed = TRUE
)[[1]])
configured_predictors <- configured_predictors[nzchar(configured_predictors)]
metadata_columns <- c("patient_id", "PatientsNum", "cohort", "group", "OS", "status_os")
predictors <- if (length(configured_predictors)) {
  configured_predictors
} else {
  setdiff(names(training_data), metadata_columns)
}
if (!length(predictors)) stop("No predictor columns were selected.")

required_columns <- c(predictors, "OS", "status_os")
for (cohort_name in names(cohorts)) {
  missing <- setdiff(required_columns, names(cohorts[[cohort_name]]))
  if (length(missing)) {
    stop(cohort_name, " is missing columns: ", paste(missing, collapse = ", "))
  }
  cohorts[[cohort_name]] <- cohorts[[cohort_name]][, required_columns, drop = FALSE]
}

reference_classes <- vapply(
  cohorts$training[predictors], function(x) class(x)[[1]], character(1)
)
for (cohort_name in names(cohorts)) {
  for (variable in predictors) {
    if (reference_classes[[variable]] %in% c("factor", "character")) {
      levels_reference <- unique(as.character(cohorts$training[[variable]]))
      cohorts[[cohort_name]][[variable]] <- factor(
        as.character(cohorts[[cohort_name]][[variable]]),
        levels = levels_reference
      )
    } else {
      cohorts[[cohort_name]][[variable]] <- as.numeric(cohorts[[cohort_name]][[variable]])
    }
  }
  if (anyNA(cohorts[[cohort_name]][predictors])) {
    stop(cohort_name, " contains missing or unseen predictor values.")
  }
}

seed <- as.integer(Sys.getenv("SAGITTAL_FD_SEED", unset = "123"))
timepoints <- as.numeric(strsplit(
  Sys.getenv("SAGITTAL_FD_TIMEPOINTS", unset = "12,24,36,48,60"),
  ",", fixed = TRUE
)[[1]])
if (any(!is.finite(timepoints)) || any(timepoints <= 0)) {
  stop("SAGITTAL_FD_TIMEPOINTS must contain positive numeric values.")
}

make_recipe <- function(method) {
  recipe_object <- recipes::recipe(OS + status_os ~ ., data = cohorts$training)
  if (method %in% c("dummy", "normalized")) {
    recipe_object <- recipes::step_dummy(
      recipe_object, recipes::all_nominal_predictors()
    )
  }
  if (identical(method, "normalized")) {
    recipe_object <- recipes::step_normalize(
      recipe_object, recipes::all_predictors()
    )
  }
  recipes::prep(recipe_object)
}

recipes_by_method <- list(
  plain = make_recipe("plain"),
  dummy = make_recipe("dummy"),
  normalized = make_recipe("normalized")
)

bake_cohort <- function(data, recipe_object) {
  baked <- recipes::bake(recipe_object, new_data = data)
  baked[, c("OS", "status_os", setdiff(names(baked), c("OS", "status_os"))), drop = FALSE]
}

make_task <- function(data, id) {
  mlr3proba::as_task_surv(
    data, id = id, time = "OS", event = "status_os", type = "right"
  )
}

with_distribution <- function(learner) {
  mlr3pipelines::as_learner(mlr3pipelines::ppl(
    "distrcompositor",
    learner = learner,
    estimator = "kaplan",
    form = "ph"
  ))
}

model_specs <- list(
  CoxPH = list(
    recipe = "plain",
    learner = mlr3::lrn("surv.coxph")
  ),
  LASSO = list(
    recipe = "dummy",
    learner = with_distribution(mlr3::lrn(
      "surv.cv_glmnet", type.measure = "C", s = "lambda.1se"
    ))
  ),
  Rpart = list(
    recipe = "plain",
    learner = with_distribution(mlr3::lrn(
      "surv.rpart", cp = 0.01, minbucket = 7L
    ))
  ),
  RSF = list(
    recipe = "plain",
    learner = mlr3::lrn(
      "surv.rfsrc", ntree = 500L, nodesize = 15L, predict_type = "distr"
    )
  ),
  GBM = list(
    recipe = "dummy",
    learner = with_distribution(mlr3::lrn(
      "surv.gbm", n.trees = 250L, interaction.depth = 2L,
      n.minobsinnode = 10L, shrinkage = 0.03,
      bag.fraction = 1, cv.folds = 0L, n.cores = 1L,
      keep.data = FALSE, verbose = FALSE
    ))
  ),
  SVM = list(
    recipe = "normalized",
    learner = with_distribution(mlr3::lrn(
      "surv.svm", type = "vanbelle1", diff.meth = "makediff1",
      kernel = "rbf_kernel", opt.meth = "ipop", gamma = 0.1
    ))
  ),
  XGBoost = list(
    recipe = "dummy",
    learner = with_distribution(mlr3::lrn(
      "surv.xgboost.cox", booster = "gbtree", nrounds = 300L,
      max_depth = 3L, learning_rate = 0.05
    ))
  )
)

extract_survival <- function(prediction, timepoints) {
  survival_function <- prediction$distr$survival
  if (!is.function(survival_function)) {
    stop("The learner did not return a survival distribution.")
  }
  probabilities <- vapply(timepoints, function(time) {
    as.numeric(survival_function(time))
  }, numeric(length(prediction$row_ids)))
  probabilities <- matrix(
    probabilities,
    nrow = length(prediction$row_ids),
    ncol = length(timepoints)
  )
  colnames(probabilities) <- paste0("survival_", timepoints)
  probabilities
}

future::plan(future::sequential)
model_results <- vector("list", length(model_specs))
names(model_results) <- names(model_specs)
metric_rows <- list()
prediction_rows <- list()

for (model_name in names(model_specs)) {
  spec <- model_specs[[model_name]]
  recipe_object <- recipes_by_method[[spec$recipe]]
  baked_cohorts <- lapply(cohorts, bake_cohort, recipe_object = recipe_object)
  tasks <- Map(make_task, baked_cohorts, names(baked_cohorts))

  learner <- spec$learner$clone(deep = TRUE)
  set.seed(seed)
  learner$train(tasks$training)

  predictions <- lapply(tasks, learner$predict)
  model_results[[model_name]] <- list(
    learner = learner,
    recipe = recipe_object
  )
  saveRDS(
    model_results[[model_name]],
    file.path(model_dir, paste0(tolower(model_name), ".rds")),
    compress = "xz"
  )

  for (cohort_name in names(predictions)) {
    prediction <- predictions[[cohort_name]]
    cindex <- unname(prediction$score(mlr3::msr("surv.cindex")))
    metric_rows[[length(metric_rows) + 1L]] <- data.frame(
      model = model_name,
      cohort = cohort_name,
      cindex = cindex
    )

    survival_probabilities <- tryCatch(
      extract_survival(prediction, timepoints),
      error = function(error) matrix(
        NA_real_, nrow = length(prediction$row_ids), ncol = length(timepoints),
        dimnames = list(NULL, paste0("survival_", timepoints))
      )
    )
    prediction_rows[[length(prediction_rows) + 1L]] <- cbind(
      data.frame(
        model = model_name,
        cohort = cohort_name,
        row_id = prediction$row_ids,
        risk_score = as.numeric(prediction$crank)
      ),
      as.data.frame(survival_probabilities, check.names = FALSE)
    )
  }
}

metrics <- dplyr::bind_rows(metric_rows)
predictions <- dplyr::bind_rows(prediction_rows)
data.table::fwrite(metrics, file.path(output_dir, "model_performance.csv"))
data.table::fwrite(predictions, file.path(prediction_dir, "survival_predictions.csv"))
saveRDS(model_results, file.path(model_dir, "all_models.rds"), compress = "xz")

performance_plot <- ggplot(metrics, aes(model, cindex, fill = cohort)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.68) +
  geom_hline(yintercept = 0.5, linetype = "dashed", colour = "#777777") +
  coord_cartesian(ylim = c(0.5, 1)) +
  labs(x = NULL, y = "Harrell's C-index", fill = "Cohort") +
  theme_classic(base_size = 11) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1))

ggsave(
  file.path(figure_dir, "model_cindex.pdf"),
  performance_plot, width = 8, height = 5
)

message("Analysis complete. Results: ", normalizePath(output_dir, mustWork = FALSE))

