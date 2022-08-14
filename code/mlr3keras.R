source("code/list_to_args.R")

# Helper function that converts to a functional feature
as_functional = function(x) {
  x = as.matrix(x)
  xdim = dim(x)
  x = apply(x, 1, list)
  class(x) = c("functional", class(x))
  attr(x, "fun.dim") = xdim
  return(x)
}

functional_input_shape = function(x) {
  UseMethod("functional_input_shape")
}

# Helper function that gets input shape from task
functional_input_shape.Task = function(x) {
  x = x$data(cols = x$feature_names)
  functional_input_shape(x)
}

functional_input_shape.data.table = function(x) {
  x = map(x, attr, "fun.dim")
  inp_shape = c(max(map_int(x, 2)), length(x))
}

# Make mlr3 know that functional features exist
library(mlr3)
mlr_reflections$task_feature_types["fun"] = "functional"


# Architecture Builder function.
build_inceptionnet = function(task, pars) {
  output_directory = "output/inception/"
  if (!dir.exists(output_directory)) dir.create(output_directory)

  inp_shape = functional_input_shape(task)
  nclasses = length(levels(task$truth()))
  inceptionNet = import_from_path("inceptionnet", path = "code/")
  inceptionNet$Classifier_INCEPTION(
    output_directory = output_directory,
    input_shape = as.integer(inp_shape),
    nb_classes = as.integer(nclasses),
    verbose = TRUE,
    depth = as.integer(pars$depth),
    nb_filters = as.integer(pars$filters),
    batch_size = 128L,
    lr = pars$lr,
    nb_epochs = pars$epochs,
    monitor_metric='val_accuracy',
    callbacks=pars$callbacks
  )
}

# Architecture Builder function.
build_fcnet = function(task, pars) {
    output_directory = "output/fcn/"
    if (!dir.exists(output_directory)) dir.create(output_directory)

    inp_shape = functional_input_shape(task)
    nclasses = length(levels(task$truth()))
    fcn = import_from_path("fcn", path = "code/")

    print(pars$filters, pars$lr)
    fcn$Classifier_FCN(
      output_directory = output_directory,
      input_shape = as.integer(inp_shape),
      nb_classes = as.integer(nclasses),
      verbose = FALSE,
      build = TRUE,
      filters=pars$filters,
      lr = pars$lr,
      patience = pars$patience,
      monitor_metric='val_accuracy',
      callbacks=pars$callbacks
    )
}


# Architecture Object for Inception Net
KerasArchitectureInceptionNet = R6::R6Class("KerasArchitectureInceptionNet",
  inherit = KerasArchitecture,
  public = list(
    initialize = function(build_arch_fn = build_inceptionnet) {
      # Hyperpars for InceptionNet Architecture
      param_set =  ps()
      super$initialize(build_arch_fn = build_arch_fn, param_set = param_set,
        x_transform = function(features, pars) {
          inp_shape = functional_input_shape(features)
          nobs = nrow(features)
          features = map(features, function(ll) as.matrix(rbindlist(map(ll, function(x) as.list(unlist(x))))))
          arr = array(unlist(features), dim = c(nobs, inp_shape))
      })
    }
  )
)

# Architecture Object for FCN
KerasArchitectureFCN = R6::R6Class("KerasArchitectureFCN",
  inherit = KerasArchitecture,
  public = list(
    initialize = function(build_arch_fn = build_fcnet) {
      # Hyperpars for InceptionNet Architecture
      param_set =  ps()
      super$initialize(build_arch_fn = build_arch_fn, param_set = param_set,
        x_transform = function(features, pars) {
          inp_shape = functional_input_shape(features)
          nobs = nrow(features)
          features = map(features, function(ll) as.matrix(rbindlist(map(ll, function(x) as.list(unlist(x))))))
          arr = array(unlist(features), dim = c(nobs, inp_shape))
      })
    }
  )
)


# Learner, architecture can be exchanged.
LearnerClassifKerasFDAFCN = R6::R6Class("LearnerClassifKerasFDA", inherit = mlr3::LearnerClassif,
  public = list(
    architecture = NULL,
    initialize = function(
        id = "classif.kerasfda",
        predict_types = c("response", "prob"),
        feature_types = c("functional"),
        properties = c("twoclass", "multiclass"),
        packages = "keras",
        man = "mlr3keras::mlr_learners_classif.keras_fda",
        architecture = KerasArchitectureFCN$new()
      ) {
      self$architecture = assert_class(architecture, "KerasArchitecture")
      param_set = ps(
        validation_split = p_dbl(lower = 0, upper = 1, default = 0.2, tags = "train"),
        callbacks = p_uty(default = list(), tags = "train"),
        augmentation_ratio = p_int(lower = 0, upper = Inf, tags = "train"),
        jitter = p_lgl(tags = "train"),
        scaling = p_lgl(tags = "train"),
        permutation = p_lgl(tags = "train"),
        randompermutation = p_lgl(tags = "train"),
        magwarp = p_lgl(tags = "train"),
        timewarp = p_lgl(tags = "train"),
        windowslice = p_lgl(tags = "train"),
        windowwarp = p_lgl(tags = "train"),
        rotation = p_lgl(tags = "train"),
        spawner = p_lgl(tags = "train"),
        dtwwarp = p_lgl(tags = "train"),
        shapedtwwarp = p_lgl(tags = "train"),
        wdba = p_lgl(tags = "train"),
        discdtw = p_lgl(tags = "train"),
        discsdtw = p_lgl(tags = "train"),
        center = p_lgl(tags = c("train", "predict")),
        scale = p_lgl(tags = c("train", "predict")),
        batch_size = p_int(lower = 1, upper = Inf, tags = "train"),
        seed = p_int(tags = "train"),
        filters = p_int(lower = 1, upper = Inf, tags = "train"),
        lr = p_dbl(lower = 0, upper = Inf, tags = "train"),
        epochs = p_int(lower = 0, upper = Inf, tags = c("train", "budget")),
        patience = p_int(lower = 0, upper = Inf, tags = "train")
      )
      param_set$values = list(callbacks = list(), validation_split = 0, augmentation_ratio = 4L,
        scaling = FALSE, permutation = FALSE, randompermutation = FALSE, magwarp = FALSE, timewarp = FALSE,
        windowwarp = FALSE, rotation = FALSE, spawner = FALSE, dtwwarp = FALSE, shapedtwwarp = FALSE,
        wdba = FALSE, discdtw = FALSE, discsdtw = FALSE, windowslice = FALSE,  jitter = TRUE,
        center = TRUE, scale = TRUE, batch_size = 128L, seed = 444L, filters = 128L, lr = .00001, epochs = 1000L, patience=50
      )

      super$initialize(
        id = assert_character(id, len = 1L),
        param_set = param_set,
        predict_types = assert_character(predict_types),
        feature_types = assert_character(feature_types),
        properties = assert_character(properties),
        packages = assert_character(packages),
        man = assert_character(man)
      )

      # Set y_transform: use to_categorical, if goal is binary crossentropy drop 2nd column.
      self$architecture$set_transform("y",
        function(target, pars) {
          if (is.data.frame(target)) {
            target = unlist(target)
          }
          y = keras::to_categorical(as.integer(target) - 1, num_classes = length(levels(target)))
        }
      )
    },
    lr_find = function(task, epochs = 5L, lr_min = 10^-4, lr_max = 0.8, batch_size = 128L) {
      data = mlr3keras::find_lr(self$clone(), task, epochs, lr_min, lr_max, batch_size)
      plot_find_lr(data)
    }
  ),

  private = list(
    .train = function(task) {

      pars = self$param_set$get_values(tags = "train")
      # Set seed
      mlr3keras_set_seeds(pars$seed)

      # Construct / Get the model depending on task and hyperparams.
      model = self$architecture$get_model(task, pars)

      # Custom transformation depending on the model.
      # Could be generalized at some point.
      rows = sample(task$row_roles$use)
      features = task$data(cols = task$feature_names, rows = rows)
      target = task$data(cols = task$target_names, rows = rows)[[task$target_names]]
      # Transform Task to Array
      x = self$architecture$transforms$x(features, pars)

      idx = seq_len(dim(x)[1])
      train = unlist(map(split(idx, target), function(xc) {
        if (length(xc) == 1L) return(xc) # R sample() is STUPID!
        sample(xc, round((1 - pars$validation_split) * length(xc)))
      }))
      val = list(train = train, test = setdiff(idx, train))

      # Scale training data, estimate from training data.
      private$.get_scale_coefs(x[val$train,,])
      # Scale x and one-hot y
      x = private$.scale(x, pars)
      y = self$architecture$transforms$y(factor(target[val$train]), pars)
      
      # Augmentation
      res = private$.augment_data(x[val$train,,], y, pars)
      x_train_aug = res[[1]]
      y_train_aug = res[[2]]

      # Scale validation data
      if(length(val$test)==0  | pars$validation_split == 0){
        x_val <- y_val <- NULL
      }else{
        x_val = private$.scale(x[val$test,,], pars)
        y_val = self$architecture$transforms$y(target[val$test], pars)
      }

      history = model$fit(
        x_train = x_train_aug,
        y_train = y_train_aug,
        x_val = x_val,
        y_val = y_val,
        y_true = y_val,
        batch_size = pars$batch_size,
        nb_epochs = pars$epochs
      )

      return(list(model = model, history = history, class_names = task$class_names))
    },

    .predict = function(task) {
      pars = self$param_set$get_values(tags = "predict")
      features = task$data(cols = task$feature_names)
      newdata = self$architecture$transforms$x(features, pars)
      newdata = private$.scale(newdata, pars)
      pf_pars = self$param_set$get_values(tags = "predict_fun")
      if (inherits(self$model$model, "keras.engine.sequential.Sequential")) {
        p = invoke(keras::predict_proba, self$model$model, x = newdata, .args = pf_pars)
      } else {
        # The signature of the inception module is stupid here
        p = self$model$model$predict(newdata, NULL, NULL, NULL, NULL)
      }
      mlr3keras:::fixup_target_levels_prediction_classif(p, task, self$predict_type)
    },
    .augment_data = function(x, y, pars = list()) {
      pars = discard(pars[augment_args], is.null)
      pars = do.call("list_to_args", pars)
      aug = import_from_path("augmentation", "code/")
      res = aug$run_augmentation(x, y, pars)
    },
    .get_scale_coefs = function(x) {
      private$.scale_coefs = list(
        mean = apply(x, c(2, 3), mean),
        var = apply(x, c(2, 3), var)
      )
    },
    .scale = function(x, pars) {
      ndim = length(dim(x))
      if (ndim == 2L) {
        x = array(x, dim = c(1L, dim(x)))
      }
      for (i in seq_len(dim(x)[1])) {
        if (pars$center) {
          x[i,,] = x[i,,] - private$.scale_coefs$mean
        }
        if (pars$scale) {
          x[i,,] = x[i,,] / sqrt(private$.scale_coefs$var)
        }
      }

      return(x)
    },
    .scale_coefs = list(mean = 0, var = 1)
  )
)

# Learner, architecture can be exchanged.
LearnerClassifKerasFDAInception = R6::R6Class("LearnerClassifKerasFDA", inherit = mlr3::LearnerClassif,
  public = list(
    architecture = NULL,
    initialize = function(
        id = "classif.kerasfda",
        predict_types = c("response", "prob"),
        feature_types = c("functional"),
        properties = c("twoclass", "multiclass"),
        packages = "keras",
        man = "mlr3keras::mlr_learners_classif.keras_fda",
        architecture = KerasArchitectureInceptionNet$new()
      ) {
      self$architecture = assert_class(architecture, "KerasArchitecture")
      param_set = ps(
        validation_split = p_dbl(lower = 0, upper = 1, default = 0.2, tags = "train"),
        val_idx = p_uty(tags = "train"),
        callbacks = p_uty(default = list(), tags = "train"),
        augmentation_ratio = p_int(lower = 0, upper = Inf, tags = "train"),
        jitter = p_lgl(tags = "train"),
        scaling = p_lgl(tags = "train"),
        permutation = p_lgl(tags = "train"),
        randompermutation = p_lgl(tags = "train"),
        magwarp = p_lgl(tags = "train"),
        timewarp = p_lgl(tags = "train"),
        windowslice = p_lgl(tags = "train"),
        windowwarp = p_lgl(tags = "train"),
        rotation = p_lgl(tags = "train"),
        spawner = p_lgl(tags = "train"),
        dtwwarp = p_lgl(tags = "train"),
        shapedtwwarp = p_lgl(tags = "train"),
        wdba = p_lgl(tags = "train"),
        discdtw = p_lgl(tags = "train"),
        discsdtw = p_lgl(tags = "train"),
        center = p_lgl(tags = c("train", "predict")),
        scale = p_lgl(tags = c("train", "predict")),
        batch_size = p_int(lower = 1, upper = Inf, tags = "train"),
        seed = p_int(tags = "train"),
        filters = p_int(lower = 1, upper = Inf, tags = "train"),
        depth = p_int(lower = 1, upper = Inf, tags = "train"),
        lr = p_dbl(lower = 0, upper = Inf, tags = "train"),
        epochs = p_int(lower = 0, upper = Inf, tags = c("train", "budget")),
        use_residual = p_lgl(tags = "train"),
        use_bottleneck = p_lgl(tags = "train"),
        kernel_size = p_int(lower = 1, upper = Inf, tags = "train"),
        patience = p_int(lower = 0, upper = Inf, tags = "train"),
        initial_weights = p_uty(tags="train")
      )
      param_set$values = list(callbacks = list(), validation_split = 0, val_idx = NULL, augmentation_ratio = 4L,
        scaling = FALSE, permutation = FALSE, randompermutation = FALSE, magwarp = FALSE, timewarp = FALSE,
        windowwarp = FALSE, rotation = FALSE, spawner = FALSE, dtwwarp = FALSE, shapedtwwarp = FALSE,
        wdba = FALSE, discdtw = FALSE, discsdtw = FALSE, windowslice = FALSE,  jitter = TRUE,
        center = TRUE, scale = TRUE, batch_size = 128L, seed = 444L, depth = 3L, filters = 8L,
        lr = .00001, epochs = 1000L, use_residual = TRUE,
        use_bottleneck = TRUE, kernel_size = 41L, patience = 50L, initial_weights = NULL
      )

      super$initialize(
        id = assert_character(id, len = 1L),
        param_set = param_set,
        predict_types = assert_character(predict_types),
        feature_types = assert_character(feature_types),
        properties = assert_character(properties),
        packages = assert_character(packages),
        man = assert_character(man)
      )

      # Set y_transform: use to_categorical, if goal is binary crossentropy drop 2nd column.
      self$architecture$set_transform("y",
        function(target, pars) {
          if (is.data.frame(target)) {
            target = unlist(target)
          }
          y = keras::to_categorical(as.integer(target) - 1, num_classes = length(levels(target)))
        }
      )
    },
    lr_find = function(task, epochs = 5L, lr_min = 10^-4, lr_max = 0.8, batch_size = 128L) {
      data = mlr3keras::find_lr(self$clone(), task, epochs, lr_min, lr_max, batch_size)
      plot_find_lr(data)
    },
    save_weights = function(filepath) {
      assert_path_for_output(filepath)
      if (is.null(self$model)) stop("Model must be trained before saving")
      keras::save_model_hdf5(self$model$model$model, filepath)
    },
    load_model_from_file = function(filepath) {
      assert_file_exists(filepath)
      self$state$model$model$model = keras::load_model_hdf5(filepath)
    },
    transfer = function(task, row_ids = NULL) {
      pars = self$param_set$values

      if (is.null(self$state)) {
        # Construct / Get the model depending on task and hyperparams.
        base_model = self$architecture$get_model(task, pars)
        self$model = list(model = base_model, history = list(), class_names = character(0))
      }
      # Get trained model
      mmodel = self$model
      assert_list(mmodel)
      assert_names(names(mmodel), must.include = c('model', 'history', 'class_names'))

      if (!is.null(pars$initial_weights)) {
        print("Loading weights")
        self$load_model_from_file(pars$initial_weights)
      }

      base_model = mmodel$model
      base_model = freeze(base_model)

      if (is.null(row_ids)) {
        row_ids = task$row_ids
      }
      rows = sample(row_ids)
      features = task$data(cols = task$feature_names, rows = rows)
      target = task$data(cols = task$target_names, rows = rows)[[task$target_names]]
      x = self$architecture$transforms$x(features, pars)
      idx = seq_len(dim(x)[1])
      train = unlist(map(split(idx, target), function(xc) {
        if (length(xc) == 1L) return(xc) # R sample() is STUPID!
        sample(xc, round((1 - pars$validation_split) * length(xc)))
      }))
      val = list(train = train, test = setdiff(idx, train))

      # Scale training data, estimate from training data.
      private$.get_scale_coefs(x[val$train,,])
      x = private$.scale(x, pars)
      y = self$architecture$transforms$y(factor(target[val$train]), pars)
      
      # Augmentation
      res = private$.augment_data(x[val$train,,], y, pars)
      x_train_aug = res[[1]]
      y_train_aug = res[[2]]

      # Scale validation data
      if(length(val$test)==0  | pars$validation_split == 0){
        x_val <- y_val <- NULL
      }else{
        x_val = private$.scale(x[val$test,,], pars)
        y_val = self$architecture$transforms$y(target[val$test], pars)
      }

      # Reset last layer
      base_model$new_output_layer(ncol(y))

      history = base_model$fit(
        x_train = x_train_aug,
        y_train = y_train_aug,
        x_val = x_val,
        y_val = y_val,
        y_true = y_val,
        batch_size = pars$batch_size,
        nb_epochs = pars$epochs
      )
      self$model = list(model = base_model, history = history, class_names = task$class_names, transfered=TRUE)
    }
  ),

  private = list(
    .train = function(task) {

      pars = self$param_set$get_values(tags = "train")
      # Set seed
      mlr3keras_set_seeds(pars$seed)

      # Construct / Get the model depending on task and hyperparams.
      model = self$architecture$get_model(task, pars)

      # Custom transformation depending on the model.
      # Could be generalized at some point.
      rows = sample(task$row_roles$use)
      features = task$data(cols = task$feature_names, rows = rows)
      target = task$data(cols = task$target_names, rows = rows)[[task$target_names]]

      # Transform Task to Array
      x = self$architecture$transforms$x(features, pars)

      idx = seq_len(dim(x)[1])
      train = unlist(map(split(idx, target), function(xc) {
        if (length(xc) == 1L) return(xc) # R sample() is STUPID!
        sample(xc, round((1 - pars$validation_split) * length(xc)))
      }))
      val = list(train = train, test = setdiff(idx, train))
      if(!is.null(pars$val_idx)){
        val$train = setdiff(val$train, pars$val_idx)
        val$test = pars$val_idx
      }
      # Scale training data, estimate from training data.
      private$.get_scale_coefs(x[val$train,,, drop = FALSE])

      # Scale x and one-hot y
      x = private$.scale(x, pars)
      y = self$architecture$transforms$y(factor(target[val$train]), pars)

      # Augmentation
      res = private$.augment_data(x[val$train,,], y, pars)
      x_train_aug = res[[1]]
      y_train_aug = res[[2]]

      # Scale validation data
      if(length(val$test)==0 | (pars$validation_split == 0 & is.null(pars$val_idx))){
        x_val <- y_val <- NULL
      } else {
        x_val = private$.scale(x[val$test, , , drop = FALSE], pars)
        y_val = self$architecture$transforms$y(target[val$test], pars)
      }

      history = model$fit(
        x_train = x_train_aug,
        y_train = y_train_aug,
        x_val = x_val,
        y_val = y_val,
        y_true = y_val,
        batch_size = pars$batch_size,
        nb_epochs = pars$epochs
      )

      return(list(model = model, history = history, class_names = task$class_names))
    },

    .predict = function(task) {
      pars = self$param_set$get_values(tags = "predict")
      features = task$data(cols = task$feature_names)
      newdata = self$architecture$transforms$x(features, pars)
      newdata = private$.scale(newdata, pars)
      pf_pars = self$param_set$get_values(tags = "predict_fun")
      if (inherits(self$model$model, "keras.engine.sequential.Sequential")) {
        p = invoke(keras::predict_proba, self$model$model, x = newdata, .args = pf_pars)
      } else {
        # The signature of the inception module is stupid here
        p = self$model$model$predict(newdata, NULL, NULL, NULL, NULL)
      }
      mlr3keras:::fixup_target_levels_prediction_classif(p, task, self$predict_type)
    },
    .augment_data = function(x, y, pars = list()) {
      pars = discard(pars[augment_args], is.null)
      pars = do.call("list_to_args", pars)
      aug = import_from_path("augmentation", "code/")
      res = aug$run_augmentation(x, y, pars)
      return(res)
    },
    .get_scale_coefs = function(x) {
      private$.scale_coefs = list(
        mean = apply(x, c(2, 3), mean),
        var = apply(x, c(2, 3), var)
      )
    },
    .scale = function(x, pars) {
      ndim = length(dim(x))
      if (ndim == 2L) {
        x = array(x, dim = c(1L, dim(x)))
      }
      for (i in seq_len(dim(x)[1])) {
        if (pars$center) {
          x[i,,] = x[i,,] - private$.scale_coefs$mean
        }
        if (pars$scale) {
          x[i,,] = x[i,,] / sqrt(private$.scale_coefs$var)
        }
      }
      return(x)
    },
    .scale_coefs = list(mean = 0, var = 1)
  )
)


SamplerUnifwDefault = R6::R6Class("SamplerUnifwDefault", inherit = SamplerUnif,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(param_set, default) {
      assert_param_set(param_set, must_bounded = TRUE, no_deps = FALSE, no_untyped = TRUE)
      # We only opzimize for params in param_set and re-order
      default = default[, names(param_set$params), with = FALSE]
      private$.default = assert_data_table(default, max.rows = 1L)
      super$initialize(param_set)
    }
  ),
  private = list(
    .default = NULL,
    .n_defaults_sampled = 0L,
    .sample = function(n) {
      ndef = private$.n_defaults_sampled
      defs = private$.default
      if (ndef >= 1L) {
        vals = map_dtc(self$samplers, function(s) s$sample(n)$data)
      } else {
        if (n == 1L) {
          vals = defs
        } else {
          vals = rbindlist(list(
            defs,
            map_dtc(self$samplers, function(s) s$sample(n - 1)$data)
          ), use.names = TRUE)
        }
        private$.n_defaults_sampled = 1L
      }
      return(vals)
    }
  )
)

inception_default = data.table(
  lr = -9.2,  # log(1e-4)
  # epochs = 50,
  augmentation_ratio = 1,
  jitter = FALSE,
  scaling = FALSE,
  permutation = FALSE,
  randompermutation = FALSE,
  magwarp = FALSE,
  timewarp = FALSE,
  windowslice = FALSE,
  windowwarp = FALSE,
  spawner = FALSE,
  dtwwarp = FALSE,
  filters = 64,
  use_residual = TRUE,
  use_bottleneck = TRUE,
  kernel_size = 8
)

fcnet_default = data.table(
  lr = -9.2, # log(1e-4)
  # epochs = 50,
  augmentation_ratio = 1,
  jitter = FALSE,
  scaling = FALSE,
  permutation = FALSE,
  randompermutation = FALSE,
  magwarp = FALSE,
  timewarp = FALSE,
  windowslice = FALSE,
  windowwarp = FALSE,
  spawner = FALSE,
  dtwwarp = FALSE,
  filters = 64L
)

freeze = function(base_model) {
  for (layer in base_model$model$layers)
    layer$trainable = FALSE
  return(base_model)
}

unfreeze = function(base_model) {
  for (layer in base_model$model$layers)
    layer$trainable = TRUE
  return(base_model)
}


LearnerClassifKerasCNN = R6::R6Class("LearnerClassifKeras",
                                     inherit = mlr3keras::LearnerClassifKeras,
                                     public = list(
                                       initialize = function() {
                                         ps = ParamSet$new(list(
                                           ParamLgl$new("use_embedding", default = TRUE, tags = c("train", "predict")),
                                           ParamUty$new("optimizer", default = "optimizer_adam(lr=3*10^-4)", tags = "train"),
                                           ParamFct$new("loss", default = "categorical_crossentropy", tags = "train",  
                                                        levels = keras_reflections$loss$classif),
                                           ParamFct$new("output_activation", levels = c("softmax", "linear", "sigmoid"), 
                                                        tags = "train"),
                                           ParamUty$new("application", tags = "train"),
                                           ParamInt$new("cl_layer_units", tags = "train", default = 1024L),
                                           ParamInt$new("unfreeze_n_last_layers", tags = "train", default = 5L),
                                           ParamUty$new("metrics", tags = "train"),
                                           ParamDbl$new("validation_fraction", tags = "train", default = 0.2, lower = 0, upper = 1)
                                         ))
                                         ps$values = list(
                                           application = keras::application_vgg16,
                                           optimizer = optimizer_adam(lr = 3*10^-4),
                                           loss = "categorical_crossentropy",
                                           metrics = "accuracy",
                                           cl_layer_units = 1024L,
                                           output_activation = "softmax",
                                           unfreeze_n_last_layers = 5L,
                                           validation_fraction = 0.2,
                                           use_embedding = FALSE
                                         )
                                         arch = KerasArchitectureFF$new(build_arch_fn = build_keras_pretrained_cnn_model, 
                                                                        param_set = ps)
                                         super$initialize(
                                           feature_types = assert_character("functional"),
                                           man = "mlr3keras::mlr_learners_classif.keras",
                                           architecture = arch
                                         )
                                       }
                                     )
)

#' @title Keras Neural Network Feed Forward architecture
#' @rdname KerasArchitecture
#' @family KerasArchitectures
#' @export
KerasArchitectureResNet = R6::R6Class("KerasArchitectureFF",
                                      inherit = KerasArchitecture,
                                      public = list(
                                        initialize = function(build_arch_fn, param_set) {
                                          super$initialize(
                                            build_arch_fn = build_arch_fn, param_set = param_set,
                                            x_transform = function(features, pars) {
                                              inp_shape = functional_input_shape(features)
                                              nobs = nrow(features)
                                              features = map(features, function(ll) 
                                                as.matrix(rbindlist(map(ll, function(x) as.list(unlist(x))))))
                                              arr = array(unlist(features), dim = c(nobs, inp_shape[1], inp_shape[2]/3, 3))
                                              arr = imager::resize(arr, size_y = 101, size_z = 101, interpolation_type = 5) %>% 
                                                as.array()
                                            })
                                        }
                                      )
)



# Builds a Keras Feed Forward Neural Network
# @param task [`Task`] \cr
#   A mlr3 Task.
# @param pars [`list`] \cr
#   A list of parameter values from the Learner(Regr|Classif)KerasFF param_set.
# @template kerasff_description
# @return A compiled keras model
build_keras_pretrained_cnn_model = function(task, pars) {
  base = pars$application(
    include_top = FALSE,
    weights = "imagenet",
    input_tensor = NULL,
    input_shape = c(101,101,3),
    pooling = pars$pooling
  )
  model = keras_model(
    inputs = base$input,
    outputs = base$output %>%
      layer_global_average_pooling_2d() %>%
      layer_dense(units = pars$cl_layer_units, activation = "relu") %>%
      layer_dense(units = length(task$class_names), activation = pars$output_activation)
  )
  freeze_weights(base)
  # unfreeze_weights(base, from = length(base$layers) - pars$unfreeze_n_last_layers)
  model$compile(
    optimizer = pars$optimizer,
    loss = pars$loss,
    metrics = pars$metrics
  )
  return(model)
}
