#' @title Flattens Functional Columns
#'
#' @section Parameters
#' * `as_features` :: `logical()`\cr
#'   Whether to add the Flattened values to the features of the task.
#' * `selector

#' * `selector` :: `function` | [`Selector`] \cr
#'   [`Selector`] function, takes a `Task` as argument and returns a `character`
#'   of features to keep. The flattening is only applied to those columns.\cr
#'   See [`Selector`] for example functions. Default is selector_type("functional")`.
#'   All features selected by this selector must be of type functional, otherwise an error is cast.
#' @export
PipeOpFlatFunct = R6::R6Class("PipeOpFlatFunct",
  inherit = mlr3pipelines::PipeOpTaskPreprocSimple,
  public = list(
    initialize = function(id = "flatfunct", param_vals = list()) {
      param_set = ps()

      input = data.table(
        name = "input", train = "Task", predict = "Task"
      )
      output = data.table(
        name = "output", train = "Task", predict = "Task"
      )
      super$initialize(
        id = id,
        param_set = ps(),
        param_vals = param_vals,
        packages = c("mlr3pipelines"),
        feature_types = "functional"
      )
    }
  ),
  private = list(
    .transform = function(task) {
      cols = self$state$dt_columns
      if (!length(cols)) {
        return(task)
      }
      data = task$data(cols = cols)

      d_flat = data.table(do.call("cbind", imap(data, function(x,nm) {
          x = do.call("rbind", map(x, 1))
          colnames(x) = paste0(nm, ".", seq_len(ncol(x)))
          return(x)
        })))

      task$select(setdiff(task$feature_names, cols))$cbind(d_flat)
    }
  )
)


    flatten_functional = function(x) {
      assert_class(x, "functional")
      args = unlist(map(x, "arg"))
      args = unique(args)
      args = sort(args)

      m = matrix(nrow = length(x), ncol = length(args))
      colnames(m) = args
      for (i in seq_along(x)) {
        idx = match(x[[i]]$arg, args)
        m[i, idx] = x[[i]]$value
      }
      return(m)
    }
    
mlr_pipeops = utils::getFromNamespace("mlr_pipeops", ns = "mlr3pipelines")
mlr_pipeops$add("flatfunct", PipeOpFlatFunct)
#' `private$.get_state` must not change its input value in-place and must return
#' something that will be written into `$state`
#' (which must not be NULL), `private$.transform()` should modify its argument in-place;
#' it is called both during training and prediction.
