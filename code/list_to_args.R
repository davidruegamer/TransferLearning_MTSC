list_to_args <- function(
  augmentation_ratio = 4L,
  seed=2L,
  jitter=TRUE,
  scaling=FALSE,
  permutation=FALSE,
  randompermutation=FALSE,
  magwarp=FALSE,
  timewarp=FALSE,
  windowslice=FALSE,
  windowwarp=FALSE,
  rotation=FALSE,
  spawner=FALSE,
  dtwwarp=FALSE,
  shapedtwwarp=FALSE,
  wdba=FALSE,
  discdtw=FALSE,
  discsdtw=FALSE,
  extra_tag="",
  dataset="test"
  )
{
  
  argparse <- import("argparse")
  parser = argparse$ArgumentParser(description='Runs augmentation model.')
  parser$add_argument('--augmentation_ratio', type=integer, default=augmentation_ratio, 
                      help="How many times to augment")
  parser$add_argument('--seed', default=seed, help="Randomization seed")
  parser$add_argument('--jitter', default=jitter, action="store_true", 
                      help="Jitter preset augmentation")
  parser$add_argument('--scaling', default=scaling, action="store_true", 
                      help="Scaling preset augmentation")
  parser$add_argument('--permutation', default=permutation, action="store_true", 
                      help="Equal Length Permutation preset augmentation")
  parser$add_argument('--randompermutation', default=randompermutation, action="store_true", 
                      help="Random Length Permutation preset augmentation")
  parser$add_argument('--magwarp', default=magwarp, action="store_true", 
                      help="Magnitude warp preset augmentation")
  parser$add_argument('--timewarp', default=timewarp, action="store_true", 
                      help="Time warp preset augmentation")
  parser$add_argument('--windowslice', default=windowslice, action="store_true", 
                      help="Window slice preset augmentation")
  parser$add_argument('--windowwarp', default=windowwarp, action="store_true", 
                      help="Window warp preset augmentation")
  parser$add_argument('--rotation', default=rotation, action="store_true", 
                      help="Rotation preset augmentation")
  parser$add_argument('--spawner', default=spawner, action="store_true", 
                      help="SPAWNER preset augmentation")
  parser$add_argument('--dtwwarp', default=dtwwarp, action="store_true", 
                      help="DTW warp preset augmentation")
  parser$add_argument('--shapedtwwarp', default=shapedtwwarp, action="store_true", 
                      help="Shape DTW warp preset augmentation")
  parser$add_argument('--wdba', default=wdba, action="store_true", 
                      help="Weighted DBA preset augmentation")
  parser$add_argument('--discdtw', default=discdtw, action="store_true", 
                      help="Discrimitive DTW warp preset augmentation")
  parser$add_argument('--discsdtw', default=discsdtw, action="store_true", 
                      help="Discrimitive shapeDTW warp preset augmentation")
  parser$add_argument('--extra_tag', type=str, default=extra_tag, help="Anything extra")
  parser$add_argument('--dataset', type=str, default=dataset, help="blub")
  args = parser$parse_args()
  
  invisible(return(args))
  
}
