"""Inference Script for ViViT (on a single batch)."""

import copy
import functools
from typing import Any, Callable, Dict, Tuple

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
import jax
import jax.numpy as jnp
import jax.profiler
import ml_collections
import numpy as np
from scenic import app
from scenic.projects.vivit import model as vivit_model
from scenic.dataset_lib import dataset_utils
from scenic.projects.vivit import evaluation_lib
from scenic.projects.vivit import train_utils as vivit_train_utils
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import pretrain_utils
from scenic.train_lib import train_utils


def train(
    *,
    rng: jnp.ndarray,
    config: ml_collections.ConfigDict,
    model_cls: Any,
    dataset: dataset_utils.Dataset,
    workdir: str,
    writer: metric_writers.MetricWriter,
) -> Tuple[train_utils.TrainState, Dict[str, Any], Dict[str, Any]]:
  """Main training loop lives in this function.

  Given the model class and dataset, it prepares the items needed to run the
  training, including the TrainState.

  Args:
    rng: Jax rng key.
    config: Configurations of the experiment.
    model_cls: Model class; A model has a flax_module, a loss_fn, and a
      metrics_fn associated with it.
    dataset: The dataset that has train_iter, eval_iter, meta_data, and
      optionally, test_iter.
    workdir: Directory for checkpointing.
    writer: CLU metrics writer instance.

  Returns:
    train_state that has the state of training (including current
      global_step, model_state, rng, and the optimizer), train_summary
      and eval_summary which are dict of metrics. These outputs are used for
      regression testing.
  """
  # Build the flax_model.
  model = model_cls(config, dataset.meta_data)
  
  # Initialize model.
  rng, init_rng = jax.random.split(rng)
  (params, model_state, num_trainable_params,
   gflops) = train_utils.initialize_model(
       model_def=model.flax_model,
       input_spec=[(dataset.meta_data['input_shape'],
                    dataset.meta_data.get('input_dtype', jnp.float32))],
       config=config,
       rngs=init_rng)

  # Create optimizer.
  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  optimizer = jax.jit(
      optimizers.get_optimizer(config).create, backend='cpu')(
          params)
  rng, train_rng = jax.random.split(rng)
  train_state = train_utils.TrainState(
      global_step=0,
      optimizer=optimizer,
      model_state=model_state,
      rng=train_rng,
      accum_train_time=0)
  start_step = train_state.global_step
  if config.checkpoint:
    train_state, start_step = train_utils.restore_checkpoint(
        workdir, train_state)

  if (start_step == 0  # Which means "no" checkpoint is restored!
      and config.get('init_from') is not None):
    restored_model_cfg = config.init_from.get('model_config')
    init_checkpoint_path = config.init_from.get('checkpoint_path')
    checkpoint_format = config.init_from.get('checkpoint_format', 'scenic')
    if checkpoint_format == 'scenic':
      restored_train_state = pretrain_utils.restore_pretrained_checkpoint(
          init_checkpoint_path, train_state, assert_exist=True)
    elif checkpoint_format == 'bigvision':
      restored_train_state = pretrain_utils.convert_bigvision_to_scenic_checkpoint(
          init_checkpoint_path, train_state)
      # Config dict in bigvision is not the same format as scenic.
      # Therefore, make sure config match the config of the loaded model!
      restored_model_cfg = copy.deepcopy(config)
      # The following is needed when the restored and target models used a
      # different classifier. As bigvision uses a different config dict, we have
      # to specify this manually.
      restored_model_cfg.model.classifier = config.init_from.get(
          'classifier_type', 'token')

    train_state = model.init_from_train_state(train_state, restored_train_state,
                                              restored_model_cfg)
    # Free unnecessary memory.
    del restored_train_state
  elif start_step == 0:
    logging.info('Training completely from scratch.'
                 'Not restoring from any checkpoint.')

  # Replicate the optimizer, state, and rng.
  train_state = jax_utils.replicate(train_state)
  del params  # Do not keep a copy of the initial params.

  # Get learning rate scheduler.
  learning_rate_fn = lr_schedules.get_learning_rate_fn(config)

  train_step_pmapped = jax.pmap(
      functools.partial(
          vivit_train_utils.train_step,
          flax_model=model.flax_model,
          learning_rate_fn=learning_rate_fn,
          loss_fn=model.loss_function,
          metrics_fn=model.get_metrics_fn('train'),
          config=config,
          debug=config.debug_train),
      axis_name='batch',
      # We can donate both buffers of train_state and train_batch.
      donate_argnums=(0, 1),
  )

  # Take a batch
  train_batch = next(dataset.train_iter)
  # Do a forward pass
  train_state, t_metrics, lr = train_step_pmapped(train_state, train_batch)
  # This will accumulate metrics in TPU memory up to the point that we log
  # them. This is no problem for small metrics but may be a problem for
  # large (e.g. segmentation) metrics. An alternative is to set
  # `log_summary_steps` to a small number, or to use
  # `train_utils.unreplicate_and_get` here instead of right before writing
  # summaries, but that means in each step, we have data transfer between
  # tpu and host, which might slow down the training.
      
  # Wait until computations are done before exiting.
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
  # Return the train and eval summary after last step for regression testing.
  return train_state, train_summary, eval_summary


def get_trainer(trainer_name: str) -> Callable[..., Any]:
  """Returns trainer given its name."""
  if trainer_name == 'vivit_trainer':
    return train
  raise ValueError(f'Unsupported trainer: {trainer_name}.')


def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter):
  """Main function for the ViViT project."""
  model_cls = vivit_model.get_model_cls(config.model_name)
  data_rng, rng = jax.random.split(rng)
  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)
  trainer = get_trainer(config.trainer_name)

  trainer(
      rng=rng,
      config=config,
      model_cls=model_cls,
      dataset=dataset,
      workdir=workdir,
      writer=writer)


if __name__ == '__main__':
  app.run(main=main)