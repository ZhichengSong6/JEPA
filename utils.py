import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer

class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")


def pusht_state_to_factor(state, world_size: float = 512.0):
    """Convert raw 7-D PushT state to the 6-D training factor target.

    Raw state layout:
        [pusher_x, pusher_y, block_x, block_y, block_theta,
         pusher_vx, pusher_vy]

    Output layout:
        [pusher_x_norm, pusher_y_norm,
         block_x_norm, block_y_norm,
         sin(theta), cos(theta)]

    Velocity is deliberately excluded: the LeWM image encoder processes one
    frame at a time, so instantaneous velocity is not identifiable from a
    single image.
    """
    state = state.float()
    if state.shape[-1] != 7:
        raise ValueError(f'Expected PushT state[..., 7], got {state.shape}.')

    pusher_xy = 2.0 * state[..., 0:2] / float(world_size) - 1.0
    block_xy = 2.0 * state[..., 2:4] / float(world_size) - 1.0
    theta = state[..., 4]
    theta_unit = torch.stack((torch.sin(theta), torch.cos(theta)), dim=-1)
    return torch.cat((pusher_xy, block_xy, theta_unit), dim=-1)


def get_pusht_factor_transform(
    source: str = 'state',
    target: str = 'factor',
    world_size: float = 512.0,
):
    """Create the raw-state -> factor transform.

    This transform must run *before* the official state z-score normalizer,
    otherwise sin/cos would be applied to a normalized angle instead of the
    physical angle in radians.
    """
    def factor_fn(x):
        return pusht_state_to_factor(x, world_size=world_size)

    return dt.transforms.WrapTorchTransform(
        factor_fn, source=source, target=target
    )


def get_pusht_factor_metric_scales(
    dataset,
    world_size: float = 512.0,
    quantile: float = 0.95,
    num_pairs: int = 200000,
    seed: int = 0,
):
    """Estimate fixed distance scales for the three PushT factors.

    The scales are the requested quantile of random pairwise distances in the
    same normalized factor coordinates used by the heads. This is a
    normalization statistic, not a tuned model hyperparameter.
    """
    if not (0.0 < quantile <= 1.0):
        raise ValueError('quantile must be in (0, 1].')
    if num_pairs <= 0:
        raise ValueError('num_pairs must be positive.')

    raw_state = np.asarray(dataset.get_col_data('state'))
    raw_state = raw_state.reshape(-1, raw_state.shape[-1])
    valid = np.isfinite(raw_state).all(axis=1)
    raw_state = raw_state[valid]
    if len(raw_state) < 2:
        raise ValueError('Need at least two valid PushT states for metric scales.')

    # Use the same factor convention as the torch training transform.
    pusher = 2.0 * raw_state[:, 0:2] / float(world_size) - 1.0
    block = 2.0 * raw_state[:, 2:4] / float(world_size) - 1.0
    theta = raw_state[:, 4]
    theta_unit = np.stack((np.sin(theta), np.cos(theta)), axis=-1)

    rng = np.random.default_rng(seed)
    count = min(int(num_pairs), max(len(raw_state), 2) * 10)
    first = rng.integers(0, len(raw_state), size=count)
    second = rng.integers(0, len(raw_state), size=count)
    same = first == second
    second[same] = (second[same] + 1) % len(raw_state)

    distances = {
        'pusher': np.linalg.norm(pusher[first] - pusher[second], axis=-1),
        'block': np.linalg.norm(block[first] - block[second], axis=-1),
        'theta': np.linalg.norm(theta_unit[first] - theta_unit[second], axis=-1),
    }
    return {
        name: max(float(np.quantile(value, quantile)), 1e-6)
        for name, value in distances.items()
    }
