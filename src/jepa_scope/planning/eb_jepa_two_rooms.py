"""An adapter that plans with eb_jepa's Two Rooms world model through the planning protocol (draft).

The model is the action-conditioned JEPA in ``examples/ac_video_jepa`` of
``facebookresearch/eb_jepa`` (checked at commit ``966e61e``): an Impala encoder whose output is a
512-dimensional latent, and a one-layer GRU predictor. The three operations map onto it as
follows.

- ``encode`` applies eb_jepa's Two Rooms ``Normalizer`` to a raw ``[B, 2, 65, 65]`` observation,
  runs the encoder on it as a one-frame clip and flattens ``[B, 512, 1, 1, 1]`` to ``[B, 512]``.
- ``predict`` runs one GRU step with the action as input and the latent as the hidden state, then
  the encoder's final LayerNorm, which the predictor shares. The normalised output is the next
  latent, and eb_jepa feeds it back as the next hidden state, so the latent carries everything
  the next step needs. This is one step of ``JEPA.unroll`` in autoregressive mode.
- ``energy`` is the mean squared difference over the 512 dimensions, which is eb_jepa's
  ``ReprTargetDistMPCObjective`` for one state. (eb_jepa's README writes this cost with an L2
  norm, but its code takes the mean of squares, and this adapter follows the code.)

eb_jepa is not a dependency of jepa-scope. It is imported when an adapter is built, and it must be
installed in the same environment (see ``docs/planning-protocol.md``).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

IMG_SIZE = 65
LATENT_DIM = 512
ACTION_DIM = 2

_INSTALL_HINT = (
    "The eb_jepa Two Rooms adapter needs eb_jepa (github.com/facebookresearch/eb_jepa, commit "
    "966e61e) installed in this environment. See 'Adapters' in docs/planning-protocol.md."
)


def _import_eb_jepa():
    try:
        from eb_jepa.architectures import ImpalaEncoder, RNNPredictor
        from eb_jepa.datasets.two_rooms.normalizer import Normalizer
    except ImportError as error:
        raise ImportError(_INSTALL_HINT) from error
    return ImpalaEncoder, RNNPredictor, Normalizer


def _read_model_config(folder: Path) -> dict:
    import yaml  # eb_jepa depends on PyYAML

    # eb_jepa's evaluation rewrites config.yaml; config.train.yaml, when present, is the
    # training-time copy. Their model sections are the same.
    for name in ("config.train.yaml", "config.yaml"):
        path = folder / name
        if path.exists():
            with path.open() as f:
                return yaml.safe_load(f)["model"]
    raise FileNotFoundError(f"No config.yaml in {folder}")


class _Modules(nn.Module):
    """The encoder and predictor, under the names they have inside eb_jepa's ``JEPA``."""

    def __init__(self, model_cfg: dict):
        super().__init__()
        impala_encoder, rnn_predictor, _ = _import_eb_jepa()
        dobs = model_cfg["dobs"]
        # As examples/ac_video_jepa/main.py builds them.
        self.encoder = impala_encoder(
            width=1,
            stack_sizes=(16, model_cfg["henc"], model_cfg["dstc"]),
            num_blocks=2,
            dropout_rate=None,
            layer_norm=False,
            input_channels=dobs,
            final_ln=True,
            mlp_output_dim=LATENT_DIM,
            input_shape=(dobs, IMG_SIZE, IMG_SIZE),
        )
        self.predictor = rnn_predictor(hidden_size=LATENT_DIM, final_ln=self.encoder.final_ln)


class EbJepaTwoRooms:
    """eb_jepa's Two Rooms world model behind the planning protocol.

    Build it with :meth:`from_run` for trained weights or :meth:`untrained` for a control.
    """

    state_shape = (LATENT_DIM,)
    action_dim = ACTION_DIM

    def __init__(self, modules: nn.Module, device: str | torch.device = "cpu"):
        _, _, normalizer = _import_eb_jepa()
        self.device = torch.device(device)
        self.modules = modules.to(self.device).eval()
        self.normalizer = normalizer()

    @classmethod
    def from_run(
        cls, folder: str | Path, checkpoint: str = "latest.pth.tar", device: str | torch.device = "cpu"
    ) -> EbJepaTwoRooms:
        """Load a run folder written by eb_jepa's ``examples/ac_video_jepa/main.py``.

        Only the encoder and predictor weights are read. The checkpoint is loaded with
        ``weights_only=True`` and strict key matching, so a file that is not a plain state dict,
        or a key layout that does not match, raises instead of leaving weights random.
        """
        folder = Path(folder)
        modules = _Modules(_read_model_config(folder))
        saved = torch.load(folder / checkpoint, weights_only=True, map_location="cpu")
        state = {}
        for key, value in saved["model_state_dict"].items():
            key = key.removeprefix("_orig_mod.")  # added by torch.compile
            if key.startswith(("encoder.", "predictor.")):
                state[key] = value
        modules.load_state_dict(state, strict=True)
        return cls(modules, device)

    @classmethod
    def untrained(cls, folder: str | Path, seed: int = 0, device: str | torch.device = "cpu") -> EbJepaTwoRooms:
        """The same architecture as the run in ``folder``, with freshly initialised weights."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            modules = _Modules(_read_model_config(Path(folder)))
        return cls(modules, device)

    @torch.no_grad()
    def encode(self, obs: Tensor) -> Tensor:
        """Encode raw observations ``[B, 2, 65, 65]`` (as ``DotWall`` renders them) into ``[B, 512]``."""
        obs = self.normalizer.normalize_state(obs.to(self.device, torch.float32))
        return self.modules.encoder(obs.unsqueeze(2)).flatten(1)

    @torch.no_grad()
    def predict(self, z: Tensor, a: Tensor) -> Tensor:
        """One GRU step: ``z`` ``[B, 512]`` and action ``a`` ``[B, 2]`` to the next latent ``[B, 512]``."""
        z = z.to(self.device, torch.float32)
        a = a.to(self.device, torch.float32)
        return self.modules.predictor(z.reshape(z.shape[0], LATENT_DIM, 1, 1, 1), a.unsqueeze(-1)).flatten(1)

    def energy(self, z: Tensor, z_goal: Tensor) -> Tensor:
        """Mean squared difference over the latent dimensions, ``[B]``."""
        return ((z - z_goal) ** 2).mean(dim=1)
