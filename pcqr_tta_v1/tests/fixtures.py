from types import SimpleNamespace
import torch
from torch import nn


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(12, 4)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x, interpolate_pos_encoding=True):
        return SimpleNamespace(last_hidden_state=self.bn(self.linear(x.flatten(1)))[:, None])


class TinyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.bn = nn.BatchNorm1d(4)
        self.drop = nn.Dropout(0.5)

    def forward(self, emb, act):
        b, t, d = emb.shape
        v = self.linear(emb) + act + 0.03 * emb.mean(dim=1, keepdim=True)
        return self.drop(self.bn(v.reshape(-1, d))).reshape(b, t, d)


class TinyJEPA(nn.Module):
    """Test double with the verified repository's public method contract."""
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()
        self.projector = nn.Linear(4, 4)
        self.action_encoder = nn.Linear(10, 4)
        self.predictor = TinyPredictor()
        self.pred_proj = nn.Linear(4, 4)
        self.eval()

    def encode(self, info):
        p = info['pixels']
        b, t = p.shape[:2]
        e = self.encoder(p.reshape(b * t, *p.shape[2:])).last_hidden_state[:, 0]
        info['emb'] = self.projector(e).reshape(b, t, -1)
        return info

    def predict(self, emb, act):
        p = self.predictor(emb, act)
        return self.pred_proj(p)

    def rollout(self, info, actions, history_size=3):
        # Independent reference transcription of the upstream two-stage loop:
        # append future actions then perform one final prediction.
        b, s, t, a = actions.shape
        initial = info['pixels'].shape[2]
        act0, future = torch.split(actions, [initial, t - initial], dim=2)
        z = self.encode({'pixels': info['pixels'][:, 0]})['emb']
        z = z[:, None].expand(b, s, initial, -1).reshape(b * s, initial, -1).clone()
        act = act0.reshape(b * s, initial, a)
        future = future.reshape(b * s, t - initial, a)
        for h in range(t - initial):
            ae = self.action_encoder(act)
            nxt = self.predict(z[:, -history_size:], ae[:, -history_size:])[:, -1:]
            z = torch.cat([z, nxt], dim=1)
            act = torch.cat([act, future[:, h:h + 1]], dim=1)
        ae = self.action_encoder(act)
        z = torch.cat([z, self.predict(z[:, -history_size:], ae[:, -history_size:])[:, -1:]], dim=1)
        info['predicted_emb'] = z.reshape(b, s, -1, z.shape[-1])
        return info

    def get_cost(self, info, actions):
        pred = self.rollout(dict(info), actions)['predicted_emb']
        # Same terminal squared latent cost; a zero goal suffices in unit tests.
        return pred[:, :, -1].square().sum(-1)
