

import os, glob, math, time, json, copy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000

DEFAULT_CFG = {
    "paths": {"data": "/content/tse-project/DATA", "proj": "/content/drive/MyDrive/tse-project"},
    "audio": {"sr": 16000, "train_seconds": 4.0, "eval_seconds": 6.0,
              "min_utt_seconds": {"train": 4.0, "val": 6.0, "test": 6.0}},
    "splits": {"train": ["train-clean-100", "train-clean-360"], "val": ["dev-clean"], "test": ["test-clean"]},
    # DEMAND recordings are split IN TIME (no leakage): fraction of each file used per role
    "noise": {"frac": {"train": [0.0, 0.8], "val": [0.8, 0.9], "test": [0.9, 1.0]}},
    "mix": {
        "p_scenario": {"speech_noise": 0.45, "speech": 0.25, "noise": 0.25, "clean": 0.05},
        "sir_db": [-5.0, 5.0],          # target-to-interfering-speaker ratio
        "snr_db": [-5.0, 15.0],         # (target+interferer)-to-noise ratio
        "level_dbfs": [-35.0, -15.0],   # RMS level of the final mixture
        "p_partial_overlap": 0.5,       # probability that the interferer only covers part of the segment
        "overlap_range": [0.3, 1.0],    # fraction of the segment covered by the interferer when partial
        "peak_limit": 0.99,
    },
    "enroll": {
        "per_speaker": {"train": 10, "val": 6, "test": 6},
        "durations_s": [2, 3, 5, 8, 12],            # VOICED seconds (after VAD) of each enrollment
        "duration_probs": [0.15, 0.2, 0.3, 0.2, 0.15],
        "min_utt_seconds": 1.5,
        "vad": {"frame_ms": 25, "hop_ms": 10, "rel_db": 100.0, "abs_db": -55.0, "hang_frames": 10},
        "batch_size": 32,
    },
    "model": {"n_fft": 512, "hop": 256, "N": 128, "hidden": 128, "layers": 8, "emb_dim": 192, "spk_hidden": 256},
    "train": {"epochs": 30, "steps_per_epoch": 1000, "batch_size": 8, "lr": 5.0e-4, "min_lr": 1.0e-5,
              "warmup_steps": 2000, "weight_decay": 1.0e-2, "grad_clip": 5.0, "amp": True,
              "num_workers": 2, "w_sisdr": 1.0, "w_mrstft": 5.0, "seed": 0},
    "eval": {"n_val": 300, "n_test": 1200, "batch_size": 16, "seed_val": 1234, "seed_test": 4321},
}


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_cfg(path):
    """Load config.yaml and fill missing keys from DEFAULT_CFG."""
    with open(path) as f:
        user = yaml.safe_load(f)
    return _merge(DEFAULT_CFG, user)


def save_cfg(cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def proj_dirs(cfg):
    p = cfg["paths"]["proj"]
    d = {k: os.path.join(p, k) for k in ["manifests", "embeddings", "checkpoints", "logs", "samples"]}
    for v in d.values():
        os.makedirs(v, exist_ok=True)
    return d

def _info(args):
    root, rel = args
    i = sf.info(os.path.join(root, rel))
    return i.frames, i.samplerate, i.channels


def build_librispeech_manifest(data_root, splits, out_csv, threads=16):
    """Scan DATA/LibriSpeech/<split>/<spk>/<chapter>/*.flac -> CSV (relative paths)."""
    rows = []
    for split in splits:
        for p in sorted(glob.glob(os.path.join(data_root, "LibriSpeech", split, "*", "*", "*.flac"))):
            rel = os.path.relpath(p, data_root)
            parts = rel.split(os.sep)
            rows.append((split, int(parts[2]), int(parts[3]), os.path.basename(p)[:-5], rel))
    df = pd.DataFrame(rows, columns=["split", "speaker", "chapter", "utt_id", "path"])
    with ThreadPoolExecutor(threads) as ex:
        infos = list(ex.map(_info, [(data_root, r) for r in df.path]))
    df["frames"] = [i[0] for i in infos]
    df["sr"] = [i[1] for i in infos]
    df["dur"] = df.frames / df.sr
    assert (df.sr == SR).all(), "LibriSpeech must be 16 kHz"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def build_noise_manifest(data_root, out_csv, threads=16):
    """Scan DATA/DEMAND/**/*.wav -> CSV (environment, channel file, frames)."""
    files = sorted(glob.glob(os.path.join(data_root, "DEMAND", "**", "*.wav"), recursive=True))
    rows = []
    for p in files:
        rel = os.path.relpath(p, data_root)
        env = rel.split(os.sep)[1]
        rows.append((env, os.path.basename(p), rel))
    df = pd.DataFrame(rows, columns=["env", "file", "path"])
    with ThreadPoolExecutor(threads) as ex:
        infos = list(ex.map(_info, [(data_root, r) for r in df.path]))
    df["frames"] = [i[0] for i in infos]
    df["sr"] = [i[1] for i in infos]
    df["channels"] = [i[2] for i in infos]
    df["dur"] = df.frames / df.sr
    bad = df[(df.sr != SR) | (df.channels != 1)]
    if len(bad):
        print(f"WARNING: {len(bad)} noise files are not 16 kHz mono and will be dropped")
        df = df.drop(bad.index).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def role_df(lib_df, cfg, role):
    return lib_df[lib_df.split.isin(cfg["splits"][role])].reset_index(drop=True)


def energy_vad(x, sr=SR, frame_ms=25, hop_ms=10, rel_db=30.0, abs_db=-55.0, hang_frames=10):
    """Sample-level boolean speech mask from frame energy (relative to the 95th percentile + absolute floor)."""
    fl, hl = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    if len(x) < fl:
        return np.ones(len(x), bool)
    n = 1 + (len(x) - fl) // hl
    idx = np.arange(fl)[None, :] + hl * np.arange(n)[:, None]
    e = 10 * np.log10((x[idx] ** 2).mean(1) + 1e-12)
    thr = max(abs_db, np.percentile(e, 95) - rel_db)
    act = (e > thr).astype(np.float32)
    act = np.convolve(act, np.ones(2 * hang_frames + 1), mode="same") > 0
    mask = np.zeros(len(x), bool)
    m = np.repeat(act, hl)[: len(x)]
    mask[: len(m)] = m
    return mask


def build_enroll_audio(rel_paths, dur_s, data_root, vad_cfg):
    """Concatenate VAD-voiced audio of the listed utterances and keep exactly dur_s seconds of speech."""
    need = int(dur_s * SR)
    chunks, tot = [], 0
    for rp in rel_paths:
        x, _ = sf.read(os.path.join(data_root, rp), dtype="float32")
        v = x[energy_vad(x, **vad_cfg)]
        if len(v):
            chunks.append(v)
            tot += len(v)
        if tot >= need:
            break
    y = np.concatenate(chunks) if chunks else np.zeros(need, np.float32)
    if len(y) < need:  # extremely rare: tile
        y = np.tile(y, int(math.ceil(need / max(len(y), 1))))
    return y[:need].astype(np.float32)


def plan_enroll_entries(df, n_per_spk, durations, probs, min_utt_s, seed):
    """For every speaker draw n_per_spk recipes (utterance list + voiced duration)."""
    rng = np.random.default_rng(seed)
    probs = np.asarray(probs, float) / np.sum(probs)
    entries = []
    for spk, g in df[df.dur >= min_utt_s].groupby("speaker"):
        paths, durs = g.path.values, g.dur.values
        for _ in range(n_per_spk):
            d = float(rng.choice(durations, p=probs))
            order = rng.permutation(len(paths))
            chosen, acc = [], 0.0
            for j in order:
                chosen.append(paths[j])
                acc += durs[j] * 0.7           # conservative voiced fraction
                if acc >= d * 1.2:
                    break
            entries.append({"speaker": int(spk), "utts": ";".join(chosen), "dur_s": d})
    return pd.DataFrame(entries)


class ECAPAEmbedder:
    """Frozen SpeechBrain ECAPA-TDNN (192-d). Call with a list of 1-D float32 arrays @16 kHz."""

    def __init__(self, device="cuda", savedir="/content/pretrained_ecapa"):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception:
            from speechbrain.pretrained import EncoderClassifier
        self.device = device
        self.m = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=savedir,
                                                run_opts={"device": device})
        self.m.eval()

    @torch.no_grad()
    def __call__(self, wavs):
        L = max(len(w) for w in wavs)
        b = torch.zeros(len(wavs), L)
        lens = torch.zeros(len(wavs))
        for i, w in enumerate(wavs):
            b[i, : len(w)] = torch.from_numpy(w)
            lens[i] = len(w) / L
        e = self.m.encode_batch(b.to(self.device), lens.to(self.device))
        return e.squeeze(1).float().cpu().numpy()


def build_bank(df, role, cfg, embedder, seed=0, threads=4, progress=True):
    """Compute (and return) the enrollment bank: meta DataFrame + float32 [N, 192] embeddings."""
    ec = cfg["enroll"]
    data_root = cfg["paths"]["data"]
    meta = plan_enroll_entries(df, ec["per_speaker"][role], ec["durations_s"], ec["duration_probs"],
                               ec["min_utt_seconds"], seed)
    embs = []
    bs = ec["batch_size"]
    rng_iter = range(0, len(meta), bs)
    if progress:
        from tqdm.auto import tqdm
        rng_iter = tqdm(rng_iter, desc=f"enroll bank [{role}]")
    with ThreadPoolExecutor(threads) as ex:
        for i in rng_iter:
            sub = meta.iloc[i:i + bs]
            wavs = list(ex.map(lambda r: build_enroll_audio(r.utts.split(";"), r.dur_s, data_root, ec["vad"]),
                               sub.itertuples()))
            embs.append(embedder(wavs))
    return meta, np.concatenate(embs).astype(np.float32)


def save_bank(meta, emb, prefix):
    meta.to_csv(prefix + "_meta.csv", index=False)
    np.save(prefix + "_emb.npy", emb)


def load_bank(prefix):
    return pd.read_csv(prefix + "_meta.csv"), np.load(prefix + "_emb.npy")


def embed_enrollment(wav, embedder, cfg, max_s=20.0):
    """Embed a user-provided enrollment waveform: VAD first, so silence never reaches ECAPA."""
    wav = wav.astype(np.float32)
    v = wav[energy_vad(wav, **cfg["enroll"]["vad"])]
    if len(v) < 0.5 * SR:
        raise ValueError(f"Only {len(v)/SR:.2f}s of speech found in the enrollment clip - record a longer one.")
    if len(v) < 2.0 * SR:
        print(f"WARNING: only {len(v)/SR:.1f}s of speech in the enrollment clip; quality will drop below ~2s.")
    return embedder([v[: int(max_s * SR)]])[0]


# ----------------------------------------------------------------------------------------------
# Mixture generation (spec sampling is shared by training and by the fixed val/test manifests)
# ----------------------------------------------------------------------------------------------
SCEN_INTERF = ("speech_noise", "speech")
SCEN_NOISE = ("speech_noise", "noise")


class SceneSampler:
    def __init__(self, lib_df, noise_df, bank_meta, cfg, role, seg_seconds):
        self.cfg, self.role = cfg, role
        self.L = int(seg_seconds * SR)
        mc = cfg["mix"]
        self.mc = mc
        min_frames = int(max(cfg["audio"]["min_utt_seconds"][role], seg_seconds) * SR)
        bank_spk = set(bank_meta.speaker.unique())
        self.bank_rows = {s_: g.index.values for s_, g in bank_meta.groupby("speaker")}
        self.bank_sets = [set(u.split(";")) for u in bank_meta.utts]
        self.bank_dur = bank_meta.dur_s.values
        df = lib_df[(lib_df.frames >= min_frames) & lib_df.speaker.isin(bank_spk)]
        # a target utterance is only usable if at least one enrollment entry of that speaker does NOT contain it
        ok = [any(p_ not in self.bank_sets[c] for c in self.bank_rows[s_]) for p_, s_ in zip(df.path, df.speaker)]
        df = df[np.asarray(ok, bool)].reset_index(drop=True)
        self.paths, self.frames, self.spk = df.path.values, df.frames.values, df.speaker.values
        self.spk_rows = {s_: np.nonzero(self.spk == s_)[0] for s_ in np.unique(self.spk)}
        self.speakers = np.array(sorted(self.spk_rows))
        # noise regions
        a, b = cfg["noise"]["frac"][role]
        n = noise_df.copy()
        n["lo"] = (n.frames * a).astype(int)
        n["hi"] = (n.frames * b).astype(int)
        n = n[(n.hi - n.lo) >= self.L].reset_index(drop=True)
        self.noise_paths, self.noise_lo, self.noise_hi = n.path.values, n.lo.values, n.hi.values
        if len(n) == 0 and any(k in cfg["mix"]["p_scenario"] and cfg["mix"]["p_scenario"][k] > 0 for k in SCEN_NOISE):
            raise RuntimeError(f"No noise file has >= {seg_seconds}s inside its '{role}' time region; "
                               "check the noise manifest / noise.frac in config.yaml")
        names = list(mc["p_scenario"])
        p = np.array([mc["p_scenario"][k] for k in names], float)
        self.scen_names, self.scen_p = names, p / p.sum()

    def sample(self, rng):
        mc, L = self.mc, self.L
        scen = str(rng.choice(self.scen_names, p=self.scen_p))
        ts = int(rng.choice(self.speakers))
        ti = int(rng.choice(self.spk_rows[ts]))
        spec = {"scenario": scen, "tgt_spk": ts, "tgt_path": self.paths[ti],
                "tgt_start": int(rng.integers(0, self.frames[ti] - L + 1))}
        if scen in SCEN_INTERF:
            while True:
                isp = int(rng.choice(self.speakers))
                if isp != ts:
                    break
            ii = int(rng.choice(self.spk_rows[isp]))
            if rng.random() < mc["p_partial_overlap"]:
                ln = int(rng.uniform(*mc["overlap_range"]) * L)
            else:
                ln = L
            spec.update(int_spk=isp, int_path=self.paths[ii], int_len=ln,
                        int_start=int(rng.integers(0, self.frames[ii] - ln + 1)),
                        int_off=int(rng.integers(0, L - ln + 1)), sir=float(rng.uniform(*mc["sir_db"])))
        if scen in SCEN_NOISE:
            k = int(rng.integers(len(self.noise_paths)))
            spec.update(noise_path=self.noise_paths[k],
                        noise_start=int(rng.integers(self.noise_lo[k], self.noise_hi[k] - L + 1)),
                        snr=float(rng.uniform(*mc["snr_db"])))
        spec["level_db"] = float(rng.uniform(*mc["level_dbfs"]))
        ok = [c for c in self.bank_rows[ts] if spec["tgt_path"] not in self.bank_sets[c]]
        e = int(rng.choice(ok))
        spec["enroll_idx"], spec["enroll_dur"] = e, float(self.bank_dur[e])
        return spec


def _read(data_root, rel, start, n):
    x, _ = sf.read(os.path.join(data_root, rel), start=start, frames=n, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    return x


def _rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def render_scene(spec, data_root, L, peak_limit=0.99):
    """Deterministically turn a spec into (mixture, target, interferer, noise) float32 arrays."""
    scen = spec["scenario"]
    t = _read(data_root, spec["tgt_path"], int(spec["tgt_start"]), L)
    i = np.zeros(L, np.float32)
    n = np.zeros(L, np.float32)
    if scen in SCEN_INTERF:
        ln, off = int(spec["int_len"]), int(spec["int_off"])
        seg = _read(data_root, spec["int_path"], int(spec["int_start"]), ln)
        g = (_rms(t) / _rms(seg)) * 10 ** (-float(spec["sir"]) / 20)
        i[off:off + ln] = seg * g
    if scen in SCEN_NOISE:
        nz = _read(data_root, spec["noise_path"], int(spec["noise_start"]), L)
        ps = float(np.mean((t + i) ** 2))
        g = math.sqrt(ps / (max(float(np.mean(nz ** 2)), 1e-9) * 10 ** (float(spec["snr"]) / 10)))
        n = nz * g
    mix = t + i + n
    g = 10 ** (float(spec["level_db"]) / 20) / _rms(mix)
    pk = float(np.abs(mix).max()) * g
    if pk > peak_limit:
        g *= peak_limit / pk
    return (mix * g).astype(np.float32), (t * g).astype(np.float32), (i * g).astype(np.float32), (n * g).astype(np.float32)


class TSEDataset(torch.utils.data.Dataset):
    """On-the-fly training mixtures. One 'item' is a fresh random scene every call."""

    def __init__(self, sampler, bank_emb, n_items, data_root):
        self.s, self.bank, self.n, self.root = sampler, bank_emb, n_items, data_root

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rng = np.random.default_rng()
        spec = self.s.sample(rng)
        mix, tgt, _, _ = render_scene(spec, self.root, self.s.L, self.s.mc["peak_limit"])
        return torch.from_numpy(mix), torch.from_numpy(tgt), torch.from_numpy(self.bank[spec["enroll_idx"]]), idx


class FixedDataset(torch.utils.data.Dataset):
    """Fixed, reproducible mixtures defined by an eval manifest (val / test)."""

    def __init__(self, man_df, bank_emb, data_root, seg_seconds, peak_limit=0.99):
        self.df, self.bank, self.root = man_df.reset_index(drop=True), bank_emb, data_root
        self.L, self.pk = int(seg_seconds * SR), peak_limit

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        spec = self.df.iloc[idx].to_dict()
        mix, tgt, _, _ = render_scene(spec, self.root, self.L, self.pk)
        return torch.from_numpy(mix), torch.from_numpy(tgt), torch.from_numpy(self.bank[int(spec["enroll_idx"])]), idx


def build_eval_manifest(sampler, n, seed, out_csv=None):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame([sampler.sample(rng) for _ in range(n)])
    df.insert(0, "id", np.arange(n))
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


# ----------------------------------------------------------------------------------------------
# Model: speaker-conditioned Band-Split RNN
# ----------------------------------------------------------------------------------------------
def make_bands(n_fft, sr=SR, plan=((1000, 125), (4000, 250), (8000, 500))):
    """Band edges in STFT bins: 125 Hz bands below 1 kHz, 250 Hz up to 4 kHz, 500 Hz up to 8 kHz."""
    n_bins = n_fft // 2 + 1
    hz = sr / n_fft
    bands, cur = [], 0
    for hi_hz, w_hz in plan:
        hi, w = min(int(round(hi_hz / hz)), n_bins), max(1, int(round(w_hz / hz)))
        while cur < hi:
            end = min(cur + w, hi)
            bands.append([cur, end])
            cur = end
    if cur < n_bins:
        bands[-1][1] = n_bins
    return [tuple(b) for b in bands]


class BandSplit(nn.Module):
    def __init__(self, bands, N):
        super().__init__()
        self.bands = bands
        self.norms = nn.ModuleList([nn.LayerNorm(2 * (e - s)) for s, e in bands])
        self.fcs = nn.ModuleList([nn.Linear(2 * (e - s), N) for s, e in bands])

    def forward(self, spec):  # spec [B,F,T] complex -> [B,K,T,N]
        x = torch.view_as_real(spec)  # B,F,T,2
        B, _, T, _ = x.shape
        out = []
        for (s, e), ln, fc in zip(self.bands, self.norms, self.fcs):
            z = x[:, s:e].permute(0, 2, 1, 3).reshape(B, T, -1)
            out.append(fc(ln(z)))
        return torch.stack(out, 1)


class RNNBlock(nn.Module):
    def __init__(self, N, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(N)
        self.rnn = nn.LSTM(N, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * hidden, N)

    def forward(self, x):
        y, _ = self.rnn(self.norm(x))
        return x + self.proj(y)


class DualPathLayer(nn.Module):
    def __init__(self, N, hidden, spk_dim):
        super().__init__()
        self.time, self.band = RNNBlock(N, hidden), RNNBlock(N, hidden)
        self.film = nn.Linear(spk_dim, 2 * N)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, spk):  # x [B,K,T,N], spk [B,S]
        B, K, T, N = x.shape
        g, b = self.film(spk).chunk(2, -1)
        x = x * (1 + g[:, None, None]) + b[:, None, None]
        x = self.time(x.reshape(B * K, T, N)).reshape(B, K, T, N)
        x = x.permute(0, 2, 1, 3).reshape(B * T, K, N)
        x = self.band(x).reshape(B, T, K, N).permute(0, 2, 1, 3)
        return x


class MaskDecoder(nn.Module):
    def __init__(self, bands, N):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(N) for _ in bands])
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(N, 4 * N), nn.Tanh(), nn.Linear(4 * N, 4 * (e - s)), nn.GLU(dim=-1))
            for s, e in bands])
        self.bands = bands

    def forward(self, x):  # [B,K,T,N] -> complex-mask as real [B,F,T,2]
        outs = []
        for k, ((s, e), ln, mlp) in enumerate(zip(self.bands, self.norms, self.mlps)):
            m = mlp(ln(x[:, k]))                         # B,T,2*bw
            B, T, _ = m.shape
            outs.append(m.reshape(B, T, e - s, 2).permute(0, 2, 1, 3))
        return torch.cat(outs, 1)


class TSEBSRNN(nn.Module):
    def __init__(self, mcfg):
        super().__init__()
        self.n_fft, self.hop = mcfg["n_fft"], mcfg["hop"]
        N, H = mcfg["N"], mcfg["hidden"]
        self.bands = make_bands(self.n_fft)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        self.band_split = BandSplit(self.bands, N)
        S = mcfg["spk_hidden"]
        self.spk_enc = nn.Sequential(nn.Linear(mcfg["emb_dim"], S), nn.PReLU(), nn.Linear(S, S))
        self.spk_add = nn.Linear(S, N)
        self.layers = nn.ModuleList([DualPathLayer(N, H, S) for _ in range(mcfg["layers"])])
        self.decoder = MaskDecoder(self.bands, N)

    def forward(self, mix, emb):
        """mix [B,L] waveform, emb [B,192] ECAPA embedding -> estimated target waveform [B,L]."""
        L = mix.shape[-1]
        scale = mix.float().std(-1, keepdim=True) + 1e-5
        x = mix.float() / scale
        with torch.autocast(device_type=mix.device.type, enabled=False):
            spec = torch.stft(x, self.n_fft, self.hop, window=self.window, return_complex=True)
        spk = self.spk_enc(F.normalize(emb.float(), dim=-1))
        h = self.band_split(spec) + self.spk_add(spk)[:, None, None]
        for layer in self.layers:
            h = layer(h, spk)
        m = self.decoder(h)
        with torch.autocast(device_type=mix.device.type, enabled=False):
            est = spec * torch.view_as_complex(m.float().contiguous())
            wav = torch.istft(est, self.n_fft, self.hop, window=self.window, length=L)
        return wav * scale


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------------------------
# Losses / metrics
# ----------------------------------------------------------------------------------------------
def si_sdr(est, ref, eps=1e-7):
    est = est - est.mean(-1, keepdim=True)
    ref = ref - ref.mean(-1, keepdim=True)
    a = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    proj = a * ref
    return 10 * torch.log10(proj.pow(2).sum(-1) + eps) - 10 * torch.log10((est - proj).pow(2).sum(-1) + eps)


_WIN = {}


def mrstft_loss(est, ref, resolutions=((512, 128), (1024, 256), (2048, 512))):
    tot = 0.0
    for n_fft, hop in resolutions:
        key = (n_fft, est.device)
        if key not in _WIN:
            _WIN[key] = torch.hann_window(n_fft, device=est.device)
        w = _WIN[key]
        e = torch.stft(est, n_fft, hop, window=w, return_complex=True).abs()
        r = torch.stft(ref, n_fft, hop, window=w, return_complex=True).abs()
        sc = (r - e).norm(dim=(-2, -1)) / (r.norm(dim=(-2, -1)) + 1e-7)
        lm = F.l1_loss(torch.log(e + 1e-5), torch.log(r + 1e-5))
        tot = tot + sc.mean() + lm
    return tot / len(resolutions)


def tse_loss(est, ref, w_sisdr=1.0, w_mr=5.0):
    est, ref = est.float(), ref.float()
    s = si_sdr(est, ref)
    loss = -w_sisdr * s.clamp(max=30.0).mean() + w_mr * mrstft_loss(est, ref)
    return loss, s.detach()


# ----------------------------------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------------------------------
def _lr_at(step, tc, total):
    if step < tc["warmup_steps"]:
        return tc["lr"] * (step + 1) / tc["warmup_steps"]
    p = (step - tc["warmup_steps"]) / max(1, total - tc["warmup_steps"])
    return tc["min_lr"] + 0.5 * (tc["lr"] - tc["min_lr"]) * (1 + math.cos(math.pi * min(p, 1.0)))


@torch.no_grad()
def evaluate(model, ds, device, batch_size=16, amp=True, stoi=False, pesq_n=0, num_workers=2):
    """Returns per-item DataFrame: si_sdr_mix, si_sdr_est, si_sdri (+ stoi / pesq optional)."""
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    rows = []
    use_amp = amp and device.type == "cuda"
    for mix, tgt, emb, idx in dl:
        mix, tgt, emb = mix.to(device), tgt.to(device), emb.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            est = model(mix, emb)
        est = est.float()
        s_mix, s_est = si_sdr(mix, tgt), si_sdr(est, tgt)
        for j in range(len(idx)):
            r = {"idx": int(idx[j]), "si_sdr_mix": float(s_mix[j]), "si_sdr_est": float(s_est[j]),
                 "si_sdri": float(s_est[j] - s_mix[j])}
            if stoi or pesq_n:
                t_np, e_np = tgt[j].cpu().numpy(), est[j].cpu().numpy()
                if stoi:
                    try:
                        from pystoi import stoi as _stoi
                        r["stoi"] = float(_stoi(t_np, e_np, SR, extended=False))
                    except Exception:
                        pass
                if pesq_n and len(rows) < pesq_n:
                    try:
                        from pesq import pesq as _pesq
                        r["pesq"] = float(_pesq(SR, t_np, e_np, "wb"))
                    except Exception:
                        pass
            rows.append(r)
    return pd.DataFrame(rows)


def train(cfg, model, train_ds, val_ds, ckpt_dir, log_dir, device, tag="tse_bsrnn"):
    tc = cfg["train"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    use_amp = tc["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total = tc["epochs"] * tc["steps_per_epoch"]
    last, best = os.path.join(ckpt_dir, f"{tag}_last.pt"), os.path.join(ckpt_dir, f"{tag}_best.pt")
    log_csv = os.path.join(log_dir, f"{tag}_log.csv")
    start_ep, best_metric, log = 0, -1e9, []
    if os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_ep, best_metric, log = ck["epoch"] + 1, ck["best"], ck["log"]
        print(f"Resumed from epoch {start_ep} (best val SI-SDRi so far: {best_metric:.2f} dB)")
    from tqdm.auto import tqdm
    for ep in range(start_ep, tc["epochs"]):
        model.train()
        dl = torch.utils.data.DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=False,
                                         num_workers=tc["num_workers"], drop_last=True,
                                         persistent_workers=False, prefetch_factor=4 if tc["num_workers"] else None)
        t0, run_l, run_s, n_ok, n_skip = time.time(), 0.0, 0.0, 0, 0
        bar = tqdm(dl, total=tc["steps_per_epoch"], desc=f"epoch {ep+1}/{tc['epochs']}", leave=False)
        for it, (mix, tgt, emb, _) in enumerate(bar):
            step = ep * tc["steps_per_epoch"] + it
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, tc, total)
            mix, tgt, emb = mix.to(device, non_blocking=True), tgt.to(device), emb.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                est = model(mix, emb)
            loss, s = tse_loss(est, tgt, tc["w_sisdr"], tc["w_mrstft"])
            if not torch.isfinite(loss):
                n_skip += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            run_l += loss.item()
            run_s += s.mean().item()
            n_ok += 1
            if it % 20 == 0:
                bar.set_postfix(loss=f"{run_l/max(n_ok,1):.3f}", si_sdr=f"{run_s/max(n_ok,1):.2f}")
        v = evaluate(model, val_ds, device, cfg["eval"]["batch_size"], tc["amp"], num_workers=2)
        val_i = v[v_mask(val_ds)].si_sdri.mean() if len(v) else float("nan")
        row = {"epoch": ep + 1, "train_loss": run_l / max(n_ok, 1), "train_si_sdr": run_s / max(n_ok, 1),
               "val_si_sdr": v.si_sdr_est.mean(), "val_si_sdri": val_i, "lr": opt.param_groups[0]["lr"],
               "skipped": n_skip, "minutes": (time.time() - t0) / 60}
        log.append(row)
        pd.DataFrame(log).to_csv(log_csv, index=False)
        print(f"epoch {ep+1:02d} | loss {row['train_loss']:.3f} | train SI-SDR {row['train_si_sdr']:.2f} | "
              f"val SI-SDR {row['val_si_sdr']:.2f} | val SI-SDRi {val_i:.2f} | {row['minutes']:.1f} min")
        if val_i > best_metric:
            best_metric = val_i
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep, "val_si_sdri": val_i}, best)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": ep, "best": best_metric, "log": log, "cfg": cfg}, last)
    return pd.DataFrame(log)


def v_mask(ds):
    """Boolean mask of val items that count for SI-SDRi (all but the 'clean' scenario)."""
    return (ds.df.scenario != "clean").values


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = TSEBSRNN(ck["cfg"]["model"]).to(device)
    m.load_state_dict(ck["model"])
    return m.eval(), ck["cfg"]


def report(res, man):
    """Join metrics with manifest and print the standard breakdowns."""
    d = man.merge(res, left_index=True, right_on="idx")
    d = d[d.scenario != "clean"]
    out = {"overall": d[["si_sdr_mix", "si_sdr_est", "si_sdri"] + [c for c in ("stoi", "pesq") if c in d]].mean()}
    out["by_scenario"] = d.groupby("scenario")[["si_sdr_mix", "si_sdr_est", "si_sdri"]].mean()
    out["by_enroll_dur"] = d.groupby("enroll_dur")[["si_sdr_est", "si_sdri"]].mean()
    s = d[d.scenario.isin(SCEN_INTERF)].copy()
    s["sir_bin"] = pd.cut(s.sir, [-5, -2.5, 0, 2.5, 5])
    out["by_sir"] = s.groupby("sir_bin", observed=True)[["si_sdr_est", "si_sdri"]].mean()
    n = d[d.scenario.isin(SCEN_NOISE)].copy()
    n["snr_bin"] = pd.cut(n.snr, [-5, 0, 5, 10, 15])
    out["by_snr"] = n.groupby("snr_bin", observed=True)[["si_sdr_est", "si_sdri"]].mean()
    out["failures(si_sdri<0)"] = float((d.si_sdri < 0).mean())
    return out, d


# ----------------------------------------------------------------------------------------------
# Listening / inference
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def separate(model, mix, emb, device, chunk_s=30.0):
    """Run the model on an arbitrary-length 16 kHz mono waveform (np array) with a fixed enrollment embedding."""
    model.eval()
    e = torch.from_numpy(np.asarray(emb, np.float32))[None].to(device)
    n = int(chunk_s * SR)
    outs = []
    for s in range(0, len(mix), n):
        seg = torch.from_numpy(mix[s:s + n].astype(np.float32))[None].to(device)
        outs.append(model(seg, e)[0].float().cpu().numpy())
    return np.concatenate(outs)


def export_samples(model, man, bank_meta, bank_emb, cfg, out_dir, device, n=10, seed=0, show=True):
    """Write mix/target/estimate/enrollment wavs to out_dir and (optionally) show audio players."""
    os.makedirs(out_dir, exist_ok=True)
    data_root, L = cfg["paths"]["data"], int(cfg["audio"]["eval_seconds"] * SR)
    rng = np.random.default_rng(seed)
    man = man.reset_index(drop=True)
    pick = []
    for sc, g in man.groupby("scenario"):
        pick += list(rng.choice(g.index.values, size=min(len(g), max(1, n // 4)), replace=False))
    if show:
        from IPython.display import Audio, display, HTML
    for k in pick:
        spec = man.iloc[k].to_dict()
        mix, tgt, _, _ = render_scene(spec, data_root, L, cfg["mix"]["peak_limit"])
        emb = bank_emb[int(spec["enroll_idx"])]
        est = separate(model, mix, emb, device)
        er = bank_meta.iloc[int(spec["enroll_idx"])]
        enr = build_enroll_audio(er.utts.split(";"), er.dur_s, data_root, cfg["enroll"]["vad"])
        sm = float(si_sdr(torch.from_numpy(mix)[None], torch.from_numpy(tgt)[None]))
        se = float(si_sdr(torch.from_numpy(est)[None], torch.from_numpy(tgt)[None]))
        base = os.path.join(out_dir, f"{k:04d}_{spec['scenario']}")
        for nm, w in [("mix", mix), ("target", tgt), ("estimate", est), ("enroll", enr)]:
            sf.write(f"{base}_{nm}.wav", w, SR)
        if show:
            info = (f"<b>#{k} {spec['scenario']}</b> | SIR {spec.get('sir', float('nan')):.1f} dB | "
                    f"SNR {spec.get('snr', float('nan')):.1f} dB | enrollment {er.dur_s:.0f}s voiced | "
                    f"SI-SDR {sm:.1f} -> {se:.1f} dB")
            display(HTML(info))
            for nm, w in [("enrollment", enr), ("mixture", mix), ("estimate", est), ("target (reference)", tgt)]:
                print(nm)
                display(Audio(w, rate=SR))
    return pick
