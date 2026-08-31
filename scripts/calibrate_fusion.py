#!/usr/bin/env python3
"""Robust fusion calibration on data never used for checkpoint selection."""
import argparse
import hashlib
import io
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_manifest_row_wave
from src.metrics import compute_dacon_metrics
import script as submission

HEADS = ("file_fake", "voice_fake", "music_fake", "voice_present", "music_present")
CALIBRATION_SCRIPT_VERSION = "robust-folds-df-gate-panns16k-v4"
FEATURE_EXTRACTOR_VERSION = "canonical-inference-panns16k-voice-max-v4"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_metadata(split_paths):
    df_path = next((path for path in (
        ROOT / "model/df_arena/df_arena_1b_int8.ort",
        ROOT / "model/df_arena/df_arena_1b_int8.onnx") if path.exists()), None)
    panns_path = ROOT / "model/panns/Cnn14_16k_mAP=0.438.pth"
    return {
        "voice_checkpoint_sha256": sha256(ROOT / "model/best.pt"),
        "music_checkpoint_sha256": sha256(ROOT / "model/music_best.pt"),
        "df_model_sha256": sha256(df_path),
        "df_model_version": "pranjal-pravesh/df_arena_1b:int8-onnx-64600",
        "panns_sha256": sha256(panns_path),
        "split_csv_sha256": {path.name: sha256(path) for path in split_paths},
        "pipeline_version": submission.PIPELINE_VERSION,
        "calibration_script_version": CALIBRATION_SCRIPT_VERSION,
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
    }


def validate_cache_metadata(actual, expected):
    stale = {key: (actual.get(key), value) for key, value in expected.items()
             if actual.get(key) != value}
    if stale:
        raise RuntimeError(f"Stale calibration cache metadata: {stale}")
    return True


def load_git_model(ref, path, task, device):
    payload = subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT)
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu")
    backbone = str(checkpoint["backbone"])
    channels = int(checkpoint["base_channels"])
    model = (submission.AASISTMultitask(base_channels=channels) if backbone == "aasist"
             else submission.MusicMultitask(base_channels=channels))
    if checkpoint.get("task", task) not in (task, "multitask"):
        raise RuntimeError(f"Baseline {path} task mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def specialist_features(models, waves, device):
    voice, music = models
    groups = [submission.select_aux_segments(wave, sr=16000, seg_sec=4.0) for wave in waves]
    voice_out, voice_bounds = submission._run_torch_segments(voice, groups, device, use_amp=True)
    music_out, music_bounds = submission._run_torch_segments(music, groups, device, use_amp=True)
    features = []
    for index in range(len(waves)):
        va, vb = voice_bounds[index]
        ma, mb = music_bounds[index]
        features.append({
            "baseline_vf": submission.aggregate_predictions(voice_out["voice_fake"][va:vb], "topk_mean", 2),
            "baseline_mf": submission.aggregate_predictions(music_out["music_fake"][ma:mb], "topk_mean", 2),
            "baseline_vfile": submission.aggregate_predictions(voice_out["file_fake"][va:vb], "topk_mean", 2),
            "baseline_mfile": submission.aggregate_predictions(music_out["file_fake"][ma:mb], "topk_mean", 2),
            "baseline_vp_model": float(np.mean(voice_out["voice_present"][va:vb])),
            "baseline_mp_model": float(np.mean(music_out["music_present"][ma:mb])),
        })
    return features


def balanced_subset(df, limit, seed):
    if limit is None or int(limit) <= 0 or int(limit) >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    keys = list(HEADS)
    groups = list(df.groupby(keys, dropna=False))
    per_group = max(1, int(limit) // max(1, len(groups)))
    pieces = [group.sample(min(len(group), per_group), random_state=seed + index)
              for index, (_, group) in enumerate(groups)]
    sampled = pd.concat(pieces)
    remaining = df.drop(index=sampled.index, errors="ignore")
    if len(sampled) < min(int(limit), len(df)):
        sampled = pd.concat([sampled, remaining.sample(
            min(int(limit) - len(sampled), len(remaining)), random_state=seed + 999)])
    return sampled.sample(frac=1, random_state=seed).head(int(limit)).reset_index(drop=True)


def load_row_wave(row):
    return load_manifest_row_wave(row, sr=16000, is_training=False, use_demucs=False)


def collect(split_name, df, models, df_session, panns, device, batch_size,
            baseline_models=None, df_config=None):
    voice, music = models
    records = []
    for start in range(0, len(df), batch_size):
        rows = df.iloc[start:start + batch_size]
        waves = [load_row_wave(row) for _, row in rows.iterrows()]
        features = submission.infer_wave_features_batch(
            voice, music, df_session, panns, waves, device, use_demucs=False,
            df_config=(df_config if df_config is not None else
                       {"enabled": True, "force_second_for_long": True}))
        baseline = specialist_features(baseline_models, waves, device) if baseline_models else [{} for _ in waves]
        for (_, row), feature, baseline_feature in zip(rows.iterrows(), features, baseline):
            records.append({
                "split": split_name,
                "calibration_fold": str(row.get("calibration_fold", split_name)),
                "source": str(row.get("source", "unknown")),
                "generator": str(row.get("generator", "unknown")),
                "is_mixed": str(row["path"]).startswith("MIX::"),
                **{f"y_{head}": float(row[head]) for head in HEADS}, **feature, **baseline_feature,
            })
    return records


def adaptive_df(frame, config):
    primary = frame["df_primary"].to_numpy(float)
    second = frame["df_second"].to_numpy(float)
    if not config["enabled"]:
        return primary
    use = (frame["duration_sec"].to_numpy(float) >= config.get("minimum_duration",12.0))
    primary_uncertain=(primary>config["low"])&(primary<config["high"])
    if config.get("trigger_mode","primary")=="any_uncertain_disagreement":
        voice=frame["vf"].to_numpy(float); music=frame["mf"].to_numpy(float)
        component=((voice>config.get("component_low",.3))&(voice<config.get("component_high",.7)))
        component|=((music>config.get("component_low",.3))&(music<config.get("component_high",.7)))
        disagreement=np.ptp(np.column_stack([primary,voice,music]),axis=1)>=config.get("disagreement",.3)
        use &= primary_uncertain|component|disagreement
    else:
        use &= primary_uncertain
    use &= np.isfinite(second)
    if config["aggregation"] == "max":
        combined=np.maximum(primary,second)
    elif config["aggregation"] == "logit_mean":
        a=np.clip(primary,submission.OUTPUT_EPS,1-submission.OUTPUT_EPS)
        b=np.clip(second,submission.OUTPUT_EPS,1-submission.OUTPUT_EPS)
        combined=1/(1+np.exp(-(np.log(a/(1-a))+np.log(b/(1-b)))/2))
    else:
        combined=(primary+second)/2.0
    return np.where(use, combined, primary)


def df_execution_mask(frame, weights, prefix=""):
    """Return the samples that execute DF using only pre-DF specialist cues."""
    policy = str(weights.get("df_gate_policy", "off"))
    values = lambda name: frame[f"{prefix}{name}"].to_numpy(float)
    if policy in ("off", "none", "always", ""):
        return np.ones(len(frame), dtype=bool)
    voice_presence = values("vp_model")
    music_presence = values("mp_model")
    if policy == "voice_presence":
        return voice_presence >= float(weights.get("df_gate_voice_presence_threshold", 0.8))
    any_presence = np.maximum(voice_presence, music_presence)
    if policy == "any_presence":
        return any_presence >= float(weights.get("df_gate_presence_threshold", 0.8))
    low = float(weights.get("df_gate_uncertainty_low", 0.2))
    high = float(weights.get("df_gate_uncertainty_high", 0.8))
    specialist_probability = (
        0.5 * values("vfile") + 0.5 * values("mfile"))
    uncertain = (specialist_probability > low) & (specialist_probability < high)
    if policy == "specialist_uncertainty":
        return uncertain
    if policy == "presence_or_uncertainty":
        threshold = float(weights.get("df_gate_presence_threshold", 0.8))
        return (any_presence >= threshold) | uncertain
    raise ValueError(f"unsupported df_gate_policy={policy}")


def score_frame(frame, weights, adaptive, prefix=""):
    """Vectorized equivalent of the canonical scalar fusion function."""
    df_scores = adaptive_df(frame, adaptive)
    def values(name):
        return frame[f"{prefix}{name}"].to_numpy(float)
    execute_df = df_execution_mask(frame, weights, prefix)
    df_scores = np.where(execute_df, df_scores, 0.5)
    legacy = float(weights.get("w_df_component", 0.0))
    voice_component = np.where(execute_df, float(weights.get("w_df_voice_component", legacy)), 0.0)
    music_component = np.where(execute_df, float(weights.get("w_df_music_component", legacy)), 0.0)
    voice_fake = voice_component * df_scores + (1.0 - voice_component) * values("vf")
    music_fake = music_component * df_scores + (1.0 - music_component) * values("mf")
    panns_weight = float(weights.get("w_panns_presence", 0.6))
    voice_present = panns_weight * frame["vp_panns"].to_numpy(float) + (1 - panns_weight) * values("vp_model")
    music_present = panns_weight * frame["mp_panns"].to_numpy(float) + (1 - panns_weight) * values("mp_model")
    probability_or = 1.0 - (1.0 - voice_fake) * (1.0 - music_fake)
    mode = str(weights.get("file_fusion_mode", "legacy"))
    wv = float(weights.get("w_voice_file", 0.5))
    wm = float(weights.get("w_music_file", 0.3))
    wo = float(weights.get("w_prob_or", 0.2))
    if mode == "legacy":
        detector_fused = wv * values("vfile") + wm * values("mfile") + wo * probability_or
    else:
        voice_risk = voice_present * voice_fake
        music_risk = music_present * music_fake
        component_or = 1.0 - (1.0 - voice_risk) * (1.0 - music_risk)
        if mode == "presence_component_or":
            detector_fused = component_or
        elif mode == "presence_weighted":
            detector_fused = (wv * voice_present * values("vfile")
                              + wm * music_present * values("mfile")
                              + wo * component_or)
        else:
            raise ValueError(f"unsupported file_fusion_mode={mode}")
    w_df = float(weights.get("w_df_arena", 0.5))
    file_fake = w_df * df_scores + (1.0 - w_df) * detector_fused
    predicted = np.column_stack((file_fake, voice_fake, music_fake, voice_present, music_present))
    predicted = np.clip(predicted, submission.OUTPUT_EPS, 1.0 - submission.OUTPUT_EPS)
    y_true = {head: frame[f"y_{head}"].to_numpy() for head in HEADS}
    y_pred = {head: predicted[:, index] for index, head in enumerate(HEADS)}
    return compute_dacon_metrics(y_true, y_pred)


def robust_score(cache, weights, adaptive):
    by_fold = {fold: score_frame(frame, weights, adaptive)
               for fold, frame in cache.groupby("calibration_fold")}
    scores = np.asarray([metrics["score"] for metrics in by_fold.values()])
    objective = 0.7 * float(scores.mean()) + 0.3 * float(scores.min())
    return objective, by_fold


def select_df_gate(cache, weights, adaptive, maximum_objective_loss=0.01):
    """Choose the cheapest calibrated gate within a bounded score loss."""
    ungated_objective, ungated_metrics = robust_score(cache, weights, adaptive)
    candidates = []
    for threshold in (0.5, 0.6, 0.7, 0.8):
        trial = dict(weights)
        trial.update({"df_gate_policy": "voice_presence",
                      "df_gate_voice_presence_threshold": threshold})
        objective, metrics = robust_score(cache, trial, adaptive)
        fraction = float((cache["vp_model"].to_numpy(float) >= threshold).mean())
        if objective >= ungated_objective - maximum_objective_loss:
            candidates.append((fraction, -objective, trial, objective, metrics))
    if not candidates:
        return dict(weights), ungated_objective, ungated_metrics, 1.0
    fraction, _, selected, objective, metrics = min(candidates, key=lambda item: item[:2])
    selected["df_gate_calibration_fraction"] = fraction
    return selected, objective, metrics, fraction


def calibration_summary(frame):
    mixed = frame[frame["path"].astype(str).str.startswith("MIX::")]
    pairs = {f"{int(v)}{int(m)}": int(count)
             for (v, m), count in mixed.groupby(["voice_fake", "music_fake"]).size().items()}
    return {
        "samples": int(len(frame)),
        "file_real_fake": {str(k): int(v) for k, v in frame["file_fake"].value_counts().sort_index().items()},
        "mixed_rr_rf_fr_ff": pairs,
        "voice_conditional_real_fake": {str(k): int(v) for k, v in frame.loc[frame["voice_present"] == 1, "voice_fake"].value_counts().sort_index().items()},
        "music_conditional_real_fake": {str(k): int(v) for k, v in frame.loc[frame["music_present"] == 1, "music_fake"].value_counts().sort_index().items()},
    }


def music_breakdown(frame):
    from sklearn.metrics import roc_auc_score
    from src.metrics import compute_eer
    present = frame[frame["y_music_present"] == 1].copy()
    present["original_mixed"] = np.where(present["is_mixed"], "mixed", "original")
    present["music_generator"] = present["generator"].astype(str).map(
        lambda value: value.rsplit("::", 1)[-1] if value.startswith("mix::") else value)
    rows = []
    for mode in ("original", "mixed"):
        mode_frame = present[present["original_mixed"] == mode]
        for generator, target in mode_frame.groupby("music_generator"):
            labels = target["y_music_fake"].to_numpy()
            # A source/generator normally has one class. Pair it with all
            # opposite-class samples from the same original/mixed condition so
            # the reported EER/AUC is defined and comparable.
            opposite = mode_frame[mode_frame["y_music_fake"] != labels[0]]
            evaluated = pd.concat([target, opposite], ignore_index=True)
            y = evaluated["y_music_fake"].to_numpy()
            scores = evaluated["mf"].to_numpy()
            valid = len(np.unique(y)) == 2
            rows.append({"split": "FINAL_HOLDOUT", "source": "+".join(sorted(target["source"].unique())),
                         "generator": generator, "original_mixed": mode,
                         "real_count": int((y == 0).sum()), "fake_count": int((y == 1).sum()),
                         "eer": compute_eer(y, scores) if valid else np.nan,
                         "auc": roc_auc_score(y, scores) if valid else np.nan})
        y = mode_frame["y_music_fake"].to_numpy()
        scores = mode_frame["mf"].to_numpy()
        rows.append({"split":"FINAL_HOLDOUT","source":"ALL","generator":"ALL",
                     "original_mixed":mode,"real_count":int((y==0).sum()),
                     "fake_count":int((y==1).sum()),"eer":compute_eer(y,scores),
                     "auc":roc_auc_score(y,scores)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_split", type=int, default=0, help="0 uses the complete calibration split")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache", default="experiments/fusion_calibration_predictions.csv")
    parser.add_argument("--reuse_cache", action="store_true")
    parser.add_argument("--skip_final_holdout", action="store_true")
    parser.add_argument("--baseline_ref", default="1b5553200d08dcf4f7867e7ecfc8cc93a5d62d5f")
    parser.add_argument("--lock_df_always_on", action="store_true",
                        help="keep DF gate OFF and adaptive second crop OFF")
    args = parser.parse_args()
    device = torch.device(args.device)
    cache_path = pathlib.Path(args.cache)
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    split_path = ROOT / "data/splits/fusion_calibration.csv"
    expected_metadata = cache_metadata([split_path])
    expected_metadata["baseline_ref"] = args.baseline_ref
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    models = panns = df_session = baseline_models = None
    if args.reuse_cache and cache_path.exists() and metadata_path.exists():
        validate_cache_metadata(json.loads(metadata_path.read_text(encoding="utf-8")), expected_metadata)
        cache = pd.read_csv(cache_path)
        print(f"Loaded verified calibration cache: {cache_path} ({len(cache)})")
    else:
        models = (submission.load_voice_model(device), submission.load_music_model(device))
        baseline_models = (load_git_model(args.baseline_ref, "model/best.pt", "voice", device),
                           load_git_model(args.baseline_ref, "model/music_best.pt", "music", device))
        panns = submission.load_panns(device)
        df_session = submission.load_df_arena(device)
        frame = balanced_subset(pd.read_csv(split_path), args.per_split, 20260831)
        print("Calibration distribution:", json.dumps(calibration_summary(frame), indent=2))
        cache = pd.DataFrame(collect("fusion_calibration", frame, models, df_session, panns,
                                     device, args.batch_size, baseline_models))
        cache.to_csv(cache_path, index=False)
        metadata_path.write_text(json.dumps(expected_metadata, indent=2), encoding="utf-8")

    detector_weights = [(a / 4, b / 4, (4 - a - b) / 4)
                        for a in range(5) for b in range(5 - a)]
    adaptive_candidates = [{"enabled": False, "low": 0.0, "high": 1.0, "aggregation": "mean"}]
    adaptive_candidates += [{"enabled": True, "low": low, "high": high, "aggregation": aggregation}
                            for low, high in ((0.20, 0.80), (0.25, 0.75), (0.30, 0.70))
                            for aggregation in ("mean", "max")]
    adaptive_candidates.append({
        "enabled": True, "low": 0.2, "high": 0.8,
        "aggregation": "logit_mean", "minimum_duration": 10.0,
        "trigger_mode": "any_uncertain_disagreement",
        "component_low": 0.3, "component_high": 0.7, "disagreement": 0.3,
    })
    # A five-dimensional Cartesian grid both overfits 656 calibration rows and
    # repeats EER/AUC computation thousands of times. Coordinate search keeps
    # the same conservative candidate values while changing one decision at a
    # time. Two fixed passes are enough for convergence on this small grid.
    selected_weights = {
        "w_voice_file": 0.25, "w_music_file": 0.5, "w_prob_or": 0.25,
        "w_df_arena": 0.25, "w_df_voice_component": 0.3,
        "w_df_music_component": 0.0, "w_panns_presence": 0.75,
        "file_fusion_mode": "legacy",
    }
    coordinate_groups = (
        ("detector", detector_weights),
        ("w_df_arena", (0.0, 0.25, 0.5, 0.75, 1.0)),
        ("w_df_voice_component", (0.0, 0.05, 0.10, 0.20, 0.30)),
        ("w_df_music_component", (0.0, 0.025, 0.05, 0.10, 0.20)),
        ("w_panns_presence", (0.0, 0.25, 0.5, 0.75, 1.0)),
    )
    coordinate_trace = []
    for pass_index in range(2):
        for name, candidates in coordinate_groups:
            best_coordinate = None
            for value in candidates:
                trial = dict(selected_weights)
                if name == "detector":
                    trial.update(dict(zip(
                        ("w_voice_file", "w_music_file", "w_prob_or"), value)))
                    complexity = 0.0
                else:
                    trial[name] = value
                    complexity = float(value) if "component" in name else 0.0
                objective, metrics = robust_score(cache, trial, adaptive_candidates[0])
                candidate = (objective, -complexity, trial, metrics)
                if best_coordinate is None or candidate[:2] > best_coordinate[:2]:
                    best_coordinate = candidate
            objective, _, selected_weights, metrics = best_coordinate
            coordinate_trace.append({"pass": pass_index + 1, "coordinate": name,
                                     "objective": objective,
                                     "weights": dict(selected_weights)})
    # FILE-only fusion candidates. Component and presence calibration remains
    # fixed so this comparison cannot improve conditional EER by side effect.
    file_mode_results = {}
    best_file_mode = None
    for mode in ("legacy", "presence_weighted", "presence_component_or"):
        mode_best = None
        for wv, wm, wo in detector_weights:
            for wdf in (0.0, 0.25, 0.5, 0.75, 1.0):
                trial = dict(selected_weights)
                trial.update({"w_voice_file": wv, "w_music_file": wm,
                              "w_prob_or": wo, "w_df_arena": wdf,
                              "file_fusion_mode": mode})
                objective, metrics = robust_score(cache, trial, adaptive_candidates[0])
                candidate = (objective, -abs(wdf - 0.5), trial, metrics)
                if mode_best is None or candidate[:2] > mode_best[:2]:
                    mode_best = candidate
        mode_objective, _, mode_weights, mode_metrics = mode_best
        file_mode_results[mode] = {"robust_objective": mode_objective,
                                   "metrics_by_fold": mode_metrics,
                                   "weights": mode_weights}
        candidate = (mode_objective, mode_weights, mode_metrics)
        if best_file_mode is None or candidate[0] > best_file_mode[0]:
            best_file_mode = candidate
    _, selected_weights, _ = best_file_mode
    # Select the gate without second-crop effects, then select adaptive policy
    # on the gated canonical path. This keeps the two decisions identifiable.
    if args.lock_df_always_on:
        weights = dict(selected_weights)
        weights.update({"df_gate_policy": "off", "df_gate_calibration_fraction": 1.0})
        adaptive = adaptive_candidates[0]
        objective, metrics = robust_score(cache, weights, adaptive)
    else:
        weights, _, _, _ = select_df_gate(
            cache, selected_weights, adaptive_candidates[0], maximum_objective_loss=0.01)
        best_adaptive = None
        for adaptive_candidate in adaptive_candidates:
            candidate_objective, candidate_metrics = robust_score(cache, weights, adaptive_candidate)
            candidate = (candidate_objective, -int(adaptive_candidate["enabled"]),
                         adaptive_candidate, candidate_metrics)
            if best_adaptive is None or candidate[:2] > best_adaptive[:2]:
                best_adaptive = candidate
        objective, _, adaptive, metrics = best_adaptive
    calibrated_adaptive = adaptive
    weights.update({
        "metric_version": "dacon236749-official-noninterpolated-v1",
        "pipeline_version": submission.PIPELINE_VERSION,
        "voice_checkpoint_sha256": expected_metadata["voice_checkpoint_sha256"],
        "music_checkpoint_sha256": expected_metadata["music_checkpoint_sha256"],
        "panns_sha256": expected_metadata["panns_sha256"],
        "df_model_version": expected_metadata["df_model_version"],
        "df_arena_class_0_is_fake": True,
        "adaptive_df_enabled": adaptive["enabled"],
        "adaptive_df_low": adaptive["low"], "adaptive_df_high": adaptive["high"],
        "adaptive_df_aggregation": adaptive["aggregation"],
        "adaptive_df_minimum_duration": adaptive.get("minimum_duration", 12.0),
        "adaptive_df_trigger_mode": adaptive.get("trigger_mode", "primary"),
        "adaptive_df_component_low": adaptive.get("component_low", 0.3),
        "adaptive_df_component_high": adaptive.get("component_high", 0.7),
        "adaptive_df_disagreement": adaptive.get("disagreement", 0.3),
        "specialist_max_segments": 3,
        "panns_max_segments": 3,
        "voice_fake_aggregation": "max",
        "voice_segment_policy": "high_energy",
        "calibration_robust_objective": objective,
        "calibration_samples": int(len(cache)),
        "calibration_metrics_by_fold": metrics,
        "file_fusion_mode_comparison": file_mode_results,
        "coordinate_search_trace": coordinate_trace,
        "calibration_cache_metadata": expected_metadata,
    })
    out = ROOT / "model/fusion_weights.json"
    out.write_text(json.dumps(weights, indent=2), encoding="utf-8")
    print(json.dumps(weights, indent=2))

    if not args.skip_final_holdout:
        if models is None:
            models = (submission.load_voice_model(device), submission.load_music_model(device))
            baseline_models = (load_git_model(args.baseline_ref, "model/best.pt", "voice", device),
                               load_git_model(args.baseline_ref, "model/music_best.pt", "music", device))
            panns = submission.load_panns(device)
            df_session = submission.load_df_arena(device)
        final_frame = pd.read_csv(ROOT / "data/splits/final_holdout.csv")
        final_cache = pd.DataFrame(collect("final_holdout", final_frame, models, df_session, panns,
                                           device, args.batch_size, baseline_models))
        final_metrics = score_frame(final_cache, weights, adaptive)
        baseline_weights = {"w_voice_file":0.0,"w_music_file":0.5,"w_prob_or":0.5,
                            "w_df_arena":0.5,"w_df_voice_component":0.25,
                            "w_df_music_component":0.25,"w_panns_presence":0.75}
        ablations = {
            "A_current_baseline": score_frame(final_cache, baseline_weights, adaptive_candidates[0], "baseline_"),
            "B_mixed_waveform_training": score_frame(final_cache, baseline_weights, adaptive_candidates[0]),
            "C_separate_df_weights": score_frame(final_cache, selected_weights, adaptive_candidates[0]),
            "D_adaptive_df_second_crop": score_frame(final_cache, selected_weights, calibrated_adaptive),
            "E_speed_selected_df_gate": final_metrics,
        }
        final_cache.to_csv(ROOT / "experiments/final_holdout_predictions.csv", index=False)
        music_breakdown(final_cache).to_csv(ROOT / "experiments/music_validation_breakdown.csv", index=False)
        (ROOT / "experiments/final_holdout_ablation.json").write_text(
            json.dumps(ablations, indent=2), encoding="utf-8")
        print("FINAL_HOLDOUT ablation:", json.dumps(ablations, indent=2))
        print("FINAL_HOLDOUT (one-shot):", json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
