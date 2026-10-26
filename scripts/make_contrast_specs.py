#!/usr/bin/env python
"""Writes the pre-registered contrast specs (scripts/contrast_boot.py) and stats_all manifests of E38b / E39 / E40 /
E41 (docs/EXPERIMENTS.md, 2026-10-26). Run names follow scripts/slurm/{e21_dp,e23_smolvla,e12_wam,e41_camera}.sbatch.

  python scripts/make_contrast_specs.py   # -> configs/e39_contrasts.json, ...
"""

import json
from pathlib import Path

C = Path(__file__).resolve().parents[1] / "configs"
KS = ["k1", "k2", "k5", "k10", "k25", "kall"]
SEEDS = [0, 1, 2]


def dp(arm, kn, s, tag=""):
    return f"e21_dp_{arm}_{kn}{tag}_t17_s{s}"


def smolvla(arm, kn, s, tag=""):  # main-table k5 seeds 0 / 1 are the Jubail re-runs (TAG=_jr), seed 2 ran on Jubail
    if kn == "k5" and s < 2 and not tag:
        tag = "_jr"
    return f"e23_smolvla_{arm}_{kn}{tag}_t17_s{s}"


def term(fn, arm, kn, w, tag="", file="rollouts_n50.json", seeds=SEEDS):
    return {"runs": [fn(arm, kn, s, tag) for s in seeds], "w": w, "file": file}


def gain(fn, kn, w=1.0, a="expert_droid", b="none", tag=""):
    return [term(fn, a, kn, w, tag), term(fn, b, kn, -w, tag)]


HOSTS39 = {"dp": dp, "smolvla": smolvla}


def e39():
    cs = []
    for h, fn in HOSTS39.items():
        for kn in KS:
            cs.append({"name": f"gain {h} {kn}", "groups": {h: {"terms": gain(fn, kn)}}})
        for kn in ("k1", "kall"):
            cs.append({"name": f"pretraining {h} {kn} (droid - scratch)",
                       "groups": {h: {"terms": gain(fn, kn, b="expert_scratch")}}})
    low = {h: {"weight": 0.5, "terms": gain(fn, "k1", 0.5) + gain(fn, "k2", 0.5)} for h, fn in HOSTS39.items()}
    trend = {h: {"weight": 0.5, "terms": sum((gain(fn, k, 1 / 3) for k in ("k1", "k2", "k5")), [])
                 + sum((gain(fn, k, -1 / 2) for k in ("k25", "kall")), [])} for h, fn in HOSTS39.items()}
    cs = [{"name": "E39-P1 low-data gain (k1, k2; mean of DP, SmolVLA)", "groups": low},
          {"name": "E39-P2 gain decline (mean k1,k2,k5 - mean k25,all; mean of hosts)", "groups": trend}] + cs
    for h in HOSTS39:  # per-host versions of the primaries (secondary)
        cs.append({"name": f"low-data gain {h}", "groups": {h: {**low[h], "weight": 1.0}}})
        cs.append({"name": f"gain decline {h}", "groups": {h: {**trend[h], "weight": 1.0}}})
    return {"name": "E39 demo-count curve (17 tasks; seeds 0-2)", "contrasts": cs,
            "primary": [cs[0]["name"], cs[1]["name"]]}


def e40():
    g_base = gain(dp, "k5")
    g_hpre = gain(dp, "k5", tag="_hpre")
    inter = g_base + [{**t, "w": -t["w"]} for t in g_hpre]
    cs = [{"name": "E40 interaction: gain(host from scratch) - gain(host pretrained on train-split tasks)",
           "groups": {"dp": {"terms": inter}}},
          {"name": "gain with pretrained host (droid - none, _hpre)", "groups": {"dp": {"terms": g_hpre}}},
          {"name": "gain with scratch host (E21 main table)", "groups": {"dp": {"terms": g_base}}},
          {"name": "host pretraining effect on none (none_hpre - none)",
           "groups": {"dp": {"terms": [term(dp, "none", "k5", 1, "_hpre"), term(dp, "none", "k5", -1)]}}},
          {"name": "droid_hpre - droid (scratch host)",
           "groups": {"dp": {"terms": [term(dp, "expert_droid", "k5", 1, "_hpre"), term(dp, "expert_droid", "k5", -1)]}}}]
    return {"name": "E40 host action pretraining x SOPT (Diffusion Policy, 17 tasks k5, seeds 0-2)", "contrasts": cs,
            "primary": [cs[0]["name"]]}


E41 = {"dp": ("e21_dp_none_k5_t17_s{}", "e21_dp_expert_droid_k5_t17_s{}"),
       "smolvla_goal": ("e13_libero_goal_none_k5_s{}", "e13_libero_goal_expert_droid_k5_s{}"),
       "groot": ("e22_groot_none_k5_t17_s{}", "e31_groot_xattn_droid_k5_t17_s{}")}
COND = {"clean": "rollouts_n50.json", "10": "e41_dose_camera1.json", "20": "e37_main_camera2.json",
        "30": "e41_dose_camera3.json"}


def e41():
    def t(pat, file, w):
        return {"runs": [pat.format(s) for s in (0, 1)], "w": w, "file": file}

    def inter(h, lvl, w=1.0):  # (sopt_L - none_L) - (sopt_clean - none_clean)
        n, s = E41[h]
        return [t(s, COND[lvl], w), t(n, COND[lvl], -w), t(s, COND["clean"], -w), t(n, COND["clean"], w)]

    def mask_cost(h, w=1.0):  # (sopt - sopt_masked) at camera 20 - the same at clean
        s = E41[h][1]
        return [t(s, COND["20"], w), t(s, "e41_mask_camera2.json", -w), t(s, COND["clean"], -w),
                t(s, "e41_mask_clean.json", w)]

    w3 = 1 / 3
    cs = [{"name": "E41-P1 dose: interaction(30 deg) - interaction(10 deg), mean of 3 hosts",
           "groups": {h: {"weight": w3, "terms": inter(h, "30") + inter(h, "10", -1.0)} for h in E41}},
          {"name": "E41-P2 mask: SOPT-off cost at camera 20 deg - at clean, mean of DP, GR00T",
           "groups": {h: {"weight": 0.5, "terms": mask_cost(h)} for h in ("dp", "groot")}}]
    for h in E41:
        for lvl in ("10", "20", "30"):
            cs.append({"name": f"interaction {h} {lvl} deg", "groups": {h: {"terms": inter(h, lvl)}}})
        cs.append({"name": f"dose {h} (30 - 10 deg)", "groups": {h: {"terms": inter(h, "30") + inter(h, "10", -1.0)}}})
    for h in ("dp", "groot"):
        s = E41[h][1]
        cs.append({"name": f"mask cost {h} clean (sopt - masked)",
                   "groups": {h: {"terms": [t(s, COND["clean"], 1), t(s, "e41_mask_clean.json", -1)]}}})
        cs.append({"name": f"mask cost {h} camera 20 (sopt - masked)",
                   "groups": {h: {"terms": [t(s, COND["20"], 1), t(s, "e41_mask_camera2.json", -1)]}}})
        cs.append({"name": f"masked sopt - none, camera 20, {h}",
                   "groups": {h: {"terms": [t(s, "e41_mask_camera2.json", 1), t(E41[h][0], COND["20"], -1)]}}})
    for lvl in ("10", "30"):
        cs.append({"name": f"interaction pooled {lvl} deg", "groups": {h: {"weight": w3, "terms": inter(h, lvl)} for h in E41}})
    return {"name": "E41 camera dose and SOPT masking (seeds 0-1)", "contrasts": cs, "primary": [cs[0]["name"], cs[1]["name"]]}


def e38b():  # stats_all manifest: prior instances on SmolVLA and FastWAM
    sm = {"none": [smolvla("none", "k5", s) for s in SEEDS],
          "droid_p0": [smolvla("expert_droid", "k5", s) for s in SEEDS],
          "droid_pB1": [f"e23_smolvla_expert_droid_k5_t17_pB1_s{s}" for s in SEEDS],
          "droid_pB2": [f"e23_smolvla_expert_droid_k5_t17_pB2_s{s}" for s in SEEDS]}
    wam = {"none": [f"e12_wam_none_k5_t17_s{s}" for s in SEEDS],
           "droid_p0": [f"e12_wam_expert_droid_k5_t17_s{s}" for s in SEEDS],
           "droid_pB1": [f"e12_wam_expert_droid_k5_t17_pB1_s{s}" for s in SEEDS],
           "droid_pB2": [f"e12_wam_expert_droid_k5_t17_pB2_s{s}" for s in SEEDS]}
    pairs = [["droid_p0", "none"], ["droid_pB1", "none"], ["droid_pB2", "none"], ["droid_pB1", "droid_p0"],
             ["droid_pB2", "droid_p0"]]
    prim = [["droid_pB1", "none"], ["droid_pB2", "none"]]
    return {"_comment": "E38b: E38 prior-instance replication extended to SmolVLA (main-table k5 runs: Jubail re-runs "
                        "_jr for seeds 0/1) and FastWAM. Holm over the 4 primaries. Generated by "
                        "scripts/make_contrast_specs.py",
            "comparisons": [{"name": "E38b SmolVLA x prior instance, LIBERO-90-17 k5", "arms": sm, "pairs": pairs,
                             "primary": prim},
                            {"name": "E38b FastWAM x prior instance, LIBERO-90-17 k5", "arms": wam, "pairs": pairs,
                             "primary": prim}]}


def smolvla_main():  # stats_all manifest: SmolVLA main table, all three seeds on Jubail
    arms = {a: [smolvla(a, "k5", s) for s in SEEDS] for a in ("none", "expert_scratch", "expert_droid")}
    return {"_comment": "SmolVLA 17-task main table from the Jubail runs (seeds 0/1 re-run with TAG=_jr, seed 2). "
                        "Generated by scripts/make_contrast_specs.py",
            "comparisons": [{"name": "SmolVLA (Jubail) LIBERO-90-17 k5", "arms": arms,
                             "pairs": [["expert_droid", "none"], ["expert_droid", "expert_scratch"],
                                       ["expert_scratch", "none"]]}]}


if __name__ == "__main__":
    for name, obj in (("e39_contrasts", e39()), ("e40_contrasts", e40()), ("e41_contrasts", e41()),
                      ("e38b_manifest", e38b()), ("smolvla_jubail_manifest", smolvla_main())):
        (C / f"{name}.json").write_text(json.dumps(obj, indent=1))
        print("wrote", C / f"{name}.json")
