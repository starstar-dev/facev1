FULL_ABLATION = {
    "flare_sampler": True,
    "modality_dropout": 0.0,
    "use_mfmp": True,
    "use_mcloss": True,
    "use_coen_lite": True,
    "use_fusion": True,
    "use_quality_loss_gate": True,
    "use_supcon": True,
    "coen_use_learned_qmap": True,
    "coen_use_image_prior": True,
    "coen_use_disagreement": True,
    "use_qmap_aux_loss": True,
    "use_global_static_fusion": False,
    "use_tir_anchor": True,
}


def _exp(desc, **overrides):
    cfg = FULL_ABLATION.copy()
    cfg.update(overrides)
    cfg["desc"] = desc
    return cfg


ABLATION_CONFIGS = {
    "backbone": _exp(
        "Backbone: CLIP-ViT x3 only",
        use_mfmp=False,
        use_mcloss=False,
        use_coen_lite=False,
        use_fusion=False,
        use_quality_loss_gate=False,
        use_supcon=False,
        coen_use_learned_qmap=False,
        coen_use_image_prior=False,
        coen_use_disagreement=False,
        use_qmap_aux_loss=False,
    ),
    "global_static_fusion": _exp(
        "Global Static Fusion: replace TPQE with learned scalar per modality",
        use_global_static_fusion=True,
        # use_coen_lite 保持 True (从 FULL_ABLATION 继承)，确保 loss gate 正常工作
    ),
    "full": _exp("Full model (same data setting as original train_exp G)"),
    "original_G": _exp("Original train_exp G equivalent: full model + flare sampler"),
    "wo_learned_qmap": _exp(
        "Full w/o learned qmap",
        coen_use_learned_qmap=False,
        use_qmap_aux_loss=False,
    ),
    "wo_image_prior": _exp(
        "Full w/o image prior",
        coen_use_image_prior=False,
    ),
    "wo_disagreement": _exp(
        "Full w/o cross-modal disagreement",
        coen_use_disagreement=False,
    ),
    "wo_quality_detection": _exp(
        "Full w/o quality detection, q=1",
        coen_use_learned_qmap=False,
        coen_use_image_prior=False,
        coen_use_disagreement=False,
        use_quality_loss_gate=False,
        use_qmap_aux_loss=False,
    ),
    "wo_quality_fusion": _exp(
        "Full w/o quality-guided fusion",
        use_fusion=False,
    ),
    "wo_quality_loss_gate": _exp(
        "Full w/o quality loss gate",
        use_quality_loss_gate=False,
    ),
    "wo_supcon": _exp(
        "Full w/o SupCon",
        use_supcon=False,
    ),
    "wo_tir_anchor": _exp(
        "w/o TIR Anchor: remove TI as fixed reference, RGB↔NI mutual only",
        use_tir_anchor=False,
    ),
}


ALIASES = {
    "a": "wo_learned_qmap",
    "b": "wo_image_prior",
    "c": "wo_disagreement",
    "d": "wo_quality_detection",
    "e": "wo_quality_fusion",
    "f": "wo_quality_loss_gate",
    "g": "wo_supcon",
}


def resolve_ablation_name(name):
    return ALIASES.get(name, name)


def apply_ablation_config(model, name):
    name = resolve_ablation_name(name)
    if name not in ABLATION_CONFIGS:
        valid = ", ".join(sorted(list(ABLATION_CONFIGS) + list(ALIASES)))
        raise KeyError(f"Unknown ablation '{name}'. Valid options: {valid}")

    cfg = ABLATION_CONFIGS[name]
    model.use_mfmp = cfg["use_mfmp"]
    model.use_mcloss = cfg["use_mcloss"]
    model.use_coen_lite = cfg["use_coen_lite"]
    model.use_fusion = cfg["use_fusion"]
    model.use_quality_loss_gate = cfg["use_quality_loss_gate"]
    model.use_supcon = cfg["use_supcon"]
    model.coen_use_learned_qmap = cfg["coen_use_learned_qmap"]
    model.coen_use_image_prior = cfg["coen_use_image_prior"]
    model.coen_use_disagreement = cfg["coen_use_disagreement"]
    model.use_qmap_aux_loss = cfg["use_qmap_aux_loss"]
    model.use_global_static_fusion = cfg["use_global_static_fusion"]
    model.fusion_R.use_tir_anchor = cfg["use_tir_anchor"]
    model.fusion_N.use_tir_anchor = cfg["use_tir_anchor"]
    return name, cfg


def format_ablation_flags(model):
    fields = [
        ("mfmp", model.use_mfmp),
        ("mcloss", model.use_mcloss),
        ("coen", model.use_coen_lite),
        ("fusion", model.use_fusion),
        ("loss_gate", model.use_quality_loss_gate),
        ("supcon", model.use_supcon),
        ("learned_qmap", model.coen_use_learned_qmap),
        ("image_prior", model.coen_use_image_prior),
        ("disagreement", model.coen_use_disagreement),
        ("qmap_aux", model.use_qmap_aux_loss),
    ]
    return ", ".join(f"{k}={v}" for k, v in fields)
