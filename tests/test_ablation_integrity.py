from cardilearn.ablation import registry, validate_pair


def test_registry_variants_exist():
    models = registry()
    assert "full" in models
    assert "no_contrastive" in models


def test_ablation_changes_single_component():
    models = registry()
    base = models["full"]
    for name, variant in models.items():
        if name != "full":
            validate_pair(base, variant)
