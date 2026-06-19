"""Deterministic cloth parameter presets for replay comparisons."""

from copy import deepcopy


CLOTH_VARIANTS = {
    "normal": {
        "description": "Baseline cloth: original 9x9 MuJoCo composite grid.",
        "model_kwargs_rownum": 0,
        "cloth_grid_count": 9,
        "geom_density": 1000.0,
        "skin_inflate": 0.001,
        "skin_subgrid": 3,
        "custom_model_kwargs": {},
    },
    "light": {
        "description": "Light cloth: same 9x9 grid with one tenth of the normal density.",
        "model_kwargs_rownum": 0,
        "cloth_grid_count": 9,
        "geom_density": 100.0,
        "skin_inflate": 0.001,
        "skin_subgrid": 3,
        "custom_model_kwargs": {},
    },
    "stiff": {
        "description": "Coarse stiff cloth: fewer faces plus thicker rendered/contact cloth.",
        "model_kwargs_rownum": 0,
        "cloth_grid_count": 5,
        "geom_density": 1000.0,
        "skin_inflate": 0.002,
        "skin_subgrid": 1,
        "custom_model_kwargs": {
            "geom_size": 0.004,
            "cone_type": "elliptic",
            "friction": 1.15,
            "impratio": 30.0,
            "joint_solimp_low": 0.993,
            "joint_solimp_high": 0.996,
            "joint_solimp_width": 0.018,
            "joint_solref_timeconst": 0.018,
            "joint_solref_dampratio": 1.0,
            "tendon_shear_solimp_low": 0.993,
            "tendon_shear_solimp_high": 0.996,
            "tendon_shear_solimp_width": 0.018,
            "tendon_shear_solref_timeconst": 0.018,
            "tendon_shear_solref_dampratio": 1.0,
            "tendon_main_solimp_low": 0.995,
            "tendon_main_solimp_high": 0.998,
            "tendon_main_solimp_width": 0.012,
            "tendon_main_solref_timeconst": 0.012,
            "tendon_main_solref_dampratio": 1.0,
            "geom_solimp_low": 0.993,
            "geom_solimp_high": 0.997,
            "geom_solimp_width": 0.012,
            "geom_solref_timeconst": 0.008,
            "geom_solref_dampratio": 1.0,
        },
    },
}


def apply_cloth_variant(variant, variant_name):
    """Apply a preset to a general_utils variant dictionary in-place."""
    if variant_name not in CLOTH_VARIANTS:
        choices = ", ".join(sorted(CLOTH_VARIANTS))
        raise ValueError(f"Unknown cloth variant '{variant_name}'. Choices: {choices}")

    preset = deepcopy(CLOTH_VARIANTS[variant_name])
    randomization_kwargs = variant["randomization_kwargs"]
    randomization_kwargs["dynamics_randomization"] = False
    randomization_kwargs["model_kwargs_rownum"] = preset["model_kwargs_rownum"]
    randomization_kwargs["cloth_grid_count"] = preset["cloth_grid_count"]
    randomization_kwargs["geom_density"] = preset["geom_density"]
    randomization_kwargs["skin_inflate"] = preset["skin_inflate"]
    randomization_kwargs["skin_subgrid"] = preset["skin_subgrid"]
    randomization_kwargs["custom_model_kwargs"] = preset["custom_model_kwargs"]
    variant["cloth_variant"] = {
        "name": variant_name,
        **preset,
    }
    return variant


def variant_names():
    return tuple(CLOTH_VARIANTS.keys())
