"""Canonical ASDSpeech feature groups.

The indices are derived from ``data/train_data.mat`` variable ``feat_names``.
Keep this file as the single source of truth for feature-group dropout,
group-quality synthetic-data scoring, and manuscript figures.
"""

from __future__ import annotations

from collections import OrderedDict


FEATURE_NAMES = [
    "mean_Pitch",
    "Pitch_var",
    "NormedPitch_var",
    "mean_MinPitch",
    "minPitch_var",
    "mean_MaxPitch",
    "MaxPitch_var",
    "mean_VoicedPitch",
    "VoicedPitch_var",
    "MeanVoicedPitch_var",
    "mean_Voicing",
    "Voicing_var",
    "mean_F1",
    "F1_var",
    "mean_F2",
    "F2_var",
    "mean_|F1-F2|",
    "|F1-F2|_var",
    "mean_Bandwidth1",
    "Bandwidth1_var",
    "mean_Bandwidth2",
    "Bandwidth2_var",
    "mean_SpecSlope20_500",
    "SpecSlope20_500_var",
    "mean_SpecSlope500_1500",
    "SpecSlope500_1500_var",
    "mean_SpecSlopeVoiced20_500",
    "mean_SpecSlopeUnvoiced20_500",
    "mean_SpecSlopeVoiced500_1500",
    "mean_SpecSlopeUnvoiced500_1500",
    "mean_Jitter",
    "Jitter_var",
    "mean_Energy",
    "mean_DeltaEnergy",
    "mean_DeltaDeltaEnergy",
    "mean_|DeltaEnergy|",
    "Energy_var",
    "DeltaEnergy_var",
    "DeltaDeltaEnergy_var",
    "|DeltaEnergy|_var",
    "mean_ZCR",
    "ZCR_var",
    "mean_VoicedZCR",
    "VoicedZCR_var",
    "mean_UnvoicedZCR",
    "UnvoicedZCR_var",
    "mean_LengthInRange",
    "LengthInRange_var",
    "Num_voc_segs",
]


FEATURE_GROUPS = OrderedDict(
    [
        ("pitch", list(range(0, 10))),
        ("voicing", list(range(10, 12))),
        ("formants", list(range(12, 22))),
        ("spectral", list(range(22, 30))),
        ("jitter", list(range(30, 32))),
        ("energy", list(range(32, 40))),
        ("zcr", list(range(40, 46))),
        ("duration", list(range(46, 48))),
        ("quantity", [48]),
    ]
)


FEATURE_GROUP_LABELS = OrderedDict(
    [
        ("pitch", "Pitch"),
        ("voicing", "Voicing"),
        ("formants", "Formants/bandwidth"),
        ("spectral", "Spectral slope"),
        ("jitter", "Jitter"),
        ("energy", "Energy"),
        ("zcr", "ZCR"),
        ("duration", "Duration"),
        ("quantity", "Quantity"),
    ]
)


TITLE_FEATURE_GROUPS = OrderedDict(
    (FEATURE_GROUP_LABELS[key], value) for key, value in FEATURE_GROUPS.items()
)


RRB_SENSITIVE_GROUPS = {"pitch", "jitter", "voicing", "spectral", "duration"}


def group_for_feature(index: int) -> str:
    for group, indices in FEATURE_GROUPS.items():
        if index in indices:
            return group
    raise KeyError(f"No feature group defined for index {index}")


def validate_feature_groups(n_features: int = 49) -> None:
    all_indices = [idx for indices in FEATURE_GROUPS.values() for idx in indices]
    expected = list(range(n_features))
    if sorted(all_indices) != expected:
        raise ValueError(
            "Feature groups must cover each feature index exactly once; "
            f"got {sorted(all_indices)}"
        )
