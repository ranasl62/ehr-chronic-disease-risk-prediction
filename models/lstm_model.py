"""Optional sequence models over irregular EHR timelines (requires PyTorch and tensorized episodes).

Tabular aggregates from ``feature_engineering`` power the default training CLI; use this module
when you need learned representations over ordered events rather than summary statistics.
"""


def build_lstm_model():
    raise NotImplementedError(
        "LSTM training is not enabled in this package build. "
        "Add PyTorch, materialize padded event sequences per episode, "
        "and tie labels to the prediction time before calling this trainer."
    )
