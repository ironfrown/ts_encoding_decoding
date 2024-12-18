"""Project pipelines."""

from kedro.pipeline import Pipeline
from ts_enc_dec.pipelines import preprocessing as prep
from ts_enc_dec.pipelines import swprep as sw


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_ts_pipeline = prep.create_pipeline()
    data_sliding_window_pipeline = sw.create_pipeline()

    return {
        "prep": data_ts_pipeline,
        "sw":   data_sliding_window_pipeline,

        "__default__": data_ts_pipeline + data_sliding_window_pipeline
    }
