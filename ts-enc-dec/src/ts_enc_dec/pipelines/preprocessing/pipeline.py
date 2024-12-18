"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import simple_ts_data_prep , plot_simple_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=simple_ts_data_prep,
                inputs=["params:target_name", "params:simple_data_prep"],
                outputs=dict(
                    x_all="x_all",
                    y_all="y_all",
                    x_train="x_train",
                    y_train="y_train",
                    x_valid="x_valid",
                    y_valid="y_valid",
                ),
                name="data_preparation",
            ),
        node(
                func=plot_simple_data,
                inputs=["x_all", "y_all", "x_train", "y_train", "x_valid", "y_valid", "params:simple_plot_tr_ts"],
                outputs="data_plot",
                name="data_plots",
            ),

    ])
