"""
This is a boilerplate pipeline 'swprep'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import sw_prep_data , plot_flat


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=sw_prep_data,
                inputs=["params:target_name", "params:sw_data_prep"],
                outputs=["X_all_sw", "y_all_sw", "x_train_sw","y_train_sw","x_valid_sw","y_valid_sw"],
                name="data_preparation_sliding_window",
            ),
        node(
                func=plot_flat,
                inputs=["y_all_sw", "y_train_sw", "y_valid_sw","params:sw_data_prep","params:sw_plot_tr_ts"],
                outputs="data_plot_sw",
                name="data_plots_sw_pic",
            ),

    ])
