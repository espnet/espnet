def sample_params(diarization_pipeline, trial: optuna.Trial):

    diarization_pipeline.segmentation.threshold = trial.suggest_float(
        "segmentation.threshold", 0.3, 0.8, step=0.02
    )
    diarization_pipeline.clustering.threshold = trial.suggest_float(
        "clustering.threshold", 0.5, 0.9, step=0.02
    )  # higher than last year
    diarization_pipeline.clustering.min_cluster_size = trial.suggest_int(
        "min_cluster_size", low=10, high=40, step=1
    )


def main():
    pass


if __name__ == "__main__":
    main()
