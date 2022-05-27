from src.PredictionPipeline import PredictionPipeline


def main():
    shuffle_data = False
    display_data_analysis = False

    prediction_pipeline = PredictionPipeline()

    if shuffle_data:
        prediction_pipeline.shuffle_data(5)

    prediction_pipeline.prepare_data()

    if display_data_analysis:
        prediction_pipeline.exploratory_data_analysis()

    prediction_pipeline.train_ridge_regression()
    prediction_pipeline.train_neural_network()
    prediction_pipeline.train_random_forest()
    prediction_pipeline.train_gradient_boosting()


main()


