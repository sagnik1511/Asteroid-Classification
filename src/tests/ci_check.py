from src.models.custom_classifier_training import classifier_training


if __name__ == "__main__":

    processing_config = "config/models/custom_training_config.yaml"
    model_config = "config/models/tree_based_model.yaml"

    # training classifier
    classifier_training(processing_config, model_config, testing=True)

    print("Process completed successfully...")
