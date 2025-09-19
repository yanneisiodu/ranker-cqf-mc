import joblib
import pandas as pd
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def show_feature_importance(model_pipeline_path, feature_names_path):
    """
    Loads a saved XGBoost pipeline and its corresponding feature names,
    then prints the feature importances.
    """
    try:
        logging.info(f"Loading XGBoost pipeline from: {model_pipeline_path}")
        pipeline = joblib.load(model_pipeline_path)
        logging.info(f"Pipeline loaded successfully. Steps: {list(pipeline.named_steps.keys())}")

        if 'ranker' not in pipeline.named_steps:
            logging.error("Error: 'ranker' step not found in the pipeline.")
            return

        xgb_model = pipeline.named_steps['ranker']
        
        logging.info(f"Loading feature names from: {feature_names_path}")
        # The feature names saved by train_xgboost_ranking_model.py are the ones *before* one-hot encoding.
        # The preprocessor step in the pipeline handles the one-hot encoding.
        # So, we need to get the transformed feature names from the preprocessor.
        
        original_feature_names = joblib.load(feature_names_path) # These are the input columns to preprocessor
        logging.info(f"Loaded {len(original_feature_names)} original feature names (inputs to preprocessor).")

        if hasattr(pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
            # This gets the names of features *after* transformation (e.g., one-hot encoding)
            transformed_feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
            logging.info(f"Preprocessor generated {len(transformed_feature_names)} transformed feature names.")
        else:
            logging.warning("Preprocessor does not have 'get_feature_names_out'. Cannot get transformed feature names accurately. "
                            "This might happen with older scikit-learn versions or custom preprocessors. "
                            "Feature importances might not be correctly named.")
            # Fallback: If we can't get transformed names, the importances array might not match original_feature_names
            # For now, we'll proceed assuming the number of importances matches transformed features.
            # A more robust solution would be to ensure train_xgboost_ranking_model.py saves the *transformed* names
            # if this becomes an issue.
            # Let's assume, for now, that the number of importances from xgb_model matches original_feature_names if get_feature_names_out is missing.
            # This assumption is likely INCORRECT if one-hot encoding is used.
            transformed_feature_names = original_feature_names # This is a risky fallback


        importances = xgb_model.feature_importances_
        
        if len(importances) != len(transformed_feature_names):
            logging.error(f"Mismatch in length of importances ({len(importances)}) and transformed feature names ({len(transformed_feature_names)}).")
            logging.error("This often happens if the preprocessor's get_feature_names_out() method is not available or not used correctly, "
                          "especially when one-hot encoding expands the number of features.")
            logging.info("Original feature names (input to preprocessor):")
            for i, name in enumerate(original_feature_names):
                logging.info(f"  {i}: {name}")
            logging.info("Feature importances array (length may differ):")
            for i, imp in enumerate(importances):
                logging.info(f"  Importance {i}: {imp}")
            return


        feature_importance_df = pd.DataFrame({
            'feature': transformed_feature_names,
            'importance': importances
        })
        
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        
        logging.info("\n--- XGBoost Model Feature Importances ---")
        print(feature_importance_df.to_string())

    except FileNotFoundError:
        logging.error(f"Error: Model or feature names file not found. Searched:\n  {model_pipeline_path}\n  {feature_names_path}")
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display feature importances for a trained XGBoost ranking model.")
    parser.add_argument(
        "--model_path", 
        type=str, 
        # Assuming the script is in xgboost_models, and model_output is a subdir
        default="model_output/xgboost_ranker_2022_2024_fixed_params_20250510_162612.joblib", 
        help="Path to the saved XGBoost pipeline (.joblib file)."
    )
    parser.add_argument(
        "--features_path", 
        type=str, 
        default="model_output/xgb_feature_names_2022_2024_20250510_162612.pkl", 
        help="Path to the saved list of feature names (.pkl file)."
    )
    
    args = parser.parse_args()

    # Construct absolute paths if the provided paths are relative to the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path_abs = args.model_path
    if not os.path.isabs(model_path_abs):
        model_path_abs = os.path.join(script_dir, model_path_abs)
        
    features_path_abs = args.features_path
    if not os.path.isabs(features_path_abs):
        features_path_abs = os.path.join(script_dir, features_path_abs)

    show_feature_importance(model_path_abs, features_path_abs) 