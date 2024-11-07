import streamlit as st
import joblib
import numpy as np

# Define feature names and descriptions
feature_names = ['target', 'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
       'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
       'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
       'Income']

def main():
    st.title('Diabetes Prediction App')
    st.write("Enter the values for the features to get a prediction.")

    # Load sample data to pre-load default values
    # sample_data = joblib.load('scaler.pkl').inverse_transform(np.array([[0]*30]))  # Placeholder for the actual sample data

    # Define input fields for user to enter feature values with proper labels
    features = []
    for i, feature in enumerate(feature_names):
        feature_value = st.number_input(f'{feature}', min_value=0.0, value=0.0))
        features.append(feature_value)

    # Handle cases where input features may need categorical encoding
    def preprocess_input(features):
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')

        # Scale the features
        features_scaled = scaler.transform(np.array(features).reshape(1, -1))
        return features_scaled

    # Button to make a prediction
    if st.button('Predict'):
        try:
            features_scaled = preprocess_input(features)
            model = joblib.load('model.pkl')
            prediction = model.predict(features_scaled)
            probability = model.predict_proba(features_scaled)[0, 1]

            # Display the result
            st.write(f"Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
            st.write(f"Probability of Positive: {probability:.2f}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

if __name__ == '__main__':
    main()
