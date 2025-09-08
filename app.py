import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import os

# Path relative to current script
model_path = os.path.join(os.path.dirname(__file__), "pipe.pkl")

with open(model_path, "rb") as f:
    pipe = pickle.load(f)
# Load dataset
data_path = os.path.join(os.path.dirname(__file__), "df.pkl")
with open(data_path, "rb") as f:
    df = pickle.load(f)

# defind sidebar items
app_mode = st.sidebar.selectbox("Select the type", ['Diabetes Predictor', 'Visualization'])


# for prediction
if app_mode == 'Diabetes Predictor':
    st.title('Diabetes Predictor')
    #categorical columns
    categorical_cols = df.select_dtypes(
        include = ['object']
    ).columns.tolist()
    
    #numeric columns
    
    numeric_cols = df.select_dtypes(
        include = ['int64','float64']
    ).columns.tolist()
    # remove target column
    if 'Outcome' in numeric_cols:
         numeric_cols.remove('Outcome')
    
    # input for user
    st.subheader("Provide your current medical parameters")
    
    user_inputs ={}
    
    # categorical input
    for feature in categorical_cols:
        if feature == 'BloodPressure':
            st.write("""
            **Diastolic Blood Pressure Categories:**
            - Normal: < 80 mm Hg  
            - High BP (Stage 1): 80 – 89 mm Hg  
            - High BP (Stage 2): 90 – 119 mm Hg
            """)
            user_inputs[feature] = st.selectbox (f"{"BloodPressure"}",df[feature].unique())
        if feature == 'BMI':
            st.write("""
            **BMI Categories:**
            - Underweight: < 18.5  
            - Healthy Weight: 18.5 – 24.9  
            - Obesity: ≥ 25
            """)  
            user_inputs[feature] = st.selectbox (f"{"BMI"}",df[feature].unique())
            
    # Numeric inputs
    for feature in numeric_cols:
        user_inputs[feature] = st.selectbox(f"{feature}", sorted(df[feature].unique()))
        
    # --- Predict Button ---
    if st.button("Predict"):    
        if file is not None:
             if hasattr(pipe, 'predict'):
                predict_output = pipe.predict(input_df)[0]
                # Convert user inputs into a DataFrame
                input_df = pd.DataFrame([user_inputs])
                
                try:
                    predict_output = file.predict(input_df)[0]
                    if predict_output == 1:
                       st.error("Your diabetes report is positive")
                    else:
                       st.success("Your diabetes report is negative")
                except Exception as e:
                   st.error(f"Prediction failed: {e}")
                
                        
   
# Visualization
if app_mode == 'Visualization':
    st.title("Diabetes Data Visualization")
    visu_type = st.sidebar.radio("Visualization", ['Show dataset basic information','Diabetes Outcome Distribution', 'Numeric features distribution','Correlation heatmap','Compare numeric features by Outcome','Age vs DiabetesPedigreeFunction'])
    
    #Show dataset basic information
    if visu_type == 'Show dataset basic information':
        if st.button("show dataset basic information"):
           st.write("### Top 5 samples")
           st.dataframe(df.head())
           st.write('### Stats report of dataset')
           st.write(df.describe(include = 'all'))
           st.write('### shape of dataset')
        
    #Diabetes Outcome Distribution    
    if visu_type == 'Diabetes Outcome Distribution':      
        outcome_counts = df['Outcome'].value_counts().reset_index()
        outcome_counts.columns = ['Outcome', 'Count']
        outcome_counts['Outcome'] = outcome_counts['Outcome'].map({0: 'Negative', 1: 'Positive'})
    
        # Create interactive pie chart
        fig = px.pie(outcome_counts, names='Outcome', values='Count', 
             color='Outcome', 
             color_discrete_map={'Negative':'green','Positive':'red'},
             title='Diabetes Outcome Distribution')
        st.plotly_chart(fig)
    
    
    # Numeric features distribution
    if visu_type == 'Numeric features distribution':
        st.write("### Numeric Feature Distribution")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'Outcome' in numeric_cols:
            numeric_cols.remove('Outcome')

            selected_feature = st.selectbox("Select a numeric feature to visualize", numeric_cols)

            # Create interactive histogram
            fig2 = ff.create_distplot(
                             [df[selected_feature]],  # List of arrays
                             group_labels=[selected_feature],
                             bin_size=(df[selected_feature].max() - df[selected_feature].min())/20,  # approx 20 bins
                             show_rug=False
                             )

            fig2.update_layout(title=f"Distribution of {selected_feature}")
            st.plotly_chart(fig2)

    # Correlation heatmap
    if visu_type == 'Correlation heatmap':
        st.write("### Correlation Heatmap")
        # Compute correlation matrix
        corr_matrix = df.corr(numeric_only=True)
        # Create interactive heatmap
        fig3 = px.imshow(
               corr_matrix,
               text_auto=True,   # Show correlation values
               color_continuous_scale='RdBu_r',
               origin='upper',
               aspect='auto',
               title="Correlation Heatmap"
        )

        st.plotly_chart(fig3)
        
    # Compare numeric features by Outcome
    if visu_type == 'Compare numeric features by Outcome':
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'Outcome' in numeric_cols:
            numeric_cols.remove('Outcome')
            selected_feature = st.selectbox("Select a numeric feature to visualize", numeric_cols)
        
            fig4 = px.box(df, x='Outcome', y=selected_feature, color='Outcome',
                           title=f'{selected_feature} by Outcome',
                           labels={'Outcome':'Diabetes Outcome', selected_feature: selected_feature})
            st.plotly_chart(fig4)
    
    color_map = {0: 'green', 1: 'red'}

    # Scatter plot: Age vs DiabetesPedigreeFunction
    if visu_type == 'Age vs DiabetesPedigreeFunction':
        fig5 = px.scatter(df,
                 x='Age',
                 y='DiabetesPedigreeFunction',
                 color='Outcome',
                 color_discrete_map=color_map,
                 title='Age vs Diabetes Pedigree Function by Outcome',
                 labels={'Age':'Age', 'DiabetesPedigreeFunction':'Diabetes Pedigree Function', 'Outcome':'Diabetes Outcome'},
                 template='plotly_dark',  # optional dark mode
                 height=500)

        st.plotly_chart(fig5)