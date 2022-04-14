import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Automobile Price Prediction App
This app predicts the **Automobile Price**!
""")
st.write('---')

# Loads the Automobile Dataset
df = pd.read_csv("automobile-cleans.csv") 
df=df.dropna()
X = pd.DataFrame(df, columns=("wheelbase","horsepower","length","width","curbweight","enginesize","bore","citympg","highwaympg","stroke"))
y=df[['price']]

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    drivewheels = st.sidebar.slider('drivewheels',0,2,1)
    wheelbase = st.sidebar.slider('wheelbase',86 ,121 ,99)
    horsepower = st.sidebar.slider('horsepower',48,262,104)
    length = st.sidebar.slider('length',0,1,0)
    width = st.sidebar.slider('width',0,1,0)
    curbweight = st.sidebar.slider('curbweight',1488,4066,2555)
    enginesize = st.sidebar.slider('enginesize',61,326,126)
    bore = st.sidebar.slider('bore',2,4,3)
    citympg = st.sidebar.slider('citympg',13,49,25)
    highwaympg = st.sidebar.slider('highwaympg',16,54,30)
    data = {'drivewheels': drivewheels,
            'wheel-base': wheelbase,
            'horsepower': horsepower,
            'length': length,
            'width': width,
            'curb-weight': curbweight,
            'engine-size': enginesize,
            'bore': bore,
            'city-mpg': citympg,
            'highway-mpg': highwaympg,}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, y)
# Apply Model to Make Prediction
pred = model.predict(df)

st.header('Prediction of Price')
st.write(pred)
st.write('---')

# Explaining the model's predictions using SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')




