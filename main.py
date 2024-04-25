import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# Load the model and encoders
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

encoders = {}
for label in ['driver', 'car', 'grand_prix']:
    with open(f'{label}_encoder.pkl', 'rb') as file:
        encoders[label] = pickle.load(file)

# Load data for visualization
@st.cache
def load_data():
    return pd.read_csv('f1_data_races.csv')

f1_data = load_data()

# Streamlit interface for prediction
st.title('F1 Top 3 Finish Predictor')
driver = st.selectbox('Select Driver', encoders['driver'].classes_)
car = st.selectbox('Select Car', encoders['car'].classes_)
grand_prix = st.selectbox('Select Grand Prix', encoders['grand_prix'].classes_)
laps = st.number_input('Enter Laps Completed', min_value=1, max_value=200, value=50)

if st.button('Predict Finish'):
    driver_encoded = encoders['driver'].transform([driver])[0]
    car_encoded = encoders['car'].transform([car])[0]
    grand_prix_encoded = encoders['grand_prix'].transform([grand_prix])[0]
    
    input_data = pd.DataFrame([{
        'driver_encoded': driver_encoded,
        'car_encoded': car_encoded,
        'grand_prix_encoded': grand_prix_encoded,
        'laps': laps
    }])
    
    prediction = model.predict(input_data)
    result = 'likely' if prediction[0] == 1 else 'unlikely'
    st.write(f"{driver} is {result} to finish in the top 3 at the {grand_prix} Grand Prix.")

# Visualization section
st.subheader('Visualizing The Drivers, Teams, Grandprix and Postions')
fig = px.scatter_3d(f1_data, x='driver', y='car', z='grand_prix', color='position', 
                    title='Relationship Between Driver, Car Team, and Grand Prix',
                    labels={'driver': 'Driver', 'car': 'Car Team', 'grand_prix': 'Grand Prix', 'position': 'Position'})
st.plotly_chart(fig)

# Distribution of Drivers
driver_counts = f1_data['driver'].value_counts()
fig_driver = px.bar(driver_counts, x=driver_counts.index, y=driver_counts.values,
                    labels={'x': 'Driver', 'y': 'Number of Races'}, title='Distribution of Drivers')
st.plotly_chart(fig_driver)

# Distribution of Car Teams
car_counts = f1_data['car'].value_counts()
fig_car = px.bar(car_counts, x=car_counts.index, y=car_counts.values,
                 labels={'x': 'Car Team', 'y': 'Number of Races'}, title='Distribution of Car Teams')
st.plotly_chart(fig_car)

# Distribution of Grand Prix Events
gp_counts = f1_data['grand_prix'].value_counts()
fig_gp = px.bar(gp_counts, x=gp_counts.index, y=gp_counts.values,
                labels={'x': 'Grand Prix Event', 'y': 'Number of Races'}, title='Distribution of Grand Prix Events')
st.plotly_chart(fig_gp)

st.subheader("Conclusion and Use Case")
st.markdown("""
            This model allows users to explore how different scenarios might affect the outcomes of Formula 1 races.
            By adjusting parameters, users can gain insights into how certain drivers might perform under various conditions, helping teams and fans make informed decisions.
            """)
