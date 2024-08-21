import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import qrcode
from PIL import Image

# Functions for data processing and recommendations
def extract_lat_lon(address):
    lat, lon = map(float, address.split(','))
    return lat, lon

def get_top_businesses(group, n=5):
    return group['business_name'].value_counts().nlargest(n).index.tolist()

def prepare_recommendation_data(df):
    df['age_group'] = pd.cut(df['age'], bins=[0, 18, 25, 35, 45, 55, 65, 100], labels=['0-18', '19-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    
    coords = df[['user_lat', 'user_log']].values
    kmeans = KMeans(n_clusters=10, random_state=42)
    df['location_cluster'] = kmeans.fit_predict(coords)
    
    grouped = df.groupby(['age_group', 'gender', 'location_cluster'])
    top_businesses = grouped.apply(get_top_businesses).reset_index()
    top_businesses.columns = ['age_group', 'gender', 'location_cluster', 'top_businesses']
    
    return df, kmeans, top_businesses

def recommend_businesses(age, gender, lat, lon, category, df, kmeans, top_businesses):
    age_group = pd.cut([age], bins=[0, 18, 25, 35, 45, 55, 65, 100], labels=['0-18', '19-25', '26-35', '36-45', '46-55', '56-65', '65+'])[0]
    location_cluster = kmeans.predict([[lat, lon]])[0]

    if category != 'All' and category is not None:
        df = df[df['categories'].str.contains(category, case=False, na=False)]

    recommendations = top_businesses[
        (top_businesses['age_group'] == age_group) &
        (top_businesses['gender'] == gender) &
        (top_businesses['location_cluster'] == location_cluster)
    ]

    if recommendations.empty:
        recommendations = top_businesses[
            (top_businesses['age_group'] == age_group) &
            (top_businesses['gender'] == gender)
        ]
    
    if recommendations.empty:
        return df['business_name'].value_counts().nlargest(5).index.tolist()
    else:
        return recommendations.iloc[0]['top_businesses']

# Load and prepare data
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv('updated_yelp_dataset.csv')
    df, kmeans, top_businesses = prepare_recommendation_data(df)
    return df, kmeans, top_businesses

def get_recommendation():
    age = st.session_state.user_info["age"]
    gender = st.session_state.user_info["gender"]
    lat = st.session_state.user_info["latitude"]
    lon = st.session_state.user_info["longitude"]
    category = st.session_state.user_info["category"]
    recommendations = recommend_businesses(age, gender, lat, lon, category, df, kmeans, top_businesses)
    return f"Based on your information, I recommend these businesses: {', '.join(recommendations[:3])}"

def main():
    st.set_page_config(page_title="Business Recommendation Chatbot", page_icon="🤖", layout="wide")
    st.title("Business Recommendation Chatbot")

    global df, kmeans, top_businesses
    df, kmeans, top_businesses = load_data()

    categories = [
        "Restaurants", "Bars", "Markets", "Grocery Stores", "Diners", "Cafes", "Arts",
        "Bakeries", "Beauty", "Car Dealer", "Event Planning", "Hotels", "Travels",
        "Finance", "Local Services", "Contractors", "Home Service", "Clothing",
        "Florists", "Makeup Artist", "Hospitals", "Delivery", "Other"
    ]

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_info" not in st.session_state:
        st.session_state.user_info = {"age": None, "gender": None, "latitude": None, "longitude": None, "category": None}
    if "step" not in st.session_state:
        st.session_state.step = 0

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Welcome message
    if st.session_state.step == 0:
        welcome_msg = """
        👋 Welcome to the Business Recommendation Chatbot! 
        I'm here to help you discover great businesses tailored to your preferences. 
        Let's get started with a few quick questions. 
        You can type 'restart' at any time to begin again.
        
        First, could you tell me your age?
        """
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
        with st.chat_message("assistant"):
            st.markdown(welcome_msg)
        st.session_state.step = 1

    # User input handling
    if prompt := st.chat_input("Your response"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if prompt.lower() == 'restart':
            st.session_state.user_info = {"age": None, "gender": None, "latitude": None, "longitude": None, "category": None}
            st.session_state.step = 0
            st.experimental_rerun()

        if st.session_state.step == 1:  # Age
            try:
                age = int(prompt)
                if 0 <= age <= 120:
                    st.session_state.user_info["age"] = age
                    response = "Great! Now, what's your gender?"
                    st.session_state.step = 2
                else:
                    response = "Please enter a valid age between 0 and 120."
            except ValueError:
                response = "Please enter a valid age as a number."

        elif st.session_state.step == 2:  # Gender
            gender = st.radio("Select your gender:", ["Male", "Female", "Non-binary"])
            st.session_state.user_info["gender"] = gender[0].upper()
            response = "Thank you. Now, could you provide your latitude?"
            st.session_state.step = 3

        elif st.session_state.step == 3:  # Latitude
            try:
                lat = float(prompt)
                if -90 <= lat <= 90:
                    st.session_state.user_info["latitude"] = lat
                    response = "Great! Now, what's your longitude?"
                    st.session_state.step = 4
                else:
                    response = "Please enter a valid latitude between -90 and 90."
            except ValueError:
                response = "Please enter a valid latitude as a number."

        elif st.session_state.step == 4:  # Longitude
            try:
                lon = float(prompt)
                if -180 <= lon <= 180:
                    st.session_state.user_info["longitude"] = lon
                    response = "Great! Finally, what category of business are you interested in?"
                    st.session_state.step = 5
                else:
                    response = "Please enter a valid longitude between -180 and 180."
            except ValueError:
                response = "Please enter a valid longitude as a number."

        elif st.session_state.step == 5:  # Category
            category = st.selectbox("Select a business category:", categories)
            st.session_state.user_info["category"] = category
            recommendations = get_recommendation()
            response = f"Based on your information, here are my top recommendations:\n\n{recommendations}"
            st.session_state.step = 6

        else:
            response = "Would you like to get new recommendations? Type 'restart' to begin again."

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Progress bar
    progress = st.session_state.step / 6  # This will give a value between 0 and 1
    st.progress(progress)

if __name__ == "__main__":
    main()