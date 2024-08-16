

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
from streamlit_folium import folium_static
from sklearn.cluster import KMeans
import random
from math import radians, sin, cos, sqrt, atan2

# # Set page config at the very beginning
# st.set_page_config(page_title="Yelp Business Recommendation System", page_icon="🍽️", layout="wide")


# Functions for data processing and recommendations
def extract_lat_lon(address):
    lat, lon = map(float, address.split(','))
    return lat, lon

def get_top_businesses(group, n=5):
    return group['business_name'].value_counts().nlargest(n).index.tolist()

@st.cache_resource
def prepare_recommendation_data(df):
    if 'user_address' in df.columns:
        df['latitude'], df['longitude'] = zip(*df['user_address'].apply(extract_lat_lon))
        df['age_group'] = pd.cut(df['user_ages'], bins=[0, 18, 25, 35, 45, 55, 65, 100], labels=['0-18', '19-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    else:
        df['age_group'] = pd.cut(df['age'], bins=[0, 18, 25, 35, 45, 55, 65, 100], labels=['0-18', '19-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    
    coords = df[['latitude', 'longitude']].values if 'latitude' in df.columns else df[['user_lat', 'user_log']].values
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

def filter_by_institution(df, institution):
    return df[df['business_name'] == institution]

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in kilometers

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c

    return distance

def chatbot_recommendation(df, kmeans, top_businesses):
    st.title("Recommendation Chatbot")

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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    def get_recommendation():
        age = st.session_state.user_info["age"]
        gender = st.session_state.user_info["gender"]
        lat = st.session_state.user_info["latitude"]
        lon = st.session_state.user_info["longitude"]
        category = st.session_state.user_info["category"]
        recommendations = recommend_businesses(age, gender, lat, lon, category, df, kmeans, top_businesses)
        return f"Based on your information, I recommend these businesses: {', '.join(recommendations[:3])}"

    if not st.session_state.messages:
        response = "Hello! I can help you find business recommendations. First, could you tell me your age?"
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    if prompt := st.chat_input("Your response"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if st.session_state.user_info["age"] is None:
            try:
                age = int(prompt)
                st.session_state.user_info["age"] = age
                response = "Great! Now, what's your gender? (M/F/N)"
            except ValueError:
                response = "Please enter a valid age as a number."
        elif st.session_state.user_info["gender"] is None:
            if prompt.upper() in ['M', 'F', 'N']:
                st.session_state.user_info["gender"] = prompt.upper()
                response = "Thank you. Now, could you provide your latitude?"
            else:
                response = "Please enter M for male, F for female, or N for non-binary."
        elif st.session_state.user_info["latitude"] is None:
            try:
                lat = float(prompt)
                st.session_state.user_info["latitude"] = lat
                response = "Great! Now, what's your longitude?"
            except ValueError:
                response = "Please enter a valid latitude as a number."
        elif st.session_state.user_info["longitude"] is None:
            try:
                lon = float(prompt)
                st.session_state.user_info["longitude"] = lon
                response = "Great! Finally, what category of business are you interested in? Please choose from the following:\n" + "\n".join([f"{i+1}. {cat}" for i, cat in enumerate(categories)])
            except ValueError:
                response = "Please enter a valid longitude as a number."
        elif st.session_state.user_info["category"] is None:
            if prompt.isdigit() and 1 <= int(prompt) <= len(categories):
                st.session_state.user_info["category"] = categories[int(prompt) - 1]
            elif prompt.lower() in [cat.lower() for cat in categories]:
                st.session_state.user_info["category"] = prompt.capitalize()
            else:
                response = f"Please enter a valid category number (1-{len(categories)}) or name from the list provided."
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                return

            response = get_recommendation()
        else:
            response = "I've already given you recommendations. Would you like to start over? (Yes/No)"
            if prompt.lower() == 'yes':
                st.session_state.user_info = {"age": None, "gender": None, "latitude": None, "longitude": None, "category": None}
                response += " Great! Let's start again. What's your age?"

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Load and prepare data
@st.cache_data
def load_data():
    try:
        df_original = pd.read_csv('new_yelp_dataset_combined.csv')
        required_columns_original = ['user_name', 'user_ages', 'gender', 'user_address', 'stars', 'state', 'rec_score', 'business_name']
        for col in required_columns_original:
            if col not in df_original.columns:
                raise ValueError(f"Required column '{col}' not found in the original dataset")
        df_original = df_original.rename(columns={'potential_ages': 'user_ages'})

        df_updated = pd.read_csv('updated_yelp_dataset.csv')
        required_columns_updated = ['user_name', 'age', 'gender', 'bus_lat', 'bus_log', 'stars', 'state', 'rec_score', 'business_name', 'shortest_distance', 'nearest_businesses', 'composite_score']
        for col in required_columns_updated:
            if col not in df_updated.columns:
                raise ValueError(f"Required column '{col}' not found in the updated dataset")

        df_original, kmeans_original, top_businesses_original = prepare_recommendation_data(df_original)
        df_updated, kmeans_updated, top_businesses_updated = prepare_recommendation_data(df_updated)

        return df_original, kmeans_original, top_businesses_original, df_updated, kmeans_updated, top_businesses_updated
    except FileNotFoundError as e:
        st.error(f"File not found: {str(e)}. Please ensure both CSV files are in the same directory as this script.")
        st.stop()
    except ValueError as e:
        st.error(f"Error in dataset: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.stop()

try:
    df_original, kmeans_original, top_businesses_original, df_updated, kmeans_updated, top_businesses_updated = load_data()
except Exception as e:
    st.error(f"Failed to load data: {str(e)}")
    st.stop()

# Create a sidebar for navigation
st.sidebar.title('Menu')
page = st.sidebar.radio('Go to', ['Home', 'User Analysis', 'Business Analysis', 'Business Recommendations', 'Recommendation Chatbot'])


if page == 'Home':
    st.title(" Business Recommendation System")
    st.write("Welcome to our business recommendation system. Choose a page from the sidebar to get started!")

    st.subheader("Quick Stats:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Users", f"{len(df_original['user_name'].unique()):,}")
    with col2:
        st.metric("Total Businesses", f"{len(df_original['business_name'].unique()):,}")
    with col3:
        st.metric("Average Rating", f"{df_original['stars'].mean():.2f}")



if page == 'User Analysis':
    st.title('User Analysis Dashboard')

    # Age Distribution
    st.header('Age Distribution')
    fig_age = px.histogram(df_original, x='user_ages', nbins=20, title='Distribution of User Ages')
    fig_age.update_layout(xaxis_title='Age', yaxis_title='Frequency')
    st.plotly_chart(fig_age)

    # Gender Distribution
    st.header('Gender Distribution')
    gender_counts = df_original['gender'].value_counts()
    fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index, title='Gender Distribution')
    st.plotly_chart(fig_gender)

    # User Map
    st.header('User Map')
    fig_map = px.scatter_mapbox(df_original, lat='latitude', lon='longitude', hover_name='user_name',
                                hover_data=['user_ages', 'gender'],
                                color_discrete_sequence=['fuchsia'], zoom=3, height=300)
    fig_map.update_layout(mapbox_style='open-street-map')
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map)

    # Average Review Scores by Age Group
    df_original['age_group'] = pd.cut(df_original['user_ages'], bins=[0, 20, 30, 40, 50, 60], labels=['0-20', '21-30', '31-40', '41-50', '51-60'])
    fig_age_review = px.bar(df_original.groupby('age_group')['stars'].mean().reset_index(), 
                            x='age_group', y='stars', title='Average Review Scores by Age Group')
    fig_age_review.update_layout(xaxis_title='Age Group', yaxis_title='Average Review Score')
    st.plotly_chart(fig_age_review)

    # Average Review Scores by Gender
    fig_gender_review = px.bar(df_original.groupby('gender')['stars'].mean().reset_index(), 
                               x='gender', y='stars', title='Average Review Scores by Gender')
    fig_gender_review.update_layout(xaxis_title='Gender', yaxis_title='Average Review Score')
    st.plotly_chart(fig_gender_review)

elif page == 'Business Analysis':
    st.title('Business Analysis Dashboard')

    # Add institution filter
    institutions = ['All'] + sorted(df_updated['business_name'].unique().tolist())
    selected_institution = st.selectbox('Select Institution', institutions)

    if selected_institution != 'All':
        filtered_df = filter_by_institution(df_updated, selected_institution)
        st.write(f"Showing data for: {selected_institution}")
    else:
        filtered_df = df_updated
        st.write("Showing data for all institutions")

    # Distribution of Review Scores
    fig_review = px.bar(filtered_df['stars'].value_counts().sort_index().reset_index(), 
                        x='stars', y='count', title='Distribution of Review Scores')
    fig_review.update_layout(xaxis_title='Review Score', yaxis_title='Count')
    st.plotly_chart(fig_review)

    # Distribution of Recommendation Scores
    fig_rec = px.bar(filtered_df['rec_score'].value_counts().sort_index().reset_index(), 
                     x='rec_score', y='count', title='Distribution of Recommendation Scores')
    fig_rec.update_layout(xaxis_title='Recommendation Score', yaxis_title='Count')
    st.plotly_chart(fig_rec)

    # Top Rated Services
    top_services = filtered_df.groupby('service_name')['stars'].mean().sort_values(ascending=False).head(10)
    fig_top_services = px.bar(top_services.reset_index(), x='service_name', y='stars',
                              title='Top 10 Rated Services')
    fig_top_services.update_layout(xaxis_title='Service', yaxis_title='Average Rating')
    st.plotly_chart(fig_top_services)

    # Service Quality Responses
    fig_quality = px.pie(filtered_df['state'].value_counts().reset_index(), 
                         values='count', names='state', title='Distribution of Service Quality Responses')
    st.plotly_chart(fig_quality)

    # User Activity by Location
    fig_activity = px.density_mapbox(filtered_df, lat='bus_lat', lon='bus_log', z='stars', radius=10,
                                     center=dict(lat=filtered_df['bus_lat'].mean(), lon=filtered_df['bus_log'].mean()), zoom=3,
                                     mapbox_style="open-street-map", title='User Activity by Location')
    st.plotly_chart(fig_activity)

    # Additional metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Rating", f"{filtered_df['stars'].mean():.2f}")
    with col2:
        st.metric("Total Reviews", filtered_df.shape[0])
    with col3:
        st.metric("Average Recommendation Score", f"{filtered_df['rec_score'].mean():.2f}")

    # Time series analysis of ratings
    if 'date' in filtered_df.columns:
        filtered_df['date'] = pd.to_datetime(filtered_df['date'])
        time_series = filtered_df.groupby('date')['stars'].mean().reset_index()
        fig_time = px.line(time_series, x='date', y='stars', title='Average Rating Over Time')
        st.plotly_chart(fig_time)

elif page == 'Business Recommendations':
    st.title('Personalized Business Recommendations')

    # Define the list of categories
    categories = [
        "All",  # Add "All" as the first option
        "Restaurants", "Bars", "Markets", "Grocery Stores", "Diners", "Cafes", "Arts",
        "Bakeries", "Beauty", "Car Dealer", "Event Planning", "Hotels", "Travels",
        "Finance", "Local Services", "Contractors", "Home Service", "Clothing",
        "Florists", "Makeup Artist", "Hospitals", "Delivery", "Other"
    ]

    # User profile input
    st.header("Your Profile")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('Enter your age:', min_value=0, max_value=100, value=30)
        gender = st.selectbox('Select your gender:', ['F', 'M', 'N'])
    with col2:
        latitude = st.number_input('Enter your latitude:', value=-1.9441)
        longitude = st.number_input('Enter your longitude:', value=30.0619)

    # Add category selection
    selected_category = st.selectbox('Select a business category:', categories)

    # Recommendation type selection
    st.header("Choose Your Recommendation Criteria")
    rec_type = st.radio(
        "What's most important to you?",
        ("Nearby Businesses", "Highly Rated Businesses", "Most Recommended Businesses", "Best Match for Your Profile")
    )

    if st.button('Get Recommendations'):
        try:
            # Base recommendations
            base_recommendations = recommend_businesses(age, gender, latitude, longitude, selected_category, df_updated, kmeans_updated, top_businesses_updated)
            
            # Filter df_updated based on selected category
            if selected_category != 'All':
                df_filtered = df_updated[df_updated['categories'].str.contains(selected_category, case=False, na=False)]
            else:
                df_filtered = df_updated

            if rec_type == "Nearby Businesses":
                nearby_df = df_filtered.sort_values('shortest_distance').head(10)
                
                # Create a map centered on the user's location
                m = folium.Map(location=[latitude, longitude], zoom_start=12)
                
                # Add user marker
                folium.Marker([latitude, longitude], popup="Your Location", icon=folium.Icon(color='red')).add_to(m)
                
                # Add business markers
                for idx, row in nearby_df.iterrows():
                    folium.Marker([row['bus_lat'], row['bus_log']], 
                                  popup=f"{row['business_name']} - {row['shortest_distance']:.2f} km").add_to(m)
                
                # Display the map
                folium_static(m)
                
                # Display recommendations
                st.subheader("Nearest Businesses:")
                for i, (idx, row) in enumerate(nearby_df.iterrows(), 1):
                    st.write(f"**{i}. {row['business_name']}** (Distance: {row['shortest_distance']:.2f} km)")

            elif rec_type == "Highly Rated Businesses":
                top_rated = df_filtered.sort_values('composite_score', ascending=False).head(5)
                st.subheader("Highest Rated Businesses:")
                for i, (idx, row) in enumerate(top_rated.iterrows(), 1):
                    st.write(f"**{i}. {row['business_name']}** (Composite Score: {row['composite_score']:.2f})")

            elif rec_type == "Most Recommended Businesses":
                most_recommended = df_filtered.sort_values('rec_score', ascending=False).head(5)
                st.subheader("Most Recommended Businesses:")
                for i, (idx, row) in enumerate(most_recommended.iterrows(), 1):
                    st.write(f"**{i}. {row['business_name']}** (Recommendation Score: {row['rec_score']:.2f})")

            else:  # Best Match for Your Profile
                recommendations = pd.Series({business: 5 for business in base_recommendations})
                st.subheader("Best Matches for Your Profile:")
                for i, business in enumerate(recommendations.index, 1):
                    st.write(f"**{i}. {business}**")

            # Additional information about the recommendations
            # st.write("")
            # st.write("These recommendations are personalized based on your profile and selected criteria.")
            # st.write(f"- Age: {age}")
            # st.write(f"- Gender: {gender}")
            # st.write(f"- Location: Latitude {latitude:.4f}, Longitude {longitude:.4f}")
            # st.write(f"- Category: {selected_category}")
            # st.write(f"- Recommendation focus: {rec_type}")

        except Exception as e:
            st.error(f"An error occurred while getting recommendations: {str(e)}")
            st.write("Exception details:", e)
            st.write("DataFrame columns:", df_updated.columns)
            if 'recommendations' in locals():
                st.write("Recommendations:", recommendations)

    # Add some space at the bottom
    st.write("")
    st.write("")
    
    # Optional: Add a note or disclaimer
    st.caption("Note: Recommendations are based on available data and may vary. Always check the latest information before making decisions.")
    
elif page == 'Recommendation Chatbot':
    chatbot_recommendation(df_updated, kmeans_updated, top_businesses_updated)



# Main execution
if __name__ == "__main__":
    #st.set_page_config(page_title="Yelp Business Recommendation System", page_icon="🍽️", layout="wide")
    
    # st.title(" Business Recommendation System")
    # st.write("Welcome to our business recommendation system. Choose a page from the sidebar to get started!")

    # You can add any additional main page content here
    # st.write("This system uses data from Yelp to provide personalized business recommendations.")
    # st.write("Use the sidebar to navigate between different analysis and recommendation options.")

    # You might want to add some instructions or a brief overview of each page
    # st.subheader("Page Descriptions:")
    # st.write("- **User Analysis**: Explore demographic information and user behavior.")
    # st.write("- **Business Analysis**: Analyze business ratings, reviews, and trends.")
    # st.write("- **Business Recommendations**: Get personalized business recommendations based on your profile.")
    # st.write("- **Recommendation Chatbot**: Interact with our chatbot for tailored recommendations.")

    # You could also add some sample visualizations or key metrics on the main page
    # For example:
    # st.subheader("Quick Stats:")
    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     st.metric("Total Users", f"{len(df_original['user_name'].unique()):,}")
    # with col2:
    #     st.metric("Total Businesses", f"{len(df_original['business_name'].unique()):,}")
    # with col3:
    #     st.metric("Average Rating", f"{df_original['stars'].mean():.2f}")

    # Add a footer
    st.markdown("---")
    st.caption("© Business Recommendation System")