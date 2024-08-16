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

# Functions for data processing and recommendations
def extract_lat_lon(address):
    lat, lon = map(float, address.split(','))
    return lat, lon

def get_top_businesses(group, n=5):
    return group['business_name'].value_counts().nlargest(n).index.tolist()

@st.cache_resource
def prepare_recommendation_data(df):
    df['latitude'], df['longitude'] = zip(*df['user_address'].apply(extract_lat_lon))
    df['age_group'] = pd.cut(df['user_ages'], bins=[0, 18, 25, 35, 45, 55, 65, 100], labels=['0-18', '19-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    
    coords = df[['latitude', 'longitude']].values
    kmeans = KMeans(n_clusters=10, random_state=42)  # Increase number of clusters
    df['location_cluster'] = kmeans.fit_predict(coords)
    
    grouped = df.groupby(['age_group', 'gender', 'location_cluster'])
    top_businesses = grouped.apply(get_top_businesses).reset_index()
    top_businesses.columns = ['age_group', 'gender', 'location_cluster', 'top_businesses']
    
    return df, kmeans, top_businesses

def recommend_businesses(age, gender, lat, lon, df, kmeans, top_businesses):
    age_group = pd.cut([age], bins=[0, 18, 25, 35, 45, 55, 65, 100], labels=['0-18', '19-25', '26-35', '36-45', '46-55', '56-65', '65+'])[0]
    location_cluster = kmeans.predict([[lat, lon]])[0]

    st.write(f"Searching for: Age Group: {age_group}, Gender: {gender}, Location Cluster: {location_cluster}")

    # First, try to find an exact match
    recommendations = top_businesses[
        (top_businesses['age_group'] == age_group) &
        (top_businesses['gender'] == gender) &
        (top_businesses['location_cluster'] == location_cluster)
    ]

    if recommendations.empty:
        st.write("No exact match found. Trying broader search...")
        # If no exact match, try matching only age_group and gender
        recommendations = top_businesses[
            (top_businesses['age_group'] == age_group) &
            (top_businesses['gender'] == gender)
        ]
    
    if recommendations.empty:
        st.write("Still no match found. Returning overall top businesses.")
        # If still no match, return overall top businesses
        return df['business_name'].value_counts().nlargest(5).index.tolist()
    else:
        # Return the top businesses for the matched group
        return recommendations.iloc[0]['top_businesses']
    
def filter_by_institution(df, institution):
    return df[df['business_name'] == institution]    

def chatbot_recommendation(df, kmeans, top_businesses):
    st.title("Recommendation Chatbot")

    # Initialize chat history and user info
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_info" not in st.session_state:
        st.session_state.user_info = {"age": None, "gender": None, "latitude": None, "longitude": None}

    # Display chat messages from history on rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Function to get recommendation based on user info
    def get_recommendation():
        age = st.session_state.user_info["age"]
        gender = st.session_state.user_info["gender"]
        lat = st.session_state.user_info["latitude"]
        lon = st.session_state.user_info["longitude"]
        recommendations = recommend_businesses(age, gender, lat, lon, df, kmeans, top_businesses)
        return f"Based on your information, I recommend these businesses: {', '.join(recommendations[:3])}"

    # Chatbot logic
    if not st.session_state.messages:
        response = "Hello! I can help you find business recommendations. First, could you tell me your age?"
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    # React to user input
    if prompt := st.chat_input("Your response"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process user input based on current state
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
                response = "Great! Finally, what's your longitude?"
            except ValueError:
                response = "Please enter a valid latitude as a number."
        elif st.session_state.user_info["longitude"] is None:
            try:
                lon = float(prompt)
                st.session_state.user_info["longitude"] = lon
                response = get_recommendation()
            except ValueError:
                response = "Please enter a valid longitude as a number."
        else:
            response = "I've already given you recommendations. Would you like to start over? (Yes/No)"
            if prompt.lower() == 'yes':
                st.session_state.user_info = {"age": None, "gender": None, "latitude": None, "longitude": None}
                response += " Great! Let's start again. What's your age?"

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


# Load and prepare data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('new_yelp_dataset_combined.csv')
        required_columns = ['user_name', 'user_ages', 'gender', 'user_address', 'stars', 'state', 'rec_score', 'business_name']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in the dataset")
        df = df.rename(columns={'potential_ages': 'user_ages'})
        df, kmeans, top_businesses = prepare_recommendation_data(df)
        return df, kmeans, top_businesses
    except FileNotFoundError:
        st.error("The file 'yelp_dataset_combined.csv' was not found. Please ensure it's in the same directory as this script.")
        st.stop()
    except ValueError as e:
        st.error(f"Error in dataset: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.stop()

# Load data
try:
    df, kmeans, top_businesses = load_data()
except Exception as e:
    st.error(f"Failed to load data: {str(e)}")
    st.stop()

# Create a sidebar for navigation
st.sidebar.title('menu')
page = st.sidebar.radio('Go to', ['User Analysis', 'Business Analysis', 'Business Recommendations', 'Recommendation Chatbot'])

if page == 'User Analysis':
    st.title('User Analysis Dashboard')

    # Age Distribution
    st.header('Age Distribution')
    fig_age = px.histogram(df, x='user_ages', nbins=20, title='Distribution of User Ages')
    fig_age.update_layout(xaxis_title='Age', yaxis_title='Frequency')
    st.plotly_chart(fig_age)

    # Gender Distribution
    st.header('Gender Distribution')
    gender_counts = df['gender'].value_counts()
    fig_gender = px.pie(values=gender_counts.values, names=gender_counts.index, title='Gender Distribution')
    st.plotly_chart(fig_gender)

    # User Map
    st.header('User Map')
    fig_map = px.scatter_mapbox(df, lat='latitude', lon='longitude', hover_name='user_name',
                                hover_data=['user_ages', 'gender'],
                                color_discrete_sequence=['fuchsia'], zoom=3, height=300)
    fig_map.update_layout(mapbox_style='open-street-map')
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map)

    # Average Review Scores by Age Group
    df['age_group'] = pd.cut(df['user_ages'], bins=[0, 20, 30, 40, 50, 60], labels=['0-20', '21-30', '31-40', '41-50', '51-60'])
    fig_age_review = px.bar(df.groupby('age_group')['stars'].mean().reset_index(), 
                            x='age_group', y='stars', title='Average Review Scores by Age Group')
    fig_age_review.update_layout(xaxis_title='Age Group', yaxis_title='Average Review Score')
    st.plotly_chart(fig_age_review)

    # Average Review Scores by Gender
    fig_gender_review = px.bar(df.groupby('gender')['stars'].mean().reset_index(), 
                               x='gender', y='stars', title='Average Review Scores by Gender')
    fig_gender_review.update_layout(xaxis_title='Gender', yaxis_title='Average Review Score')
    st.plotly_chart(fig_gender_review)

elif page == 'Business Analysis':
    st.title('Business Analysis Dashboard')

    # Add institution filter
    institutions = ['All'] + sorted(df['business_name'].unique().tolist())
    selected_institution = st.selectbox('Select Institution', institutions)

    if selected_institution != 'All':
        filtered_df = filter_by_institution(df, selected_institution)
        st.write(f"Showing data for: {selected_institution}")
    else:
        filtered_df = df
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
    fig_activity = px.density_mapbox(filtered_df, lat='latitude', lon='longitude', z='stars', radius=10,
                                     center=dict(lat=filtered_df['latitude'].mean(), lon=filtered_df['longitude'].mean()), zoom=3,
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

    # User profile input
    st.header("Your Profile")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('Enter your age:', min_value=0, max_value=100, value=30)
        gender = st.selectbox('Select your gender:', ['F', 'M', 'N'])
    with col2:
        latitude = st.number_input('Enter your latitude:', value=-1.9441)
        longitude = st.number_input('Enter your longitude:', value=30.0619)

    # Recommendation type selection
    st.header("Choose Your Recommendation Criteria")
    rec_type = st.radio(
        "What's most important to you?",
        ("Nearby Businesses", "Highly Rated Businesses", "Most Recommended Businesses", "Best Match for Your Profile")
    )

    if st.button('Get Personalized Recommendations'):
        try:
            # Base recommendations
            base_recommendations = recommend_businesses(age, gender, latitude, longitude, df, kmeans, top_businesses)
            
            if rec_type == "Nearby Businesses":
                # Filter for nearby businesses
                nearby_df = df[(df['latitude'] - latitude).abs() + (df['longitude'] - longitude).abs() < 0.1]
                recommendations = nearby_df.groupby('business_name')['stars'].mean().sort_values(ascending=False).head(5)

            elif rec_type == "Highly Rated Businesses":
                # Filter for highly rated businesses
                high_rated = df.groupby('business_name').agg({
                    'stars': 'mean',
                    'business_name': 'count'
                }).rename(columns={'business_name': 'review_count'})
                recommendations = high_rated[high_rated['review_count'] >= 10].sort_values('stars', ascending=False).head(5)

            elif rec_type == "Most Recommended Businesses":
                # Check if 'rec_score' column exists, if not use 'stars'
                score_column = 'rec_score' if 'rec_score' in df.columns else 'stars'
                most_recommended = df.groupby('business_name').agg({
                    score_column: 'mean',
                    'business_name': 'count'
                }).rename(columns={'business_name': 'review_count'})
                recommendations = most_recommended[most_recommended['review_count'] >= 10].sort_values(score_column, ascending=False).head(5)

            else:  # Best Match for Your Profile
                recommendations = pd.Series({business: 5 for business in base_recommendations})

            st.subheader("Top 5 Recommended Businesses for You:")
            for i, (business, score) in enumerate(recommendations.items(), 1):
                if rec_type == "Nearby Businesses":
                    st.write(f"**{i}. {business}** (Average Rating: {score:.2f} ★)")
                elif rec_type == "Highly Rated Businesses":
                    review_count = high_rated.loc[business, 'review_count']
                    st.write(f"**{i}. {business}** (Average Rating: {score:.2f} ★, Reviews: {review_count})")
                elif rec_type == "Most Recommended Businesses":
                    review_count = most_recommended.loc[business, 'review_count']
                    score_type = "Recommendation Score" if 'rec_score' in df.columns else "Average Rating"
                    st.write(f"**{i}. {business}** ({score_type}: {score:.2f}, Reviews: {review_count})")
                else:
                    st.write(f"**{i}. {business}**")

            # Additional information about the recommendations
            st.write("")
            st.write("These recommendations are personalized based on your profile and selected criteria.")
            st.write(f"- Age: {age}")
            st.write(f"- Gender: {gender}")
            st.write(f"- Location: Latitude {latitude:.4f}, Longitude {longitude:.4f}")
            st.write(f"- Recommendation focus: {rec_type}")

        except Exception as e:
            st.error(f"An error occurred while getting recommendations: {str(e)}")
            st.write("Exception details:", e)
            st.write("DataFrame columns:", df.columns)  # For debugging
            if 'recommendations' in locals():
                st.write("Recommendations:", recommendations)  # For debugging

    # Add some space at the bottom
    st.write("")
    st.write("")
    
    # Optional: Add a note or disclaimer
    st.caption("Note: Recommendations are based on available data and may vary. Always check the latest information before making decisions.")

elif page == 'Recommendation Chatbot':
    chatbot_recommendation(df, kmeans, top_businesses)            

                        