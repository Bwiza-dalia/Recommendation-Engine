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
from streamlit import config
from PIL import Image
import streamlit.components.v1 as components
import datetime
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from PIL import Image




# # Set page config at the very beginning
st.set_page_config(page_title="Data analytics", page_icon="🍽️", layout="wide")


# Functions for data processing and recommendations
def extract_lat_lon(address):
    lat, lon = map(float, address.split(','))
    return lat, lon

def get_top_businesses(group, n=5):
    return group['business_name'].value_counts().nlargest(n).index.tolist()

def load_wordcloud():
    try:
        return Image.open('wordcloud.png')
    except FileNotFoundError:
        st.error("Word cloud image not found. Please run sentiment.py first.")
        return None
    


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

    # Display QR Code for the chatbot URL
    chatbot_url = "https://myfirstap.streamlit.app/?view=chatbot"
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(chatbot_url)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')

    # Display QR Code on the dashboard
    st.image(img, caption="Scan to interact with the chatbot")

    if st.experimental_get_query_params().get('view', [None])[0] == 'chatbot':
        # Your existing chatbot code
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
    else:
        st.write("Please use the QR code to access the chatbot interface.")
        

# Load and prepare data
@st.cache_data(ttl=3600)
def load_data():
    try:
        df_original = pd.read_csv('the_dashboard_dataset.csv')
        # st.write("First few rows of the loaded data:", df_original.head())
        required_columns_original = ['user_name', 'user_ages', 'date','gender', 'user_address', 'stars', 'state', 'rec_score', 'business_name']
        for col in required_columns_original:
            if col not in df_original.columns:
                raise ValueError(f"Required column '{col}' not found in the original dataset")
        df_original = df_original.rename(columns={'potential_ages': 'user_ages'})
        df_original['date'] = pd.to_datetime(df_original['date'])

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
page = st.sidebar.radio('Go to', [ 'User Analysis', 'Business Analysis', 'Institution Analysis'])

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
                                color_discrete_sequence=['fuchsia'], zoom=7, height=300)
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
    institutions = ['All'] + sorted(df_original['business_name'].unique().tolist())
    selected_institution = st.selectbox('Select Institution', institutions)

    if selected_institution != 'All':
        filtered_df = df_original[df_original['business_name'] == selected_institution]
        st.write(f"Showing data for: {selected_institution}")
    else:
        filtered_df = df_original
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
    if 'service_name' in filtered_df.columns:
        top_services = filtered_df.groupby('service_name')['stars'].mean().sort_values(ascending=False).head(10)
        fig_top_services = px.bar(top_services.reset_index(), x='service_name', y='stars',
                                  title='Top 10 Rated Services')
        fig_top_services.update_layout(xaxis_title='Service', yaxis_title='Average Rating')
        st.plotly_chart(fig_top_services)
    else:
        st.write("Service name information not available in the dataset.")

    # Service Quality Responses
    if 'state' in filtered_df.columns:
        fig_quality = px.pie(filtered_df['state'].value_counts().reset_index(), 
                             values='count', names='state', title='Distribution of Service Quality Responses')
        st.plotly_chart(fig_quality)
    else:
        st.write("Service quality state information not available in the dataset.")

    # User Activity by Location
    if 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns:
        fig_activity = px.density_mapbox(filtered_df, lat='latitude', lon='longitude', z='stars', radius=10,
                                         center=dict(lat=filtered_df['latitude'].mean(), lon=filtered_df['longitude'].mean()), zoom=6,
                                         mapbox_style="open-street-map", title='User Activity by Location')
        st.plotly_chart(fig_activity)
    else:
        st.write("Location information not available in the dataset.")

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
    else:
        st.write("Date information not available for time series analysis.")


elif page == 'Institution Analysis':
    st.title('Institution Analysis')

    

    # Date range selection
    st.subheader('Select Time Period')
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", min(df_original['date']))
    with col2:
        end_date = st.date_input("End date", max(df_original['date']))

    # Filter data based on selected date range
    mask = (df_original['date'] >= pd.Timestamp(start_date)) & (df_original['date'] <= pd.Timestamp(end_date))
    filtered_df = df_original.loc[mask]

    # Word cloud of business names
    st.subheader('Word Cloud of Institution Names')
    text = ' '.join(df_original['business_name'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    fig_wordcloud, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig_wordcloud)


    # Display the pre-generated map
    st.subheader('User and Institution Map')
    with open('rwanda_new_user_map.html', 'r') as f:
        map_html = f.read()
    components.html(map_html, height=600)

    # Add institution layer
    st.subheader('Institution Locations')
    institution_data = filtered_df[['latitude', 'longitude', 'business_name']]
    st.map(institution_data)

    # Institution Statistics
    st.subheader('Institution Statistics')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Institutions", f"{len(filtered_df['business_name'].unique()):,}")
    with col2:
        st.metric("Average Rating", f"{filtered_df['stars'].mean():.2f}")
    with col3:
        st.metric("Average Recommendation Score", f"{filtered_df['rec_score'].mean():.2f}")

    
    # Top rated institutions
    st.subheader('Top Rated Institutions')
    top_institutions = filtered_df.groupby('business_name').agg({
        'stars': 'mean',
        'business_name': 'count'
    }).rename(columns={'business_name': 'review_count'}).sort_values('stars', ascending=False).head(10)

    fig_top = px.bar(top_institutions.reset_index(), x='business_name', y='stars', 
                     title='Top 10 Rated Institutions',
                     text='stars',
                     hover_data=['review_count'])
    fig_top.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_top.update_layout(xaxis_title='Institution', yaxis_title='Average Rating')
    st.plotly_chart(fig_top)

    # Distribution of ratings
    st.subheader('Distribution of Ratings')
    fig_dist = px.histogram(filtered_df, x='stars', nbins=20, 
                            title='Distribution of Institution Ratings')
    fig_dist.update_layout(xaxis_title='Rating', yaxis_title='Count')
    st.plotly_chart(fig_dist)

    # Time series analysis of ratings
    st.subheader('Rating Trends Over Time')
    rating_trend = filtered_df.groupby(filtered_df['date'].dt.to_period('M')).agg({
        'stars': 'mean',
        'business_name': 'count'
    }).rename(columns={'business_name': 'review_count'}).reset_index()
    rating_trend['date'] = rating_trend['date'].dt.to_timestamp()

    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(go.Scatter(x=rating_trend['date'], y=rating_trend['stars'], name="Average Rating"), secondary_y=False)
    fig_trend.add_trace(go.Bar(x=rating_trend['date'], y=rating_trend['review_count'], name="Number of Reviews", opacity=0.5), secondary_y=True)
    fig_trend.update_layout(title='Average Rating and Review Count Over Time', xaxis_title='Date')
    fig_trend.update_yaxes(title_text="Average Rating", secondary_y=False)
    fig_trend.update_yaxes(title_text="Number of Reviews", secondary_y=True)
    st.plotly_chart(fig_trend)

   

    # Scatter plot of ratings vs. recommendation scores
    st.subheader('Ratings vs. Recommendation Scores')
    fig_scatter = px.scatter(filtered_df, x='stars', y='rec_score', 
                             hover_data=['business_name'],
                             title='Ratings vs. Recommendation Scores')
    fig_scatter.update_layout(xaxis_title='Rating', yaxis_title='Recommendation Score')
    st.plotly_chart(fig_scatter)

    # Institution categories
    st.subheader('Institution Categories')
    if 'categories' in filtered_df.columns:
        # Split categories and count occurrences
        category_counts = filtered_df['categories'].str.split(', ', expand=True).stack().value_counts()
        
        # Bar chart for top categories
        st.subheader('Top 10 Institution Categories')
        fig_top_cat = px.bar(category_counts.head(10), 
                             x=category_counts.head(10).index, 
                             y=category_counts.head(10).values,
                             labels={'x': 'Category', 'y': 'Count'},
                             title='Top 10 Institution Categories')
        fig_top_cat.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_top_cat)

        # Pie chart for overall distribution
        st.subheader('Overall Distribution of Institution Categories')
        fig_cat_pie = px.pie(values=category_counts.values, 
                             names=category_counts.index, 
                             title='Distribution of Institution Categories')
        fig_cat_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_cat_pie)


elif page == 'Institution Analysis':
    st.title('Institution Analysis')

    # Display the pre-generated map
    st.subheader('Institution Map')
    with open('rwanda_new_user_map.html', 'r') as f:
        map_html = f.read()
    components.html(map_html, height=600)

    # Institution Statistics
    st.subheader('Institution Statistics')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Institutions", f"{len(df_original['business_name'].unique()):,}")
    with col2:
        st.metric("Average Rating", f"{df_original['stars'].mean():.2f}")
    with col3:
        st.metric("Average Recommendation Score", f"{df_original['rec_score'].mean():.2f}")

    # Top rated institutions
    st.subheader('Top Rated Institutions')
    top_institutions = df_original.groupby('business_name').agg({
        'stars': 'mean',
        'business_name': 'count'
    }).rename(columns={'business_name': 'review_count'}).sort_values('stars', ascending=False).head(10)

    fig_top = px.bar(top_institutions.reset_index(), x='business_name', y='stars', 
                     title='Top 10 Rated Institutions',
                     text='stars',
                     hover_data=['review_count'])
    fig_top.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_top.update_layout(xaxis_title='Institution', yaxis_title='Average Rating')
    st.plotly_chart(fig_top)

    # Distribution of ratings
    st.subheader('Distribution of Ratings')
    fig_dist = px.histogram(df_original, x='stars', nbins=20, 
                            title='Distribution of Institution Ratings')
    fig_dist.update_layout(xaxis_title='Rating', yaxis_title='Count')
    st.plotly_chart(fig_dist)

    # Time series analysis of ratings
    st.subheader('Rating Trends Over Time')
    df_original['date'] = pd.to_datetime(df_original['date'])
    rating_trend = df_original.groupby(df_original['date'].dt.to_period('M')).agg({
        'stars': 'mean',
        'business_name': 'count'
    }).rename(columns={'business_name': 'review_count'}).reset_index()
    rating_trend['date'] = rating_trend['date'].dt.to_timestamp()

    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(go.Scatter(x=rating_trend['date'], y=rating_trend['stars'], name="Average Rating"), secondary_y=False)
    fig_trend.add_trace(go.Bar(x=rating_trend['date'], y=rating_trend['review_count'], name="Number of Reviews", opacity=0.5), secondary_y=True)
    fig_trend.update_layout(title='Average Rating and Review Count Over Time', xaxis_title='Date')
    fig_trend.update_yaxes(title_text="Average Rating", secondary_y=False)
    fig_trend.update_yaxes(title_text="Number of Reviews", secondary_y=True)
    st.plotly_chart(fig_trend)

    # Correlation heatmap
    st.subheader('Correlation Between Metrics')
    corr_columns = ['stars', 'rec_score']
    if 'composite_score' in df_original.columns:
        corr_columns.append('composite_score')
    if 'shortest_distance' in df_original.columns:
        corr_columns.append('shortest_distance')
    corr_matrix = df_original[corr_columns].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                         title='Correlation Heatmap of Key Metrics')
    st.plotly_chart(fig_corr)

    # Scatter plot of ratings vs. recommendation scores
    st.subheader('Ratings vs. Recommendation Scores')
    fig_scatter = px.scatter(df_original, x='stars', y='rec_score', 
                             hover_data=['business_name'],
                             title='Ratings vs. Recommendation Scores')
    fig_scatter.update_layout(xaxis_title='Rating', yaxis_title='Recommendation Score')
    st.plotly_chart(fig_scatter)

    # Institution categories
    st.subheader('Institution Categories')
    if 'categories' in df_original.columns:
        category_counts = df_original['categories'].str.split(', ', expand=True).stack().value_counts()
        fig_cat = px.pie(values=category_counts.values, names=category_counts.index, 
                         title='Distribution of Institution Categories')
        st.plotly_chart(fig_cat)
    else:
        st.write("Category information not available in the dataset.")

    # Word cloud of business names
    st.subheader('Word Cloud of Institution Names')
    text = ' '.join(df_original['business_name'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    fig_wordcloud, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig_wordcloud)



# Main execution
if __name__ == "__main__":

    # Add a footer
    st.markdown("---")
