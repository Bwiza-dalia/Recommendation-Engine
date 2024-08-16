import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from shapely.geometry import Point
from pyproj import CRS

# Load your data
df = pd.read_csv('new_yelp_dataset_combined.csv')

# Load Rwanda districts shapefile (you'll need to obtain this)
# rwanda_districts = gpd.read_file('district.shp')

rwanda_districts = gpd.read_file('district.shp')

# Create a DataFrame with unique businesses
unique_businesses = df.drop_duplicates(subset='business_id')
print(f"Unique businesses after deduplication: {len(unique_businesses)}")

# print(f"Total rows in df: {len(df)}")
# print(f"Unique business IDs: {df['business_id'].nunique()}")
# print(f"Unique business addresses: {df['business_address'].nunique()}")

# # Check for null values
# print(f"Null values in business_id: {df['business_id'].isnull().sum()}")
# print(f"Null values in business_address: {df['business_address'].isnull().sum()}")

# # Print a few rows where business_id and business_address don't match up
# mismatched = df[df.duplicated(subset='business_id', keep=False) & ~df.duplicated(subset=['business_id', 'business_address'], keep=False)]
# print("\nSample of mismatched business IDs and addresses:")
# print(mismatched[['business_id', 'business_address']].head())


# print("Columns in rwanda_districts:")
# print(rwanda_districts.columns)
# print("Information about rwanda_districts:")
# print(rwanda_districts.info())
# print("\nFirst few rows of rwanda_districts:")
# print(rwanda_districts.head())


# Get the CRS of your districts
districts_crs = rwanda_districts.crs

# Create a DataFrame with unique users
unique_users = df.drop_duplicates(subset='user_id')
print(f"Unique users after deduplication: {len(unique_users)}")

# Create GeoDataFrame for users
user_gdf = gpd.GeoDataFrame(
    unique_users,
    geometry=[Point(map(float, addr.split(', ')[::-1])) for addr in unique_users['user_address']],
    crs=CRS.from_epsg(4326)
).to_crs(districts_crs)

print(f"Rows in user_gdf: {len(user_gdf)}")

# Create GeoDataFrame for businesses
business_gdf = gpd.GeoDataFrame(
    unique_businesses,
    geometry=[Point(map(float, addr.split(', ')[::-1])) for addr in unique_businesses['business_address']],
    crs=CRS.from_epsg(4326)
).to_crs(districts_crs)

print(f"Rows in business_gdf: {len(business_gdf)}")

# Create base map
m = folium.Map(location=[-1.9403, 29.8739], zoom_start=8)

# Add district boundaries
folium.GeoJson(
    rwanda_districts,
    style_function=lambda feature: {
        'fillColor': 'lightblue',
        'color': 'black',
        'weight': 2,
        'fillOpacity': 0.7,
    }
).add_to(m)

# 1. Show user locations
user_cluster = MarkerCluster(name="Users").add_to(m)

for _, row in df.drop_duplicates(subset='user_id').iterrows():
    user_lat, user_lon = map(float, row['user_address'].split(', '))
    folium.Marker(
        location=[user_lat, user_lon],
        popup=f"User: {row['user_name']}",
        icon=folium.Icon(color='blue', icon='user', prefix='fa')
    ).add_to(user_cluster)

# 2. Show business locations
business_cluster = MarkerCluster(name="Businesses").add_to(m)

for _, row in df.drop_duplicates(subset='business_id').iterrows():
    business_lat, business_lon = map(float, row['business_address'].split(', '))
    folium.Marker(
        location=[business_lat, business_lon],
        popup=f"Business: {row['business_name']}<br>Reviews: {row['review_count_business']}",
        icon=folium.Icon(color='red', icon='store', prefix='fa')
    ).add_to(business_cluster)

# 3 & 4. Create choropleth layers for user and business density
# You'll need to assign districts to each user and business based on their coordinates
# This is a placeholder - you'll need to implement the district assignment logic
df['user_district'] = 'placeholder'
df['business_district'] = 'placeholder'

# Assign districts to users
users_with_district = gpd.sjoin(user_gdf, rwanda_districts, how="left", op="within")

# Count users per district
users_per_district = users_with_district.groupby('District').size().reset_index(name='user_count')

# Assign districts to businesses
businesses_with_district = gpd.sjoin(business_gdf, rwanda_districts, how="left", op="within")

# Count businesses per district
businesses_per_district = businesses_with_district.groupby('District').size().reset_index(name='business_count')

# Merge counts with district data
rwanda_districts = rwanda_districts.merge(users_per_district, on='District', how='left')
rwanda_districts = rwanda_districts.merge(businesses_per_district, on='District', how='left')
rwanda_districts['user_count'] = rwanda_districts['user_count'].fillna(0)
rwanda_districts['business_count'] = rwanda_districts['business_count'].fillna(0)

# Find businesses with multiple addresses
businesses_with_multiple_addresses = df[df.duplicated(subset='business_id', keep=False)].sort_values('business_id')
print("\nBusinesses with multiple addresses:")
print(businesses_with_multiple_addresses[['business_id', 'business_address']].head(10))


# Create base map
m = folium.Map(location=[-1.9403, 29.8739], zoom_start=8)

# Add district boundaries with user count choropleth
folium.Choropleth(
    geo_data=rwanda_districts,
    name='User Density',
    data=rwanda_districts,
    columns=['District', 'user_count'],
    key_on='feature.properties.District',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Number of Users'
).add_to(m)

# Add user markers
user_cluster = MarkerCluster(name="Users").add_to(m)

for _, row in user_gdf.iterrows():
    folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        popup=f"User: {row['user_name']}",
        icon=folium.Icon(color='blue', icon='user', prefix='fa')
    ).add_to(user_cluster)

# Add layer control and save the map
folium.LayerControl().add_to(m)
m.save("rwanda_new_user_map.html")