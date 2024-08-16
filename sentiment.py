import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Read the data
data = pd.read_csv('review.csv') 

# Combine all text into a single string
text = ' '.join(data['text'].astype(str))

# Tokenize the text
tokens = word_tokenize(text)

# Remove stopwords and convert to lowercase
stop_words = set(stopwords.words('english'))
cleaned_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]

# Join the cleaned tokens back into a string
cleaned_text = ' '.join(cleaned_tokens)

# Create and generate a word cloud image
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)

# Display the generated image
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Reviews')
plt.show()

# Save the word cloud image
wordcloud.to_file('wordcloud.png')