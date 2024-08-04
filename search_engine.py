import os
import random
import requests
import pyttsx3
import datetime
from num2words import num2words

# Open the text file for reading
with open('predicted_words.txt', 'r') as file:
    # Read the contents of the file and create a set of unique words
    predicted_words = set(word.strip() for word in file.readlines())

# Sort the predicted words and join them into a single query string
query = '+'.join(sorted(predicted_words))

# Check if the query is for the current time
if query.lower() in ['what+time+now', 'now+time+what']:
    # Get the current time in India using an NTP server
    response = requests.get('http://worldtimeapi.org/api/timezone/Asia/Kolkata')
    if response.status_code == 200:
        current_time = response.json()['datetime']
        time_parts = current_time.split('T')[1][:5].split(':')  # Extract only the time part and split into hours and minutes
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        
        # Convert hours and minutes to spoken words
        spoken_time = num2words(hours) + " " + ("o'clock" if minutes == 0 else num2words(minutes) + " minutes past") + " " + ("noon" if hours == 12 else "am" if hours < 12 else "pm")
        
        print("Current time in India:", spoken_time)
        
        engine = pyttsx3.init()
        engine.say("The current time in India is " + spoken_time)
        engine.runAndWait()
    else:
        print("Failed to retrieve current time.")
else:
    # Use Google Custom Search API for other queries
    API_KEY = "AIzaSyB-TxVhT_LQ7tH8w3RHfyoHJ9fTjBjZhh4"
    SEARCH_ENGINE_ID = "c32858bdfd68d4fc1"
    num_results = 1

    # Perform the prediction and get the query string
    url = f"https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}&num={num_results}"

    data = requests.get(url).json()

    search_items = data.get("items", [])
    if search_items:
        search_item = search_items[0]
        try:
            long_description = search_item["pagemap"]["metatags"][0]["og:description"]
        except KeyError:
            long_description = "N/A"

        title = search_item.get("title")
        snippet = search_item.get("snippet")
        html_snippet = search_item.get("htmlSnippet")
        link = search_item.get("link")
        print(snippet)

        engine = pyttsx3.init()
        engine.say(snippet)
        engine.runAndWait()
    else:
        print("No search results found.")

