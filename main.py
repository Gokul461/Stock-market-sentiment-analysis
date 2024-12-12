from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

finviz_url = "https://finviz.com/quote.ashx?t="
companies =[]
no = int(input("Enter the number of stock for analysis: "))
for i in range(no):
    print("Enter a Stock name: ")
    n = input()
    companies.append(n)

news_data = {}
for com in companies:
    url = finviz_url + com
    req = Request(url=url, headers={'user-agent': 'my-analysis'})
    try:
        response = urlopen(req)
        html = BeautifulSoup(response, 'html.parser')
        news_table = html.find(id='news-table')
        if news_table:
            news_data[com] = news_table
        else:
            print(f"News table not found for {com}")
    except Exception as e:
        print(f"Error fetching data for {com}: {e}")

parsed_data = []
for com, news_table in news_data.items():
    for row in news_table.findAll('tr'):
        title = row.a.text.strip()
        date_data = row.td.text.strip().split(' ')
        if len(date_data) == 1:
            time = date_data[0]
            date = None
        else:
            date, time = date_data[0], date_data[1]
        parsed_data.append([com, date, time, title])

df = pd.DataFrame(parsed_data, columns=['Company', 'Date', 'Time', 'Title'])

# Sentiment Analysis
vader = SentimentIntensityAnalyzer()
f = lambda title: vader.polarity_scores(title)['compound']
df['compound'] = df['Title'].apply(f)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date


grouped = df.groupby(['Company', 'Date'])['compound'].mean().reset_index()

# Decision logic
def sentiment_to_action(sentiment):
    if sentiment > 0.3:
        return 'Buy'
    elif sentiment < -0.3:
        return 'Sell'
    else:
        return 'Hold'

grouped['Action'] = grouped['compound'].apply(sentiment_to_action)

# Save results
grouped.to_csv('stock_sentiment_recommendations.csv', index=False)
print("Stock sentiment recommendations saved to stock_sentiment_recommendations.csv")

# Display results
print(grouped)

for company in companies:
    company_data = grouped[grouped['Company'] == company]
    plt.plot(company_data['Date'], company_data['compound'], label=company)

plt.axhline(y=0.3, color='green', linestyle='--', label='Buy Threshold')
plt.axhline(y=-0.3, color='red', linestyle='--', label='Sell Threshold')
plt.title('Sentiment Scores Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
