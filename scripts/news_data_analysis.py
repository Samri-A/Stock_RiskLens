import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textblob as tb
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
print("Downloading NLTK resources...")
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
print("NLTK resources downloaded successfully.")
class NewsDataAnalysis:

    """ A class for analyzing financial news headlines and articles, with a focus on
        sentiment, topics, and publication trends """
    def __init__(self, path):
        try:
            self.df = pd.read_csv(path)
            if "Unnamed: 0" in self.df.columns:
                self.df.drop(columns=["Unnamed: 0"], inplace=True)
            if "date" in self.df.columns:
                self.df["date"] = pd.to_datetime(self.df["date"], errors='coerce' , format='ISO8601')
        except Exception as e:
            print(f"Error reading the CSV file: {e}")
        self.stop_words = set(stopwords.words('english'))
        self.punct = set(string.punctuation)
        self.topic_keywords = {
              "FDA Approval": ["fda", "approval", "approved", "clearance", "authorization"],
              "Price Target": ["price", "target", "pt", "raises", "cuts", "upgrade", "downgrade"],
              "Earnings Report": ["earnings", "eps", "revenue", "quarter", "results", "guidance", "forecast"],
              "Stock Movement": ["stock", "jumps", "drops", "soars", "plunges", "rises", "falls", "plummets"],
              "Mergers & Acquisitions": ["merger", "acquisition", "acquire", "buyout", "takeover", "joint venture"],
              "Product Launch": ["launch", "unveil", "introduce", "product", "new", "release", "debut"],
              "Legal or Regulatory": ["lawsuit", "fine", "investigation", "regulatory", "settlement"],
              "Partnerships / Collaborations": ["partnership", "collaboration", "deal", "alliance", "joint venture"],
              "Operations / Production": ["shipment", "supply", "production", "factory", "manufacturing", "shortage"]
          }
        


    def filter_by_stock(self, stock_symbol):
        self.df = self.df[self.df["stock"] == stock_symbol]

    def get_sentiment(self, text):
        analysis = tb.TextBlob(text)
        return analysis.sentiment.polarity
    def assign_topic(self ,text):
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word not in self.stop_words and word not in self.punct]
        for topic , keywords in self.topic_keywords.items():
            if any(keyword in tokens for keyword in keywords):
                return topic
        return "Other"

    def sentiment_analysis(self):
        self.df["sentiment_score"] = self.df["headline"].apply(self.get_sentiment)
        self.df["sentiment_category"] = self.df["sentiment_score"].apply(lambda x: "positive" if x > 0 else "neutral" if x == 0 else "negative")
        plt.figure(figsize=(10, 6))
        self.df["sentiment_category"].value_counts().plot(kind='bar', title="Sentiment Score of Headlines Distribution", ylabel='Number of Articles', xlabel='Sentiment Category')
        plt.show()
        self.df["headline_length"] = self.df["headline"].apply(lambda x: len(x))

    def topic_analysis(self):
        text_data = self.df["headline"].fillna("").astype(str)
        self.df["topic"] = text_data.apply(self.assign_topic)
        self.df["topic"].value_counts().plot(kind="bar", title="News Publication Topic Distribution" , xlabel="Topic", ylabel="Number of Articles")
        plt.show()

    def publication_timeseries_analysis(self):
        print("Publication timeseries analysis")   

        self.df["hour_of_day"] = self.df['date'].dt.hour
        group_by_hour = self.df.groupby("hour_of_day").size()
        plt.figure(figsize=(12, 6))
        group_by_hour.plot(kind="bar", title="Publication Count by Hour of Day", ylabel="Number of Articles", xlabel="Hour of Day")
        plt.show()

        self.df["day_of_week"] = self.df['date'].dt.day_name()
        group_by_day = self.df.groupby("day_of_week").size()
        plt.figure(figsize=(12, 6))
        group_by_day.plot(kind="bar", title="Publication Count by Day of Week", ylabel="Number of Articles", xlabel="Day of Week")
        plt.show()

       
        self.df["date"].resample("M").size().plot(title="Monthly Publication Trend")
        plt.xlabel("Date")
        plt.ylabel("Number of Articles")
        plt.show()

        self.df["date"].resample("Y").size().plot(title="Yearly Publication Trend")
        plt.xlabel("Date")
        plt.ylabel("Number of Articles")
        plt.show()

    def publisher_analysis(self):
        print("Publisher statistics:")
        print(self.df["publisher"].describe())

        publisher_counts = self.df["publisher"].value_counts().sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=publisher_counts.head(7).index, y=publisher_counts.head(7).values, palette='viridis')
        plt.title('Publisher Articles Analysis')
        plt.xlabel('Publishers')
        plt.ylabel('Number of Articles')
        plt.show()

    def organizations_publication(self):
        print("Organizations publication analysis:")
        self.df["domain"] = self.df["publisher"].apply(lambda x: x.split("@")[-1] if "@" in x else "None")

        plt.figure(figsize=(12, 6))
        self.df[(self.df["domain"] != "None") & (self.df["domain"] != "gmail.com")]["domain"].value_counts().plot(
            kind="bar", title="Company Publication Distribution" , ylabel='Number of Articles', xlabel= 'Companies')
        plt.show()

    def save_data(self , name):
        self.df.to_csv(f"../src/data/processed/{name}.csv", index=False)

