from crewai import Agent, Task, Crew, LLM
from googleapiclient.discovery import build 
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import re

# Load environment variables
load_dotenv()
api_key = os.getenv('API_KEY')

# Define Ollama LLM
ollama_llm = LLM(
    model="ollama/tinyllama",  
    provider="ollama",
    base_url="http://localhost:11434"
)

# Fetch comments from YouTube
def fetch_youtube_comments(video_id, max_comments=20):
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    response = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_comments,
        textFormat="plainText"        
    ).execute()

    for item in response.get("items", []):
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)

    return "\n".join([f"{i+1}. {text}" for i, text in enumerate(comments)])

# Get channel name from video ID
def get_channel_name(video_id):
    youtube = build("youtube", "v3", developerKey=api_key)
    response = youtube.videos().list(
        part="snippet",
        id=video_id
    ).execute()
    return response["items"][0]["snippet"]["channelTitle"]

# Define agents
def create_agents(llm):
    return [
        Agent(
            role="Comment Formatter",
            goal="Prepare YouTube comments for sentiment analysis",
            backstory="You specialize in structuring user-generated content from video platforms for NLP tasks.",
            verbose=True,
            llm=llm
        ),
        Agent(
            role="Sentiment Analyst",
            goal="Analyze emotional tone in YouTube comments written in Hindi-English (Hindlish).",
            backstory=(
                "You are an NLP expert trained to detect emotional polarity in viewer feedback, especially in culturally adaptive Hindlish language. "
                "You understand sarcasm, slang, emojis, and mixed-language expressions common in Indian YouTube comments."
            ),
            verbose=True,
            llm=llm
        ),
        Agent(
            role="Trend Summarizer",
            goal="Summarize emotional trends and tone shifts in viewer comments",
            backstory="You are a communication strategist who distills audience sentiment into actionable insights.",
            verbose=True,
            llm=llm
        )
    ]

# Define tasks
def create_tasks(comments, agents):
    return [
        Task(
            description=f"Here are recent YouTube comments extracted from the video:\n{comments}\nFormat them for sentiment analysis.",
            expected_output="A clean list of comment lines ready for analysis.",
            agent=agents[0]
        ),
        Task(
            description="Analyze the emotional tone of each comment and classify them as positive, neutral, or negative.",
            expected_output="Sentiment scores and emotional tone for each comment with reasoning.",
            agent=agents[1]
        ),
        Task(
            description="Summarize the overall emotional trend and describe how the audience's tone evolves across these comments.",
            expected_output="A concise summary of emotional patterns and tone shifts.",
            agent=agents[2]
        )
    ]

# List of video IDs
video_ids = [
    "465iEeNOEBg", "a4JEu2Y84yQ", "h0hVkfD_omk", "6oW6kzS3S90", "hXebwJGaERM",
    "ZdECSfn-qTc", "9kfScGV6W1Y", "33BR5Vhcfcc", "8_Bx7HRo_ms", "PvUx3Zjh3zE"
]

# Track sentiment counts
creators = []
positive_counts = []
negative_counts = []
neutral_counts = []

# Run sentiment analysis
for video_id in video_ids:
    try:
        channel_name = get_channel_name(video_id)
        creators.append(channel_name)
        print(f"\n Channel Name: {channel_name} - Video ID: {video_id}")

        comments = fetch_youtube_comments(video_id)
        if not comments:
            print("⚠️ No comments found.")
            positive_counts.append(0)
            negative_counts.append(0)
            neutral_counts.append(0)
            continue

        agents = create_agents(ollama_llm)
        tasks = create_tasks(comments, agents)

        crew = Crew(
            agents=agents,
            tasks=tasks,
            llm=ollama_llm
        )

        result = crew.kickoff()

        # Safely convert CrewOutput to string
        output_text = str(result)

        print(f"\n📊 Sentiment Summary for {channel_name}:\n{output_text}")

        # Loosened sentiment matching
        pos = output_text.lower().count("positive")
        neg = output_text.lower().count("negative")
        neu = output_text.lower().count("neutral")

        positive_counts.append(pos)
        negative_counts.append(neg)
        neutral_counts.append(neu)

    except Exception as e:
        print(f"❌ Error processing video {video_id}: {type(e).__name__} - {e}")
        creators.append(f"Unknown-{video_id}")
        positive_counts.append(0)
        negative_counts.append(0)
        neutral_counts.append(0)

# Align lengths to avoid shape mismatch
# min_len = min(len(creators), len(positive_counts), len(negative_counts), len(neutral_counts))
# creators = creators[:min_len]
# positive_counts = positive_counts[:min_len]
# negative_counts = negative_counts[:min_len]
# neutral_counts = neutral_counts[:min_len]
# x = np.arange(min_len)

# # Plot sentiment chart
# width = 0.25
# plt.figure(figsize=(18, 6))

# # Shift bars so the group is centered on each creator
# bars1 = plt.bar(x - width, positive_counts, width, label='Positive', color='green')
# bars2 = plt.bar(x, neutral_counts, width, label='Neutral', color='yellow')
# bars3 = plt.bar(x + width, negative_counts, width, label='Negative', color='red')

# # Add value labels on bars
# for bar_group in [bars1, bars2, bars3]:
#     for bar in bar_group:
#         height = bar.get_height()
#         if height > 0:
#             plt.text(bar.get_x() + bar.get_width()/2, height + 0.1, str(height),
#                      ha='center', va='bottom', fontsize=8)

# # Center x-tick labels under the group of bars
# plt.xticks(x, creators, rotation=45, ha='center')

# # Add padding and spacing
# plt.xlabel('YouTube Creators')
# plt.ylabel('Comment Counts (out of 20)')
# plt.title('Sentiment Analysis of 20 YouTube Comments per Creator')
# plt.ylim(0, max(max(positive_counts), max(negative_counts), max(neutral_counts)) + 2)
# plt.grid(axis='y', linestyle='--', alpha=0.5)
# plt.legend(loc='upper right')
# plt.tight_layout(pad=3)
# plt.savefig("sentiment_chart.png")
# plt.show()


# Replace these with your actual data



x = np.arange(len(creators))  # label locations
width = 0.25  # width of each bar

fig, ax = plt.subplots(figsize=(18, 6))

# Plot each sentiment category with offset
bars1 = ax.bar(x - width, positive_counts, width, label='Positive', color='green')
bars2 = ax.bar(x, neutral_counts, width, label='Neutral', color='gold')
bars3 = ax.bar(x + width, negative_counts, width, label='Negative', color='red')

# Add value labels
for bar_group in [bars1, bars2, bars3]:
    for bar in bar_group:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.3, str(height),
                    ha='center', va='bottom', fontsize=8)

# Customize axes
ax.set_xticks(x)
ax.set_xticklabels(creators, rotation=45, ha='right')
ax.set_ylabel('Comment Counts (out of 20)')
ax.set_title('Sentiment Analysis of YouTube Comments by Creator')
ax.set_ylim(0, max(max(positive_counts), max(neutral_counts), max(negative_counts)) + 5)
ax.legend(loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout(pad=3)
plt.savefig("sentiment_chart.png")
plt.show()
print("positive_counts:", positive_counts)
print("neutral_counts:", neutral_counts)
print("negative_counts:", negative_counts)  