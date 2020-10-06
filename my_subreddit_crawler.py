import praw
from reddit_crawler_account import *
from pprint import pprint as pp
import json
from datetime import datetime

reddit = praw.Reddit(username=USERNAME,
                     password=PASSWORD,
                     client_id=CLIENT_ID,
                     client_secret=CLIENT_SECRET,
                     user_agent=USER_AGENT)

epoch = datetime.utcfromtimestamp(0)


# Convert datetime to unix time (a.k.a. number of seconds since 1970-01-01
def utc_from_datetime(dt):
    return int((dt - epoch).total_seconds())


def datetime_string_from_utc(ut):
    return '{:%B %d, %Y}'.format(datetime.fromtimestamp(ut))


# We're gonna fetch all news from 2006-12-31 to 2016-12-31, 1000 titles each 30 days
start_timestamp = utc_from_datetime(datetime(2006, 1, 1))
end_timestamp = utc_from_datetime(datetime(2017, 1, 1))
interval = 31536000  # 365 days in seconds

# We check if login information is correct
pp(reddit.user.me())
reddit.read_only = True

subreddit_names = ['qualitynews', 'neutralnews', 'uncensorednews', 'usanews', 'businessnews',
                   'StockNews', 'UpliftingNews', 'news', 'worldnews']

# subreddit_names = ['science']

time_range = ['hour', 'day', 'week', 'month', 'year', 'all']
news_titles = {}

# We're gonna fetch more than just a titles. Let's get the timestamp and #upvotes

try:
    for subreddit_name in subreddit_names:
        subreddit = reddit.subreddit(subreddit_name)

        for start_time in range(start_timestamp, end_timestamp, interval):
            end_time = min((start_time + interval, end_timestamp))

            titles = {post.id: (post.title, datetime_string_from_utc(post.created_utc), post.score, post.num_comments)
                      for post in subreddit.submissions(start=start_time, end=end_time)
                      if post.id not in news_titles}
            pp('Fetched ' + str(len(titles)) + ' news in /r/' + subreddit_name +
               ' from ' + datetime_string_from_utc(start_time) +
               ' to ' + datetime_string_from_utc(end_time))
            news_titles = {**news_titles, **titles}

finally:
    with open(subreddit_names[0]+'_titles.json', mode='w', encoding='utf-8') as f:
        json.dump(news_titles, f)


# for t in time_range:
#     top_titles = {post.id: post.title for post in subreddit.top(t, limit=1000)
#                   if post.id not in news_titles}
#     controversial_titles = {post.id: post.title for post in subreddit.controversial(t, limit=1000)
#                             if post.id not in news_titles}
#     pp(len(top_titles)+len(controversial_titles))
#     news_titles = {**news_titles, **top_titles, **controversial_titles}
#
# hot_titles = {post.id: post.title for post in subreddit.hot(limit=1000)
#               if post.id not in news_titles}
# new_titles = {post.id: post.title for post in subreddit.new(limit=1000)
#               if post.id not in news_titles}
# rising_titles = {post.id: post.title for post in subreddit.rising(limit=1000)
#                  if post.id not in news_titles}
# news_titles = {**news_titles, **hot_titles, **rising_titles, **news_titles}