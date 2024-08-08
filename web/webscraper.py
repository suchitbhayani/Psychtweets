import requests
import json
import model.model as model

username = "CashNasty"
num_tweets = 75

url = f"https://syndication.twitter.com/srv/timeline-profile/screen-name/{username}"

request = requests.get(url)
html = request.text

start_str = '<script id="__NEXT_DATA__" type="application/json">'
end_str = '</script></body></html>'

start_index = html.index(start_str) + len(start_str)
end_index = html.index(end_str, start_index)

json_str = html[start_index: end_index]
data = json.loads(json_str)

tweets = ""
for i in range(num_tweets):
    try:
        #print(f"{i}:", data["props"]["pageProps"]["timeline"]["entries"][i]["content"]["tweet"]["full_text"])
        tweets += data["props"]["pageProps"]["timeline"]["entries"][i]["content"]["tweet"]["full_text"]
    except IndexError:
        break

#print(tweets)
print(model.predict(tweets))