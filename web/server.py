from flask import Flask, render_template, request, redirect, flash
from webscraper import scrape
import model.model as model

app = Flask(__name__)
app.config['SECRET_KEY'] = "asbdiuabdiuasbduib"

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.form.to_dict()
        username = data['username']

        tweets = scrape(username)

        if tweets == "":
            flash("No tweets")
        else:
            personality_type = model.predict(tweets)

            flash(personality_type)

        return redirect("/")
    else:
        return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)