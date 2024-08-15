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
        text = data['text'].strip()

        tweets = scrape(username) if username != "" else ""

        if tweets == "":
            if text == "":
                flash("No tweets found for this account. Please enter another username or enter text below.")
            else:
                personality_type = model.predict(text)
                flash(personality_type)
        else:
            personality_type = model.predict(tweets)
            flash(personality_type)

        return redirect("/")
    else:
        return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)