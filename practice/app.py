from flask import (
    Flask,
    render_template,
    request,
)

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Retrieve the text from the textarea
        text = request.form.get("textarea")

        # Print the text in terminal for verification
        print(text)

    return render_template("index.html")


if __name__ == "__main__":
    app.run()
