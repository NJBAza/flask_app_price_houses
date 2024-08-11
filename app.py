from flask import (
    Flask,
    redirect,
    render_template,
    request,
    url_for,
)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def submit():
    # Process form data here
    data = request.form
    print(data)  # Example: print data to console
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
