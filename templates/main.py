import pickle
from flask import Flask, render_template


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/home2')
def home2():
    result = "This is a prediction or second page content"
    return render_template('index2.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/home2')
def home2():
    return render_template('index2.html')

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/')
def home():
    with open("animations.pkl", "rb") as f:  # Load saved CSS code
        css_code = pickle.load(f)
    return render_template("index.html", css=css_code)

    
    