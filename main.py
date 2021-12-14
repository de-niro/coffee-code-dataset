from flask import Flask, render_template
from flask import jsonify, request
from analyzer.evaluate import Evaluator
from config import basedir

app = Flask(__name__)
app.config.from_object('config')
ev = Evaluator(basedir)

@app.route('/')
def index():
    content = {"graphs": ev.graph_ext, "stats": ev.stats, "extra_stats": ev.extra_stats}
    return render_template('index.html', content=content)

@app.route('/stats')
def stats():
    content = {"graphs": ev.graph_ext, "stats": ev.stats}
    return render_template('stats.html', content=content)

@app.route('/nn')
def nn():
    content = {"nn_stats": ev.nn_stats}
    return render_template('nn.html', content=content)

@app.route('/predict')
def predict():
    coding_hours = request.args.get('hours', type=int)
    coffee_cups = request.args.get('cups', type=int)

    return ev.Predict(coding_hours, coffee_cups)

if __name__ == "__main__":
    app.run(debug=False)
