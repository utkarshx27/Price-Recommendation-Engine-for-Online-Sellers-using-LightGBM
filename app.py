from flask import Flask, render_template, request
import pickle

model = pickle.load(open('model.plk', 'rb'))
app = Flask(__name__)

# define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# define the route for the recommendation page
@app.route('/recommend', methods=['POST'])
def recommend():
    # get the input values from the form
    item_condition_id = request.form['item_condition_id']
    shipping = request.form['shipping']
    name = request.form['name']
    category_name = request.form['category_name']
    brand_name = request.form['brand_name']
    item_description = request.form['item_description']
    # preprocess the input values
    input_data = [[item_condition_id, shipping, name, category_name, brand_name, item_description]]

    # make the prediction
    prediction = model.predict(input_data)

    # round the prediction to 2 decimal places
    predicted_price = round(prediction[0], 2)

    # return the predicted price to the recommendation page
    return render_template('recommendation.html', predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
