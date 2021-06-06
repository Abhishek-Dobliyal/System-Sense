# Required imports
from flask import Flask, render_template, request, url_for, jsonify

# Initialize the app
app = Flask(__name__)

selected_option = 0
# Defining Routes
@app.route('/', methods=['GET', 'POST'])
def user_choice():
    global selected_option
    if request.method == 'POST':
        selected_option = int(request.form.get('select_action'))
        return render_template('index.html', selected_option=selected_option)
    
    return render_template('index.html', selected_option=selected_option)

@app.route('/response')
def response():
    global selected_option
    json_res = jsonify({'option': selected_option})
    selected_option = 0
    return json_res

if __name__ == '__main__':
    app.run(debug=True,  host='0.0.0.0')