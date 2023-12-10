from flask import Flask, request, render_template, jsonify
from dl import predict as prd
from nlp import predict
# import your model here

app = Flask(__name__)

@app.route("/")
def home():
  return render_template('index.html')

@app.route("/dl")
def dl():
  return render_template('dl.html')

@app.route('/dl_predict', methods=['POST'])
def dl_predict():
    Pclass = float(request.form.get('pclass'))
    Age = float(request.form.get('age'))
    Sex = float(request.form.get('sex'))
    sibsp = float(request.form.get('sibsp'))
    parch = float(request.form.get('parch'))
    Fare = float(request.form.get('fare'))
    embarked = request.form.get('embarked')
    title = request.form.get('title')
    Cabin = request.form.get('cabin')
    FamilySize = sibsp + parch
    Embarked_Q, Embarked_S = 0, 0

    if embarked == 'Q':
        Embarked_Q = 1.0
        Embarked_S = 0.0
    elif embarked == 'S':
        Embarked_Q = 0.0
        Embarked_S = 1.0
    else:
        Embarked_Q = 0.0
        Embarked_S = 0.0
    
    Title_Miss, Title_Mr, Title_Mrs, Title_Other = None, None, None, None
    if title == 'miss':
        Title_Miss = 1.0
        Title_Mr = 0.0
        Title_Mrs = 0.0
        Title_Other = 0.0
    elif title == 'mr':
        Title_Miss = 0.0
        Title_Mr = 1.0
        Title_Mrs = 0.0
        Title_Other = 0.0
    elif title == 'mrs':
        Title_Miss = 0.0
        Title_Mr = 0.0
        Title_Mrs = 1.0
        Title_Other = 0.0
    elif title == 'other':
        Title_Miss = 0.0
        Title_Mr = 0.0
        Title_Mrs = 0.0
        Title_Other = 1.0
    else:
        Title_Miss = 0.0
        Title_Mr = 0.0
        Title_Mrs = 0.0
        Title_Other = 0.0
    
    # lst = [3.0,1.0,34.5,7.8292,0.0,1.0,0.0,1.0,0.0,0.0,1.0,0.0]
    c = {'A': 1, 'B':2, 'C': 3, 'D': 4, 'E':5, 'F':6, 'G':7, 'T':8}
    if Cabin in (''):
        Cabin = 0.0
    else:
        Cabin = c[Cabin[0]]
    
    lst = [Pclass,Sex,Age,Fare,Cabin,FamilySize,Title_Miss,Title_Mr,Title_Mrs,Title_Other,Embarked_Q,Embarked_S]
    text = prd(lst)
    return jsonify({'text': f"<p class='card-text' id='prediction'>{text}</p>"})

@app.route("/nlp")
def nlp():
    return render_template('nlp.html')

@app.route("/nlp_predict", methods=['POST'])
def nlp_predict():
    # test: Vào lúc 12 giờ ngày 12, ông Nguyễn Văn Công đã bắt đầu làm việc ở Việt Nam, sắp tới sẽ sang công tác ở Thái Lan.
    texts = request.form['input_paragraph']
    texts = [i for i in texts.split('\n') if i != '']
    res = ''
    for text in texts:
        list_text = predict(text)
        for tup in list_text:
            if tup[1]:
                res += f"<span class='highlight'>{tup[0]}</span> "
            else:
                res += f"{tup[0]} "
        res += '<br>'
    
    return jsonify({'text': f"<p class='card-text' id='prediction'>{res}</p>"})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port='8000')
