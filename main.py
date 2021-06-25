from flask import Flask, render_template, request
app = Flask(__name__)
import pickle


# open a file, where you stored the pickled data
file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        gender=int(myDict['gender'])
        age_year= int(myDict['age_year'])
        fever = int(myDict['fever'])
        cough = int(myDict['cough'])
        runny_nose = int(myDict['runny_nose'])
        muscle_soreness = int(myDict['muscle_soreness'])
        pneumonia = int(myDict['pneumonia'])
        diarrhea = int(myDict['diarrhea'])
        lung_infection = int(myDict['lung_infection'])
        travel_history = int(myDict['travel_history'])
        isolation_treatment = int(myDict['isolation_treatment'])
        # Code for inference
        inputFeatures = [gender, age_year, fever, cough, runny_nose, muscle_soreness, pneumonia, diarrhea, lung_infection, travel_history, isolation_treatment]
        infProb =clf.predict_proba([inputFeatures])[0][1]
        print(infProb)
        return render_template('show.html', inf=round(infProb*100))
    return render_template('index.html')
    # return 'Hello, World!' + str(infProb)


if __name__ == "__main__":
    app.run(debug=True)