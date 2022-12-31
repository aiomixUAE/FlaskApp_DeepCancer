from flask import Flask, send_file, render_template
from distutils.log import debug
from fileinput import filename
from flask import *  
import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

fig,ax=plt.subplots(figsize= (6,6))
ax=sns.set_style(style="darkgrid")

x=[i for i in range (100)]
y=[i for i in range (100)]
  
app = Flask(__name__)

@app.route('/')  
def main():  
    return render_template("index.html")  
  
@app.route('/success', methods = ['POST'])
def success():  
    if request.method == 'POST':  
        f = request.files['file']
        f.save(f.filename)
        print()
        filename = 'azurepikel.pkl'
        loaded_model = pickle.load(open(filename, 'rb'))

        df = pd.read_csv(f.filename)

        # print(df.head)

        y_pred = loaded_model.predict(df)
        # actual_mean = pd.DataFrame(y_test.mean(axis=0))
        pred_mean = pd.DataFrame(y_pred.mean(axis=0))

        # act=actual_mean.values.flatten()
        pred=pred_mean.values.flatten()

        s1 = pd.Series(df)
        s2 = pd.Series(pred)

        plt.figure(figsize=(20,10))
        ax = plt.subplot(111)
        plt.title('Average of actual and predicted CNVs across all samples')
        plt.xlabel('No. of features (genes)')
        plt.ylabel('Average of CNVs across samples')
        ax.plot(s1, 'b--', label='Actual')
        ax.plot(s2, 'r--', label='Predicted')
        ax.legend()
        plt.grid(True)
        plt.show()

        return render_template("Acknowledgement.html", name = f.filename)  
  

# @app.route('/visualize')
# def visualize ():
#     ypoints = np.array([3, 8, 1, 10])
#     plt.plot(ypoints, linestyle = 'dotted')
#     # sns.lineplot(x,y)
#     canvas=FigureCanvas(fig)
#     img = io.BytesIO()
#     fig.savefig(img)
#     img.seek(0)
#     return send_file(img, mimetype ='img/png')

if __name__ == '__main__':  
    app.run(debug=True)