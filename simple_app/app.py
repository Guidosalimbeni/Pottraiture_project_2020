from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os 

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'D:\Pottraiture_project_2020\simple_app\images'

@app.route('/')
def upload_f():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
      return 'file uploaded successfully'

# if __name__ == '__main__':
app.run(debug = True)