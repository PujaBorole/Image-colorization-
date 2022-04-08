from flask import *  
from tensorflow import keras
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array


app = Flask(__name__) #creating the Flask class object   
app.secret_key = b'PujaSamiksha@ImgColor/'
model = keras.models.load_model('model.h5')

@app.route('/') #decorator drfines the   
def index():  
    return  render_template('index.html'); 

@app.route('/upload',methods=['POST'])  
def upload():  
    fileobj = request.files['file']
    file_extensions =  ["JPG","JPEG","PNG"]
    uploaded_file_extension = fileobj.filename.rsplit(".",1)[1]
        #validating file extension
    if(uploaded_file_extension.upper() in file_extensions):
        destination_path= f"static/gray/{fileobj.filename}"
        fileobj.save(destination_path)
        flash('Image saved successfully ')
        output= predictClass(destination_path)
        return redirect(url_for('index'))
        
    else:
        flash("Only images are accepted (png, jpg, jpeg)")
        return redirect(url_for('index')) 
    
def predictClass(destination_path):
    SIZE = 160
    gray_img = []
    img = cv2.imread(destination_path,1)

    #resizing image
    img = cv2.resize(img, (SIZE, SIZE))
    img = img.astype('float32') / 255.0
    gray_img.append(img_to_array(img))
    test_gray_image = gray_img[0:]
    test_gray_image = np.reshape(test_gray_image,(len(test_gray_image),SIZE,SIZE,3))
    predicted = np.clip(model.predict(test_gray_image[0].reshape(1,SIZE, SIZE,3)),0.0,1.0).reshape(SIZE, SIZE,3)

    cv2.imwrite("static/color/image.jpg", 255*predicted)

if __name__ =='__main__':  
    app.run(debug = True)  