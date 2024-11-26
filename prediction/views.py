from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
import os


def download_model():
    import os
    from google.cloud import storage

    # モデル保存用ディレクトリの作成
    os.makedirs('prediction/models', exist_ok=True)

    storage_client = storage.Client()
    bucket = storage_client.bucket('photosorting-app-dev-models')
    blob = bucket.blob('vgg16.h5')
    blob.download_to_filename('prediction/models/vgg16.h5')

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # モデルのダウンロード
            download_model()

            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 224, 224, 3))
            img_array = preprocess_input(img_array)

            # モデルパスの指定
            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
            model = load_model(model_path)
            result = model.predict(img_array)
            prediction = decode_predictions(result)
            img_data = request.POST.get('img_data')
            predictions = [{'name': label[1], 'probability': label[2] * 100} for label in prediction[0]]
            return render(request, 'home.html', {'form': form, 'predictions': predictions, 'img_data': img_data})
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})