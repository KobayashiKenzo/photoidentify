# グローバル変数としてモデルを保持
model = None


def get_model():
    global model
    if model is None:
        # モデル保存用ディレクトリの作成
        os.makedirs('prediction/models', exist_ok=True)

        model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
        if not os.path.exists(model_path):
            from google.cloud import storage
            storage_client = storage.Client()
            bucket = storage_client.bucket('photosorting-app-dev-models')
            blob = bucket.blob('vgg16.h5')
            blob.download_to_filename(model_path)

        # モデルのロード（初回のみ）
        model = load_model(model_path)
    return model


def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 224, 224, 3))
            img_array = preprocess_input(img_array)

            # キャッシュされたモデルを使用
            model = get_model()
            result = model.predict(img_array)
            prediction = decode_predictions(result)
            img_data = request.POST.get('img_data')
            predictions = [{'name': label[1], 'probability': label[2] * 100} for label in prediction[0]]
            return render(request, 'home.html', {'form': form, 'predictions': predictions, 'img_data': img_data})

        return render(request, 'home.html', {'form': form})