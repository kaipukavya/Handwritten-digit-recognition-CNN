import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

print(f'✅ TensorFlow : {tf.__version__}')
print(f'✅ GPU Available: {len(tf.config.list_physical_devices("GPU")) > 0}')

# 1. Load & Preprocess Data
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()

# Normalize and reshape for CNN (28x28x1) and MLP (784)
X_train_cnn = X_train_raw.astype('float32')[..., np.newaxis] / 255.0
X_test_cnn  = X_test_raw.astype('float32')[...,  np.newaxis] / 255.0
X_train_mlp = X_train_raw.astype('float32').reshape(-1, 784) / 255.0
X_test_mlp  = X_test_raw.astype('float32').reshape(-1,  784) / 255.0

y_train = to_categorical(y_train_raw, 10)
y_test  = to_categorical(y_test_raw, 10)
y_true  = y_test_raw

# 2. Build Models
def build_cnn():
    # Architecture: Conv -> BN -> Conv -> BN -> Pool -> Dropout -> Conv -> BN -> Pool -> Dense
    m = models.Sequential([
        layers.Input(shape=(28,28,1)),
        layers.Conv2D(32,(3,3),padding='same',activation='relu',name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.Conv2D(64,(3,3),padding='same',activation='relu',name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling2D((2,2),name='pool1'),
        layers.Dropout(0.25,name='drop1'),
        layers.Conv2D(128,(3,3),padding='same',activation='relu',name='conv3'),
        layers.BatchNormalization(name='bn3'),
        layers.MaxPooling2D((2,2),name='pool2'),
        layers.Dropout(0.25,name='drop2'),
        layers.Flatten(name='flatten'),
        layers.Dense(256,activation='relu',name='dense1'),
        layers.BatchNormalization(name='bn4'),
        layers.Dropout(0.5,name='drop3'),
        layers.Dense(10,activation='softmax',name='output')
    ], name='CNN_MNIST')
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

def build_mlp():
    m = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(512,activation='relu'),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(256,activation='relu'),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(10,activation='softmax')
    ], name='MLP_MNIST')
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return m

# 3. Training
cnn_model = build_cnn()
mlp_model = build_mlp()
cb = [EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)]

print('\n🚀 Training CNN (5 Epochs)...'); cnn_model.fit(X_train_cnn, y_train, epochs=5, batch_size=128, validation_split=0.1, callbacks=cb, verbose=1)
print('\n🚀 Training MLP (5 Epochs)...'); mlp_model.fit(X_train_mlp, y_train, epochs=5, batch_size=128, validation_split=0.1, callbacks=cb, verbose=1)

# 4. Feature Map Setup
inp = keras.Input(shape=(28,28,1))
x_layer = inp
feature_extractors = {}
for layer in cnn_model.layers:
    x_layer = layer(x_layer)
    if layer.name in ('conv1','conv2','conv3'):
        feature_extractors[layer.name] = keras.Model(inputs=inp, outputs=x_layer)

print('\n✅ Training Complete. Model Comparison:')
cnn_acc = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)[1]
mlp_acc = mlp_model.evaluate(X_test_mlp, y_test, verbose=0)[1]
print(f"CNN Accuracy: {cnn_acc*100:.2f}% | Parameters: {cnn_model.count_params():,}")
print(f"MLP Accuracy: {mlp_acc*100:.2f}% | Parameters: {mlp_model.count_params():,}")
from IPython.display import display, HTML
import base64, io
from PIL import Image
import google.colab.output

canvas_html = """
<div style="background:#f8f9fa; padding:25px; border-radius:20px; width:340px; text-align:center; font-family: sans-serif; border: 2px solid #3498db; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
    <h3 style="color:#2c3e50; margin-bottom:15px;">🏦 Bank Check Digitizer</h3>
    <p style="font-size:12px; color:#7f8c8d; margin-bottom:10px;">Draw a digit (0-9) clearly in the center</p>
    <canvas id="canvas" width="280" height="280" style="border:5px solid #2c3e50; background:black; border-radius:12px; cursor:crosshair;"></canvas>
    <div style="margin-top:20px; display: flex; justify-content: center; gap: 12px;">
        <button id="clear" style="padding:10px 25px; background:#e74c3c; color:white; border:none; border-radius:8px; cursor:pointer; font-weight:bold;">Clear</button>
        <button id="predict" style="padding:10px 25px; background:#2ecc71; color:white; border:none; border-radius:8px; cursor:pointer; font-weight:bold;">Predict</button>
    </div>
    <div id="out" style="margin-top:25px; font-size:24px; font-weight:bold; color:#2c3e50; min-height:30px;"></div>
</div>

<script>
    var canvas = document.getElementById('canvas');
    var ctx = canvas.getContext('2d');
    var isDrawing = false;
    ctx.strokeStyle = "white";
    ctx.lineWidth = 18;
    ctx.lineCap = "round";

    canvas.onmousedown = (e) => { isDrawing = true; ctx.beginPath(); ctx.moveTo(e.offsetX, e.offsetY); };
    canvas.onmousemove = (e) => { if(isDrawing) { ctx.lineTo(e.offsetX, e.offsetY); ctx.stroke(); } };
    canvas.onmouseup = () => { isDrawing = false; };

    document.getElementById('clear').onclick = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        document.getElementById('out').innerHTML = "";
    };

    document.getElementById('predict').onclick = () => {
        var dataURL = canvas.toDataURL('image/png');
        google.colab.kernel.invokeFunction('notebook.PredictDigit', [dataURL], {});
    };
</script>
"""

def process_and_predict(data_url):
    # 1. Decode Image from Canvas
    img_str = data_url.split(',')[1]
    img_data = base64.b64decode(img_str)
    img = Image.open(io.BytesIO(img_data)).convert('L')

    # 2. Preprocess (Resize to 28x28 and Normalize)
    img = img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # 3. CNN Prediction
    preds = cnn_model.predict(img_array, verbose=0)
    digit = np.argmax(preds)
    conf = np.max(preds) * 100

    # 4. UI Update
    display(HTML(f"<script>document.getElementById('out').innerHTML = 'Digit: {digit} <small>({conf:.1f}%)</small>'</script>"))

    # 5. Visualize Feature Maps for the drawing
    fig, axes = plt.subplots(1, 3, figsize=(15, 3))
    for i, name in enumerate(['conv1','conv2','conv3']):
        fmaps = feature_extractors[name].predict(img_array, verbose=0)
        axes[i].imshow(fmaps[0,:,:,0], cmap='magma')
        axes[i].set_title(f'Live {name} Features')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

# Connect JS to Python
google.colab.output.register_callback('notebook.PredictDigit', process_and_predict)
display(HTML(canvas_html))
