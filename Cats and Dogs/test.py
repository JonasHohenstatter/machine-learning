import keras

image_path = "path_to_cat_or_dog_image"
model = "path_to_model"

model = keras.saving.load_model("save_at_25_v2.keras")
img = keras.utils.load_img(image_path, target_size=(180, 180))
img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)

prediction = model.predict(img_array)[0][0]
print(prediction)
if prediction < 0.5:
    print("Katze")
else:
    print("Hund")