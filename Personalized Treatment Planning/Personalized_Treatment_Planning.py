# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# %% data create

"""
**Patient Dataset Information (1000 Patients)**

- **Age:** 20-80  
- **Gender:** Female (0), Male (1)  
- **Disease:** 0 (None), 1 (Hypertension), 2 (Diabetes), 3 (Both)  

**Symptoms:** Fever + Cough + Headache  
- **Fever:** 0 (None), 1 (Mild), 2 (Severe)  
- **Cough:** 0 (None), 1 (Mild), 2 (Severe)  
- **Headache:** 0 (None), 1 (Mild), 2 (Severe)  

**Blood Pressure:** 90 - 180  
**Blood Sugar Level:** 70 - 200  
**Previous Treatment Response:** 0 (Poor), 1 (Moderate), 2 (Good)  

---

**Treatment Rules:**  

- **0 (Treatment - 1)** (Basic Treatment): Blood pressure < 120, Blood sugar < 100, Symptoms ≤ 1  
- **1 (Treatment - 2)** (Intermediate Treatment): Blood pressure > 120, Blood sugar < 140, Symptoms == 1  
- **2 (Treatment - 3)** (Advanced Treatment): Blood pressure > 140, Blood sugar ≥ 150, Symptoms == 2  
"""

def assign_treatment_1(blood_pressure, blood_sugar, symptom):
    return (blood_pressure < 120) & (blood_sugar < 100) & (symptom <= 1)

def assign_treatment_2(blood_pressure, blood_sugar, symptom):
    return (blood_pressure > 120) & (blood_sugar < 140) & (symptom == 1)

def assign_treatment_3(blood_pressure, blood_sugar, symptom):
    return (blood_pressure > 140) & (blood_sugar >= 150) & (symptom == 2)

num_samples = 1000

age = np.random.randint(20, 80, size = num_samples)
gender = np.random.randint(0, 2, size = num_samples)
disease = np.random.randint(0, 4, size = num_samples)
symptom_fever = np.random.randint(0, 3, size = num_samples)
symptom_cough = np.random.randint(0, 3, size = num_samples)
symptom_headache = np.random.randint(0, 3, size = num_samples)
blood_pressure = np.random.randint(90, 180, size = num_samples)
blood_sugar = np.random.randint(70, 200, size = num_samples)
previous_treatment_responce = np.random.randint(0, 3, size = num_samples)

symptom = symptom_fever + symptom_cough + symptom_headache

treatment_plan = np.zeros(num_samples)

for i in range(num_samples):
    if assign_treatment_1(blood_pressure[i], blood_sugar[i], symptom[i]):
        treatment_plan[i] = 0 # Treatment 1
    elif assign_treatment_2(blood_pressure[i], blood_sugar[i], symptom[i]):
        treatment_plan[i] = 1 # Treatment 2
    else:
        treatment_plan[i] = 2 # Treatment 3

data = pd.DataFrame({
    "age":age,
    "gender":gender,
    "disease":disease,
    "symptom_fever":symptom_fever,
    "symptom_cough":symptom_cough,
    "symptom_headache":  symptom_headache,
    "blood_pressure":blood_pressure,
    "blood_sugar":blood_sugar,
    "previous_treatment_responce":previous_treatment_responce,
    "symptom":symptom,
    "treatment_plan":treatment_plan})

# %% training: DL -> Artificial Neural Network
X = data.drop(["treatment_plan"], axis = 1).values
y = to_categorical(data["treatment_plan"], num_classes = 3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = Sequential([
    Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(3, activation="softmax")])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test), batch_size = 32)

# %% evaluation
val_loss, val_accuracy = model.evaluate(X_test, y_test)
print(f"Validation accuracy: {val_accuracy}, validation loss: {val_loss}")

plt.figure()

plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], "bo-", label = "Training Accuracy")
plt.plot(history.history["val_accuracy"], "r^-", label = "Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history["loss"], "bo-", label = "Training Loss")
plt.plot(history.history["val_loss"], "r^-", label = "Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)





































