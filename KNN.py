import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing

from utils.utils import encode_label

data = pd.read_csv("car.data")
LABELS = ["unacc", "acc", "good", "vgood"]
TARGET = "class"

le = preprocessing.LabelEncoder()

buying = encode_label(le, data, "buying")
maint = encode_label(le, data, "maint")
door = encode_label(le, data, "door")
persons = encode_label(le, data, "persons")
lug_boot = encode_label(le, data, "lug_boot")
safety = encode_label(le, data, "safety")
cls = encode_label(le, data, "class")

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.1
)

model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train, y_train)
acc = model.score(x_train, y_train)

predicted = model.predict(x_test)
