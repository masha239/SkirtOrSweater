import pickle
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.model_selection import GridSearchCV
from utils import *


dirname = 'data'

categories = [folder for folder in os.listdir(dirname)]

# читаем и обрабатываем фотографии
pictures = []
target = []

for cat in categories:
    path = os.path.join(dirname, cat)
    for img in os.listdir(path):
        pictures.append(prepare_picture1(os.path.join(path, img), n=400))
        target.append(cat)

pictures = np.array(pictures)
target = np.array(target)


df = make_features(pictures, hist_step=20, angle_step=100)

# обучаемся
rf = RandomForestClassifier(random_state=0, max_depth=6)
parameters_rf = {'max_depth': range(2, 10)}

clf_rf = GridSearchCV(rf, parameters_rf, scoring='accuracy', cv=5)
clf_rf.fit(df, target)
print(f'RF: accuracy {clf_rf.best_score_:.3} {clf_rf.best_params_}')

# сохраняем модель
with open('model.pkl', 'wb') as file:
    pickle.dump(clf_rf.best_estimator_, file)
