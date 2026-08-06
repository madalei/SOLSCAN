**AOI (Area Of Interest)**:
Zone d'intérêt géographique — l'emprise (bbox) sur laquelle on récupère/traite les données. Ex: `AOI_BBOX = (2.25, 51.00, 2.40, 51.07)` (lon_min, lat_min, lon_max, lat_max en WGS84) délimite le rectangle dans lequel on va chercher la scène Sentinel-2 et le tuile WorldCover correspondante.

**Accuracy**:
Measures overall proportion of correct predictions - Best when classes are balanced

**checkpoint**:
une sauvegarde de l'état d'un modèle entraîné à un instant donné — typiquement ses poids (paramètres appris), parfois accompagnés d'autres informations (état de l'optimizer, epoch atteint, historique de loss). Sert a réutiliser sans réentraîner 

**criterion**:
Autre nom de Loss fonction (historique Torch 7)

**ESA WorldCover**:
fournit une carte d'occupation des sols pixel par pixel (résolution 10m), avec plusieurs classes de couverture (forêt, eau, bâti, etc.). 

**F1-score**:
Measure balance between precision and recall - Best when Both false positives and false negatives matter

**Hyperparamètres**
Ce sont tous les réglages fixés à l'avance, avant l'entraînement, que le modèle n'apprend pas via la descente de gradient — contrairement aux paramètres (les poids, model.state_dict()) -> lr=1e-4 (learning rate), epochs=5, BATCH_SIZE = 64, etc

**Inference**:
means running a trained model on new data to get predictions — as opposed to training, where the model's weights are being learned/updated from labeled data

**logits**: 
Un logit, c'est le nombre brut que sort la dernière couche du réseau (model.fc), avant toute transformation en probabilité. Dans ton cas, model(images) renvoie un vecteur de 10 logits par image (un par classe EuroSAT).

**mask**
c'est l'image de vérité-terrain (ground truth) associée à chaque tuile satellite : une image en niveaux de gris où chaque pixel dit "artificialisé" ou "pas artificialisé".

**Precision**:
Measure How many predicted positives are correct - Best when false positives are costly

**Recall**:
Measure How many actual positives are found - Best when False negatives are costly

**Softmax**:
Utilisée principalement dans les réseaux de neurones de classification multi-classes. Son rôle est de transformer un vecteur de scores bruts (appelés logits) en probabilités qui somment à 1.

**Train loss**:
Valeur de la fonction de perte (loss function) calculée sur les données d'entraînement. C'est l'indicateur que le modèle essaie de minimiser pendant l'apprentissage.