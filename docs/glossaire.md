
**checkpoint**:
une sauvegarde de l'état d'un modèle entraîné à un instant donné — typiquement ses poids (paramètres appris), parfois accompagnés d'autres informations (état de l'optimizer, epoch atteint, historique de loss). Sert a réutiliser sans réentraîner 

**criterion**:
Autre nom de Loss fonction (historique Torch 7)

**Hyperparamètres**
Ce sont tous les réglages fixés à l'avance, avant l'entraînement, que le modèle n'apprend pas via la descente de gradient — contrairement aux paramètres (les poids, model.state_dict()) -> lr=1e-4 (learning rate), epochs=5, BATCH_SIZE = 64, etc

**logits**: 
Un logit, c'est le nombre brut que sort la dernière couche du réseau (model.fc), avant toute transformation en probabilité. Dans ton cas, model(images) renvoie un vecteur de 10 logits par image (un par classe EuroSAT).