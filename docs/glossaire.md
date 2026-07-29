**Logits**: 
Un logit, c'est le nombre brut que sort la dernière couche du réseau (model.fc), avant toute transformation en probabilité. Dans ton cas, model(images) renvoie un vecteur de 10 logits par image (un par classe EuroSAT).

**criterion**:
Autre nom de Loss fonction (historique Torch 7)
