# Feuille de route : segmentation multi-classe des zones artificialisées (U-Net)

Statut : **repris et fonctionnel** (branche `unet`). Le premier essai (§5) avait été abandonné (step back) faute de résultats exploitables ; un second essai (§8) a corrigé les causes identifiées alors (pondération de classe, volume/diversité du dataset, epochs) et en a découvert de nouvelles, propres à la classe Parking et à la couverture géographique de Cartofriches -- voir §8 pour le diagnostic détaillé et la conclusion actuelle sur cette classe.

## 1. Objectif

Remplacer la segmentation binaire ("artificialisé" vs "non-artificialisé", basée sur ESA WorldCover) par une segmentation **multi-classe**, pour distinguer les types de zones réellement pertinents pour la prospection solaire :

| ID | Classe | Justification métier |
|---|---|---|
| 0 | Fond (non-artificialisé) | — |
| 1 | Parking (surface moyenne/grande) | Cible directe : loi APER (2023) impose des ombrières solaires sur les grands parkings |
| 2 | Industriel / commercial | Toitures/friches potentielles |
| 3 | Friche | Cible prioritaire : sites déjà délaissés, mobilisables sans destruction d'usage actif |
| 4 | Residentiel | Ajoutée au second essai (§8) -- initialement hors périmètre (ligne ci-dessous), son absence faisait retomber tout pixel résidentiel par défaut sur Fond, sans classe correcte où le router ; le modèle le confondait avec Parking/Friche (ambigu à 10m/pixel) |

En cas de chevauchement de polygones, **friche est prioritaire** (un site industriel devenu friche doit être étiqueté friche), puis parking/industriel, résidentiel en dernier (priorité la plus basse) -- voir `helpers/mask_rasterize.py`.

## 2. Sources de données validées

Aucune de ces sources ne nécessite de compte/clé API — accessibles en HTTP simple.

### OpenStreetMap (Overpass API)
- Endpoint : `https://overpass-api.de/api/interpreter`
- Tags utilisés : `amenity=parking` (classe 1) ; `building=industrial|warehouse`, `landuse=industrial|commercial` (classe 2)
- **Piège technique** : le serveur renvoie `HTTP 406` si le header `User-Agent` par défaut de la librairie `requests` est utilisé — il faut un `User-Agent` explicite (ex. `"MonProjet/1.0 (contact)"`).
- Les `relation` de type `multipolygon` **sont gérées depuis le second essai**, mais parking uniquement (`helpers/geo_fetch.fetch_osm_polygons`) : un premier essai les activait aussi pour industriel/résidentiel, ce qui a fait timeout tous les miroirs Overpass sur des AOI qui marchaient très bien en way-only -- une relation `landuse=residential` peut couvrir tout un quartier avec des centaines de ways membres, rendant `out geom;` beaucoup trop coûteux à résoudre côté serveur. Les anneaux intérieurs (`role=inner`, îlots de végétation/luminaires) sont repeints en Fond plutôt que laissés dans la classe extérieure. Simplification assumée (pas de shapely) : chaque way `outer` est traité comme un anneau fermé indépendant -- correct si le contour est un seul way, approximatif s'il est fragmenté en plusieurs.
- Filtre de surface minimale nécessaire côté parking (ex. ~1500 m², proxy du seuil loi APER) pour exclure les petites places résidentielles — se calcule facilement avec une formule de shoelace une fois les polygones reprojetés en CRS métrique, pas besoin de shapely.

### Cartofriches (Cerema, WFS)
- Endpoint : `https://www.geo2france.fr/geoserver/cerema/ows`, couche `cerema:cartofriche`
- Format : GeoJSON (`OUTPUTFORMAT=application/json`), champs utiles : `site_id` (préfixé par le code INSEE de la commune), `site_nom`, `site_type` (souvent `"inconnu"` — sans importance, seule la présence compte)
- **Piège technique** : le paramètre `BBOX` du WFS est peu fiable sur ce GeoServer (souci d'ordre d'axes lat/lon vs lon/lat). Solution robuste : filtrer par département via `CQL_FILTER=site_id LIKE '<préfixe_INSEE>%'`, récupérer tout le département (léger, ~1400 friches pour le Nord), puis filtrer côté client par intersection de bounding box avec l'AOI.
- Couverture confirmée : 5557 friches en Hauts-de-France, 1401 dans le Nord (59), 42 dans une AOI élargie autour de Dunkerque.
- **Couverture hors Hauts-de-France incertaine, probablement faible** : constaté au second essai (§8) -- les 4 AOI ajoutées hors 59/62 pour cibler Parking (Bouches-du-Rhône 13, Gironde 33, Haute-Garonne 31, Seine-et-Marne 77) remontent toutes `0 friche(s)`, alors que ce sont des zones urbaines/industrielles denses où il devrait statistiquement y en avoir. À vérifier avant d'ajouter d'autres AOI hors Hauts-de-France en comptant sur Cartofriches pour la classe Friche.

### Choix technique : pas de dépendance géo lourde
`rasterio.warp.transform_geom` (reprojection) + `rasterio.features.rasterize` (rasterisation multi-classe, avec priorité par ordre d'empilement des formes) suffisent. Pas besoin de `shapely`/`geopandas`/`pyproj`/`osmnx` — cohérent avec l'empreinte de dépendances existante du projet (déjà `rasterio`-only). Seul ajout nécessaire : `requests` (déjà présent en transitif via `pystac-client`, à passer en dépendance directe si repris).

## 3. Pipeline de préparation des données (structure validée)

1. Recherche + lecture d'une scène Sentinel-2 L2A sur l'AOI (Microsoft Planetary Computer STAC) — identique au pipeline binaire existant.
2. Fetch OSM (Overpass) + Cartofriches (WFS) sur la même AOI.
3. Reprojection de tous les polygones vers le CRS de la scène S2 (`rasterio.warp.transform_geom`).
4. Filtre de surface minimale sur les parkings.
5. Rasterisation en un seul mask multi-classe (`rasterio.features.rasterize`), formes empilées dans l'ordre de priorité (friche en dernier).
6. Découpage en tuiles fixes (ex. 256×256), sauvegarde images/masks en PNG. **Les masks stockent des indices de classe bruts (0..N-1), pas du 0-255** — pas visualisables tels quels dans un visualiseur standard, nécessite un remap couleur pour l'inspection visuelle.

## 4. Modèle et entraînement (structure validée)

- Modèle : `segmentation_models_pytorch.Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=N)` — déjà générique, aucun changement nécessaire pour passer de 1 à N classes.
- Loss : `DiceLoss(mode="multiclass")` + `CrossEntropyLoss()` (au lieu de `mode="binary"` + `BCEWithLogitsLoss`).
- Métrique : IoU moyenné sur les classes (mean IoU) via `argmax` des logits — **mais toujours reporter l'IoU par classe séparément**, pas seulement la moyenne (voir §5, c'est ce qui a révélé le vrai problème).
- Chargement des masks : `PILToTensor()` (pas `ToTensor()`, qui diviserait les indices de classe par 255) + `.long()`.

## 5. Résultat du premier essai — diagnostic détaillé

Essai réalisé sur une AOI de ~475 km² autour de Dunkerque (66 tuiles générées, 46/9/11 train/val/test), 15 epochs.

**IoU par classe obtenu :**

| Classe | IoU |
|---|---|
| Fond | 0.888 |
| Industriel/commercial | 0.119 |
| Parking | 0.001 |
| Friche | 0.0002 |

**Causes identifiées (vérifiées empiriquement, pas des hypothèses) :**

1. **Déséquilibre de classe extrême, même localement.** Parking et friche ne représentent respectivement que 0.28% et 0.25% des pixels sur l'ensemble de l'AOI. Même dans les tuiles qui *contiennent* ces classes, elles restent minoritaires (médiane 0.27% pour parking, 0.53% pour friche par tuile). La loss n'étant pas pondérée par classe, le réseau trouve un minimum trivial ("prédire fond/industriel partout") sans jamais apprendre à distinguer ces classes rares.
2. **Objets minuscules à la résolution Sentinel-2 (10m/pixel).** Le seuil de 1500 m² pour un parking ne représente que ~15 pixels ; une friche de 2 ha ne fait que ~200 pixels sur une tuile de 65 536 pixels.
3. **Dataset trop petit et non diversifié.** 46 tuiles d'entraînement, toutes issues d'une seule ville (Dunkerque) — pas assez de diversité visuelle pour généraliser une signature "parking"/"friche".
4. **Nombre d'epochs insuffisant** compte tenu du signal faible disponible pour les classes rares.

Point de vérification confirmé par script (pas juste supposé) : les classes rares sont bien présentes dans les 3 splits (train/val/test), donc ce n'est pas un problème de fuite de split — c'est un problème de signal/poids dans la loss et de volume de données.

## 6. Prochaines étapes (priorisées) -- état après le second essai (§8)

### Doit être fait avant toute reprise sérieuse
1. ~~**Pondérer la loss par classe**~~ -- fait (`helpers/segmentation_dataset.compute_class_pixel_weights` + `training/seg_engine.SegEngine`).
2. ~~**Étendre le dataset à plusieurs AOI distinctes**~~ -- fait, 17 AOI (`helpers/landuse_aois.py`), 1097+ tuiles. Voir §8 pour le détail et une limite découverte (Cartofriches hors Hauts-de-France).

### À évaluer ensuite
3. ~~Gérer les `relation` OSM (multipolygones)~~ -- fait pour Parking uniquement (voir §2, §8) ; industriel/résidentiel laissés en way-only (coût Overpass prohibitif constaté empiriquement).
4. Stratifier le split train/val/test par présence de classe rare plutôt qu'un `random_split` uniforme -- toujours pas fait ; `helpers/dataloaders.build_group_kfold_dataloaders` (groupé par AOI) existe comme alternative pour une estimation de généralisation plus honnête, mais ne résout pas spécifiquement la stratification par classe rare.
5. Envisager du sur-échantillonnage (oversampling) des tuiles contenant les classes rares -- **non fait**, seul levier de ce type encore non essayé sur Parking (voir §8, piste possible mais rendement incertain vu le plafond structurel constaté).
6. ~~Réévaluer le nombre d'epochs~~ -- fait, early stopping sur `val_mean_iou` (`training/seg_engine.SegEngine.train_model`, param `patience`) plutôt qu'un nombre fixe.
7. ~~Exporter une version colorée des masks~~ -- fait (`data/landuse/masks_preview/`, généré par `notebooks/fetch_landuse_dataset.ipynb`).

## 7. Fichiers de référence (état à l'abandon du premier essai, pour mémoire)

Pour mémoire, la structure qui avait été mise en place et validée fonctionnellement avant le retrait qui a suivi le premier essai (§5) -- **tous ces fichiers existent de nouveau et sont à jour dans le repo actuel**, voir §8 pour l'état courant :
- `notebooks/fetch_landuse_dataset.ipynb` — fetch + rasterisation + tuilage
- `notebooks/train_unet_landuse.ipynb` — entraînement + visualisation + IoU par classe
- `helpers/segmentation_dataset.py` — `SegmentationTileDataset`, transforms image/mask
- `models/unet_builder.py` — `build_unet(num_classes, device)` (générique, réutilisable tel quel)
- `training/seg_engine.py` — `SegEngine` (loss Dice+CE, IoU par classe, boucle train/eval générique)
- `training/pipeline_configs.py` — entrée `UNET_CONFIG` (`num_classes=5` désormais, residentiel ajoutée)
- Données générées : `data/landuse/{images,masks}`, checkpoint `checkpoints/unet_landuse.pth`

## 8. Second essai (repris) -- ce qui a été ajouté, et le plafond constaté sur Parking

Le second essai a corrigé, dans l'ordre, les causes identifiées au §5 (pondération de classe, volume/diversité du dataset, epochs), puis exploré des leviers supplémentaires spécifiquement pour la classe Parking (restée quasi à 0 malgré tout le reste). Chronologie des ajouts, chacun mesuré séparément :

1. **Classe Residentiel** (5e classe, §1) -- absente du scope initial, son absence faisait retomber le résidentiel sur Fond par défaut ; le modèle le confondait avec Parking/Friche. A amélioré Friche et Industriel de façon notable au premier run où elle a été ajoutée.
2. **Pondération de classe** (`compute_class_pixel_weights`, inverse-fréquence) -- fix #1 du §6 original.
3. **Augmentation de données** (flip horizontal/vertical + rotation 90/180/270°, `helpers/segmentation_dataset.py`) -- appliquée au train uniquement (deux instances du dataset, une `augment=True` une `augment=False`, mêmes indices), val/test restent stables pour des métriques comparables.
4. **Sélection du meilleur checkpoint par `val_mean_iou`**, pas par `val_loss` minimal (`SegEngine.best_state_dict`) -- constat empirique que les deux ne bougent pas ensemble ici : le val loss est dominé par le volume de pixels de Fond même pondéré, alors qu'une epoch peut avoir un val loss moins bon mais mieux détecter les classes rares.
5. **`GroupKFold` par AOI** (`helpers/dataloaders.build_group_kfold_dataloaders`) -- alternative au split aléatoire pour une estimation de généralisation plus honnête (les tuiles d'une même ville se ressemblent visuellement), disponible en section 10 du notebook d'entraînement, coûteuse donc utilisée en vérification ponctuelle plutôt qu'en boucle de dev.
6. **Extension du dataset à 17 AOI, dont 4 choisies spécifiquement pour leur densité de grands parkings** (Vitrolles, Val d'Europe, Bordeaux-Mérignac, Toulouse-Labège -- `helpers/landuse_aois.py`) -- 793 → 1097 tuiles. **Piège rencontré et documenté dans le code** : plusieurs AOI candidates initiales (Lille-Englos, Marseille Plan-de-Campagne, Vélizy, Bordeaux-Lac) débordaient du bord de leur tuile Sentinel-2/MGRS, lisant silencieusement une fraction de l'AOI au lieu de sa totalité (même piège que Le Havre/Fos-sur-Mer au premier essai) -- détecté en comparant la forme de lecture réelle (`rasterio` `.read()`) à la forme nominale de la fenêtre, corrigé en substituant des AOI voisines vérifiées une par une.
7. **Early stopping sur `val_mean_iou`** (`patience`, `SegEngine.train_model`) plutôt qu'un nombre d'epochs fixe -- nécessaire une fois le dataset plus diversifié : le plafond de 20 epochs qui suffisait sur un dataset homogène (cluster Nord-Pas-de-Calais) devenait insuffisant, le val IoU n'ayant pas fini d'osciller/converger à l'epoch 20 sur un dataset plus hétérogène.
8. **Relations OSM `multipolygon`** pour Parking uniquement (§2) -- gros parkings mappés avec anneau extérieur + trous (îlots), invisibles pour un fetch way-only. A mesurablement augmenté le volume de pixels Parking (poids de classe 85.6 → 72.0 → 57.3 au fil des ajouts 6+8), les 4 AOI ciblées portant à elles seules ~48% du total des pixels Parking du dataset pour ~26% des tuiles -- la stratégie de ciblage a fonctionné pour produire plus de signal Parking en amont.

### Conclusion sur Parking

**Malgré tous les leviers ci-dessus, l'IoU Parking n'a jamais dépassé ~0.03 sur aucun run, et reste le plus souvent entre 0.005 et 0.02** -- y compris le run avec le plus de pixels Parking jamais atteint (poids de classe le plus bas, 57.3). Ce n'est plus attribuable à l'overfitting, au déséquilibre de classe non corrigé, ou au volume de données : ces trois causes ont été adressées et l'IoU n'a pas bougé de façon significative pour autant.

Diagnostic retenu : **plafond structurel de résolution**, pas un problème de réglage. Un parking de 1500m² (seuil loi APER, le minimum ciblé) ne fait que ~15 pixels à 10m/pixel (Sentinel-2) -- proche du plancher de ce qu'un U-Net avec encodeur ResNet34 (downsampling ×32 au niveau le plus profond) peut apprendre à localiser de façon fiable, quel que soit le volume/la diversité des exemples fournis.

**Recommandation** : ne pas continuer à itérer sur les leviers données/entraînement pour Parking spécifiquement -- rendement décroissant démontré empiriquement. Pistes qui *pourraient* encore aider mais changent significativement le scope, non tentées :
- Sur-échantillonnage des tuiles à Parking (item 5 du §6) -- rendement incertain, le problème semble être la taille de l'objet plus que sa fréquence d'apparition dans le train set.
- Une source d'imagerie à résolution plus fine que Sentinel-2 (10m/pixel) -- changerait toute la pipeline de fetch, hors scope de simples ajustements du pipeline actuel.
- Accepter la limite et concentrer l'évaluation/la présentation du projet sur les 4 autres classes (Fond, Industriel/commercial, Friche, Residentiel), qui répondent bien aux améliorations listées ci-dessus.

### Effet de bord découvert : Friche diluée par les nouvelles AOI

Les 4 AOI ajoutées au point 6 (toutes hors Hauts-de-France) remontent **`0 friche(s)`** de Cartofriches (voir §2) -- probablement un vrai trou de couverture de cette source hors Hauts-de-France, pas un bug. Conséquence : le poids de classe Friche est remonté (27.5 → 33.9) au lieu de continuer à baisser, ces 280 tuiles sans friche diluant sa représentation relative. À surveiller si le dataset s'étend encore hors Hauts-de-France : Friche a besoin d'une autre source de données (ou d'AOI dans des départements où Cartofriches est confirmé riche) pour continuer à s'améliorer en parallèle de Parking/Industriel.
