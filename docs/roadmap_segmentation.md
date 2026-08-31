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

### Cartofriches (Cerema)
- **Source actuelle (depuis le second essai, §8) : extrait national statique**, GeoPackage téléchargé une fois et mis en cache sous `data/cartofriches/` (`helpers/geo_fetch.py` : `CARTOFRICHES_GPKG_URL`, `_ensure_cartofriches_gpkg`), interrogé localement via l'index spatial R-tree du GeoPackage (`fetch_cartofriches_polygons(bbox)`, plus besoin de préfixe INSEE de département). Source : dataset data.gouv.fr "Sites référencés dans Cartofriches" (~36 000 sites, mis à jour régulièrement).
- **Ancienne source, abandonnée** : WFS `https://www.geo2france.fr/geoserver/cerema/ows`, couche `cerema:cartofriche`. **Bug découvert au second essai** : ce GeoServer ne mirror que les données Hauts-de-France (confirmé empiriquement en listant les préfixes de département présents : uniquement 02/59/60/62/80) alors que Cartofriches est un jeu de données national -- donc toute AOI hors 59/62 remontait silencieusement `0 friche(s)`, non pas par absence réelle de friches mais parce que ce serveur ne les avait jamais eues. Vérifié après correction : Bouches-du-Rhône (13) 687 sites, Gironde (33) 311, Haute-Garonne (31) 206, Seine-et-Marne (77) 150 dans l'extrait national -- les AOI ajoutées pour Parking (§8) ont maintenant de vraies friches (ex. Vitrolles : 121, contre 0 avant).
- Format du GeoPackage : géométries `Polygon`/`MultiPolygon` 2D en EPSG:4326, parsées à la main depuis le blob binaire GPB (`_parse_gpkg_polygon`, `struct` stdlib uniquement, pas de nouvelle dépendance -- cohérent avec le choix "pas de shapely/geopandas" ci-dessous). Seuls les anneaux extérieurs sont gardés (mêmes trous ignorés que pour les polygones OSM).
- Couverture confirmée : 5557 friches en Hauts-de-France, 1401 dans le Nord (59), 42 dans une AOI élargie autour de Dunkerque (chiffres identiques entre ancienne et nouvelle source pour cette région, bon signe de cohérence).

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
9. **Définition de Parking élargie à l'entraînement, testée, puis annulée (revert)** -- `rasterize_landuse_mask`'s `min_parking_area_m2` a brièvement valu `0` par défaut (`helpers/mask_rasterize.py`) : tous les parkings, petits ou grands, étiquetés Parking dans les masques d'entraînement, avec le seuil loi APER appliqué en post-traitement sur la prédiction à la place (`helpers/mask_postprocess.filter_small_regions`, composantes connexes via `scipy.ndimage.label`, branché dans `api/segmentation_inference.py` -- ce module est conservé, voir résultat ci-dessous). Hypothèse de départ : un petit parking a la même signature visuelle qu'un grand, donc le filtrer à la génération des masques ne fait que jeter des exemples pour la classe déjà la plus rare. **Mesuré, infirmé** : sur le même dataset (1097 tuiles), même en comparant à définition égale (Parking "toute taille"), l'IoU test est descendu de 0.0157 à 0.0110 -- pire, pas meilleur -- et Friche (0.204→0.062) et Fond (0.791→0.632) ont aussi nettement reculé. Diagnostic : l'hypothèse de départ était fausse à cette résolution -- un petit parking (souvent <100m², donc plus petit qu'un seul pixel Sentinel-2 de 10x10m) n'a pas un cœur de pixels "propre" comme un grand ; son pixel est dominé par ce qui l'entoure (jardin, toit, bord d'allée), donc l'étiqueter Parking ajoute du bruit d'étiquetage plutôt que du signal utile. **Revert appliqué** : `min_parking_area_m2` est revenu à `MIN_PARKING_AREA_M2` (1500) par défaut. `helpers/mask_postprocess.filter_small_regions` reste branché comme filet de sécurité (attrape les rares faux positifs de petite taille que le modèle prédit malgré tout), même si ce n'est plus le mécanisme de filtrage principal.

### Conclusion sur Parking

**Malgré tous les leviers 1 à 9 ci-dessus, l'IoU Parking n'a jamais dépassé ~0.03 sur aucun run, et reste le plus souvent entre 0.005 et 0.02.** Ce n'est plus attribuable à l'overfitting, au déséquilibre de classe non corrigé, au volume de données, ou à une définition trop restrictive : ces quatre causes ont été adressées et l'IoU n'a pas bougé de façon significative pour autant -- l'item 9 a même montré qu'élargir la définition empire les choses.

Diagnostic retenu : **plafond structurel de résolution**, pas un problème de réglage. Un parking de 1500m² (seuil loi APER, le minimum ciblé) ne fait que ~15 pixels à 10m/pixel (Sentinel-2) -- proche du plancher de ce qu'un U-Net avec encodeur ResNet34 (downsampling ×32 au niveau le plus profond) peut apprendre à localiser de façon fiable, quel que soit le volume/la diversité des exemples fournis.

**Recommandation** : ne pas continuer à itérer sur les leviers données/entraînement pour Parking spécifiquement -- rendement décroissant démontré empiriquement, y compris pour une piste (item 9) qui semblait pourtant raisonnable a priori. Pistes qui *pourraient* encore aider mais changent significativement le scope :
- **Seuil par classe à l'inférence plutôt qu'un argmax brut** -- non tenté : appliquer un seuil de probabilité bas spécifiquement pour Parking (ex. prédire Parking si `P(Parking) > 0.15` même si Fond a une probabilité plus haute) plutôt que de toujours prendre la classe la plus probable. Ne change rien à l'entraînement, juste à la lecture des probabilités déjà calculées -- coût quasi nul, gain incertain (compromis rappel/précision à régler sur le val set).
- Sur-échantillonnage des tuiles à Parking (item 5 du §6) -- rendement incertain, le problème semble être la taille de l'objet plus que sa fréquence d'apparition dans le train set ; l'item 9 suggère que le signal lui-même (pas juste sa fréquence) est le facteur limitant.
- Une source d'imagerie à résolution plus fine que Sentinel-2 (10m/pixel) -- ex. BD ORTHO IGN (20cm/pixel, gratuite), changerait toute la pipeline de fetch, hors scope de simples ajustements du pipeline actuel. C'est le seul levier qui s'attaque vraiment à la cause racine (résolution physique) plutôt qu'à ses symptômes -- l'item 9 renforce cette conclusion : le problème est bien la résolution, pas la quantité ou la diversité des exemples.
- Accepter la limite et concentrer l'évaluation/la présentation du projet sur les 4 autres classes (Fond, Industriel/commercial, Friche, Residentiel), qui répondent bien aux améliorations listées ci-dessus.

### Effet de bord découvert puis corrigé : Friche diluée par les nouvelles AOI

Les 4 AOI ajoutées au point 6 (toutes hors Hauts-de-France) remontaient **`0 friche(s)`** de Cartofriches -- pas une vraie absence de friches, mais un bug de source de données : le WFS `geo2france.fr` utilisé jusque-là ne mirror que les données Hauts-de-France (voir §2, `helpers/geo_fetch.py`). Corrigé en passant à l'extrait national data.gouv.fr (téléchargé et mis en cache localement, interrogé par index spatial). Après correction, ces mêmes AOI ont de vraies friches (Vitrolles 121, Val d'Europe 32, Bordeaux-Mérignac 135, Toulouse-Labège 62) -- à re-mesurer si l'IoU Friche s'améliore avec un run d'entraînement sur le dataset regénéré.
