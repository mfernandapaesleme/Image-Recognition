import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

def evaluate_matches(pts1, pts2, matches, M, threshold=3.0):
    """Évalue la qualité des appariements en utilisant la transformation connue"""
    correct_matches = 0
    distances = []
    
    for match in matches:
        # Points appariés
        pt1 = pts1[match.queryIdx].pt
        pt2 = pts2[match.trainIdx].pt
        
        # Position attendue après transformation
        expected_pt = np.dot(M, np.array([pt1[0], pt1[1], 1]))
        
        # Distance entre position réelle et attendue
        distance = np.linalg.norm(np.array(pt2) - expected_pt[:2])
        distances.append(distance)
        
        if distance < threshold:
            correct_matches += 1
    
    inlier_rate = correct_matches / len(matches) if matches else 0
    mean_error = np.mean(distances) if distances else float('inf')
    
    return inlier_rate, mean_error, correct_matches

if len(sys.argv) != 2:
    print("Usage :", sys.argv[0], "detector(= orb ou kaze)")
    sys.exit(2)
    
detector = sys.argv[1].lower()
if detector not in ['orb', 'kaze']:
    print("Usage :", sys.argv[0], "detector(= orb ou kaze)")
    sys.exit(2)

# Lecture de l'image originale
img1 = cv2.imread('./images/torb_small1.png')
print("Dimension de l'image 1 :", img1.shape[0], "lignes x", img1.shape[1], "colonnes")
print("Type de l'image 1 :", img1.dtype)

# Définition de la transformation géométrique connue (rotation + translation)
angle = 30  # degrés
tx, ty = 20, 10  # pixels
M = cv2.getRotationMatrix2D((img1.shape[1]/2, img1.shape[0]/2), angle, 1.0)
M[0, 2] += tx  # Ajout de la translation en x
M[1, 2] += ty  # Ajout de la translation en y

# Application de la transformation
img2 = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))

# Initialisation du détecteur
t1 = cv2.getTickCount()
if detector == 'orb':
    kp = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
    print("Détecteur : ORB")
    norm_type = cv2.NORM_HAMMING
else:  # 'kaze'
    kp = cv2.KAZE_create(upright=False, threshold=0.001, 
                         nOctaves=4, nOctaveLayers=4, diffusivity=2)
    print("Détecteur : KAZE")
    norm_type = cv2.NORM_L2

# Conversion en niveaux de gris
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Détection et description des points-clés
pts1, desc1 = kp.detectAndCompute(gray1, None)
pts2, desc2 = kp.detectAndCompute(gray2, None)

# Visualisation des points non appariés
img_kp1 = cv2.drawKeypoints(gray1, pts1, None, color=(127,127,127), flags=0)
img_kp2 = cv2.drawKeypoints(gray2, pts2, None, color=(127,127,127), flags=0)

t2 = cv2.getTickCount()
detection_time = (t2 - t1) / cv2.getTickFrequency()
print("Détection points et calcul descripteurs :", detection_time, "s")

# Méthodes d'appariement à évaluer
methods = [
    ('CrossCheck', cv2.BFMatcher(norm_type, crossCheck=True), None),
    ('RatioTest', cv2.BFMatcher(norm_type, crossCheck=False), 0.7),
    ('FLANN', cv2.FlannBasedMatcher() if detector == 'kaze' else cv2.BFMatcher(norm_type, crossCheck=False), 0.7)
]

results = []

for method_name, matcher, ratio_threshold in methods:
    t1 = cv2.getTickCount()
    
    if method_name == 'CrossCheck':
        # Appariement avec Cross-Check
        matches = matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:200]  # On prend les 200 meilleurs
    else:
        # Appariement avec kNN (pour RatioTest et FLANN)
        if method_name == 'FLANN' and detector == 'kaze':
            matches = matcher.knnMatch(desc1, desc2, k=2)
        else:
            matches = matcher.knnMatch(desc1, desc2, k=2)
        
        # Application du ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
        
        # Tri par distance
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:200]
    
    t2 = cv2.getTickCount()
    matching_time = (t2 - t1) / cv2.getTickFrequency()
    
    # Évaluation quantitative
    inlier_rate, mean_error, num_correct = evaluate_matches(pts1, pts2, good_matches, M)
    
    # Stockage des résultats
    results.append({
        'method': method_name,
        'inlier_rate': inlier_rate,
        'mean_error': mean_error,
        'num_matches': len(good_matches),
        'time': matching_time
    })
    
    # Visualisation
    img_matches = cv2.drawMatches(img_kp1, pts1, img_kp2, pts2, good_matches[:100], None,
                                 matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure(figsize=(15, 5))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    title = f"{method_name} - {len(good_matches)} appariements\n"
    title += f"Inliers: {inlier_rate*100:.1f}% - Erreur: {mean_error:.2f}px - Temps: {matching_time:.3f}s"
    plt.title(title)
    plt.axis('off')
    plt.show()

# Affichage des résultats comparatifs
print("\nRésultats comparatifs:")
print("{:<10} {:<12} {:<12} {:<12} {:<10}".format(
    "Méthode", "Inliers (%)", "Erreur (px)", "Matches", "Temps (s)"))
for res in results:
    print("{:<10} {:<12.1f} {:<12.2f} {:<12} {:<10.3f}".format(
        res['method'], res['inlier_rate']*100, res['mean_error'], 
        res['num_matches'], res['time']))