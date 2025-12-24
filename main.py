import cv2
import numpy as np
import time

class WebcamMasks:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        
        # Chargement des cascades de Haar
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Chargement des images
        self.cap_img = cv2.imread('img/cap.png', cv2.IMREAD_UNCHANGED)
        self.beard_img = cv2.imread('img/bird.png', cv2.IMREAD_UNCHANGED)
        self.glasses_img = cv2.imread('img/glas.png', cv2.IMREAD_UNCHANGED)
        
        # Vérifier si les images sont chargées
        if self.cap_img is None:
            print("ATTENTION: cap.png non trouvé!")
        if self.beard_img is None:
            print("ATTENTION: bird.png non trouvé!")
        if self.glasses_img is None:
            print("ATTENTION: glas.png non trouvé!")
        
        # États des filtres
        self.filter_sepia = False
        self.filter_contrast = False
        self.show_cap = False
        self.show_glasses = False
        self.show_beard = False
        
        # Objets animés
        self.falling_objects = []
        self.smile_objects = []
        self.last_smile_time = 0
        
    def apply_sepia(self, img):
        """Applique un filtre sépia"""
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        sepia_img = cv2.transform(img, kernel)
        return np.clip(sepia_img, 0, 255).astype(np.uint8)
    
    def adjust_contrast(self, img, alpha=1.5):
        """Modifie le contraste"""
        return cv2.convertScaleAbs(img, alpha=alpha, beta=0)
    
    def overlay_image(self, background, overlay, x, y, target_width=None, target_height=None):
        """Superpose une image avec redimensionnement et canal alpha"""
        if overlay is None:
            return background
        
        # Redimensionner si nécessaire
        if target_width is not None or target_height is not None:
            if target_width is not None and target_height is not None:
                overlay = cv2.resize(overlay, (target_width, target_height))
            elif target_width is not None:
                ratio = target_width / overlay.shape[1]
                target_height = int(overlay.shape[0] * ratio)
                overlay = cv2.resize(overlay, (target_width, target_height))
            elif target_height is not None:
                ratio = target_height / overlay.shape[0]
                target_width = int(overlay.shape[1] * ratio)
                overlay = cv2.resize(overlay, (target_width, target_height))
        
        h, w = overlay.shape[:2]
        
        # Vérifier les limites
        if x + w > background.shape[1]:
            w = background.shape[1] - x
            overlay = overlay[:, :w]
        if y + h > background.shape[0]:
            h = background.shape[0] - y
            overlay = overlay[:h, :]
        if x < 0:
            overlay = overlay[:, -x:]
            w = overlay.shape[1]
            x = 0
        if y < 0:
            overlay = overlay[-y:, :]
            h = overlay.shape[0]
            y = 0
        
        if h <= 0 or w <= 0:
            return background
        
        # Gérer la transparence
        if overlay.shape[2] == 4:
            alpha = overlay[:, :, 3] / 255.0
            for c in range(3):
                background[y:y+h, x:x+w, c] = (alpha * overlay[:, :, c] + 
                                              (1 - alpha) * background[y:y+h, x:x+w, c])
        else:
            # Si pas de canal alpha, convertir en BGR
            if len(overlay.shape) == 2:
                overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
            background[y:y+h, x:x+w] = overlay
            
        return background
    
    def create_falling_object(self, frame_width):
        """Crée un objet qui tombe (flocon de neige)"""
        return {
            'x': np.random.randint(0, frame_width),
            'y': 0,
            'speed': np.random.randint(2, 6),
            'size': np.random.randint(10, 30),
            'color': (255, 255, 255),
            'intersecting': False
        }
    
    def draw_snowflake(self, img, x, y, size, color):
        """Dessine un flocon de neige"""
        cv2.line(img, (x, y-size), (x, y+size), color, 2)
        cv2.line(img, (x-size, y), (x+size, y), color, 2)
        cv2.line(img, (x-int(size*0.7), y-int(size*0.7)), 
                 (x+int(size*0.7), y+int(size*0.7)), color, 2)
        cv2.line(img, (x-int(size*0.7), y+int(size*0.7)), 
                 (x+int(size*0.7), y-int(size*0.7)), color, 2)
    
    def check_intersection(self, obj_x, obj_y, obj_size, face_x, face_y, face_w, face_h):
        """Vérifie l'intersection entre un objet et un visage"""
        return (face_x < obj_x < face_x + face_w and 
                face_y < obj_y < face_y + face_h)
    
    def draw_rounded_rectangle(self, img, pt1, pt2, color, thickness, radius):
        """Dessine un rectangle avec coins arrondis"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Lignes horizontales
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        
        # Lignes verticales
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # Coins arrondis
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    
    def draw_menu(self, frame):
        """Dessine le menu interactif stylé et professionnel"""
        menu_height = 250
        menu_width = frame.shape[1]
        menu = np.zeros((menu_height, menu_width, 3), dtype=np.uint8)
        
        # Gradient de fond (bleu foncé vers noir)
        for i in range(menu_height):
            intensity = int(40 * (1 - i / menu_height))
            menu[i, :] = (intensity + 20, intensity + 10, intensity)
        
        # Titre principal avec effet de barre
        title_height = 50
        cv2.rectangle(menu, (0, 0), (menu_width, title_height), (80, 50, 20), -1)
        cv2.rectangle(menu, (0, title_height-3), (menu_width, title_height), (120, 80, 30), -1)
        
        cv2.putText(menu, "MASQUES WEBCAM - THEME NOEL", (menu_width//2 - 280, 32), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        
        # Section des contrôles
        y_start = 80
        col1_x = 30
        col2_x = menu_width // 2 + 20
        
        # Colonne 1 - Filtres
        cv2.putText(menu, "FILTRES", (col1_x, y_start), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        
        y_offset = y_start + 35
        
        # Filtre Sépia
        status_sepia = "ON " if self.filter_sepia else "OFF"
        color_sepia = (100, 255, 100) if self.filter_sepia else (100, 100, 100)
        box_color_sepia = (50, 150, 50) if self.filter_sepia else (50, 50, 50)
        
        cv2.rectangle(menu, (col1_x, y_offset - 20), (col1_x + 250, y_offset + 5), box_color_sepia, -1)
        self.draw_rounded_rectangle(menu, (col1_x, y_offset - 20), (col1_x + 250, y_offset + 5), (80, 80, 80), 2, 8)
        cv2.putText(menu, "[S]", (col1_x + 10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 2)
        cv2.putText(menu, "Sepia", (col1_x + 60, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(menu, status_sepia, (col1_x + 200, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_sepia, 2)
        
        y_offset += 35
        
        # Filtre Contraste
        status_contrast = "ON " if self.filter_contrast else "OFF"
        color_contrast = (100, 255, 100) if self.filter_contrast else (100, 100, 100)
        box_color_contrast = (50, 150, 50) if self.filter_contrast else (50, 50, 50)
        
        cv2.rectangle(menu, (col1_x, y_offset - 20), (col1_x + 250, y_offset + 5), box_color_contrast, -1)
        self.draw_rounded_rectangle(menu, (col1_x, y_offset - 20), (col1_x + 250, y_offset + 5), (80, 80, 80), 2, 8)
        cv2.putText(menu, "[C]", (col1_x + 10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 2)
        cv2.putText(menu, "Contraste", (col1_x + 60, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(menu, status_contrast, (col1_x + 200, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_contrast, 2)
        
        # Colonne 2 - Accessoires
        cv2.putText(menu, "ACCESSOIRES", (col2_x, y_start), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        
        y_offset = y_start + 35
        
        # Casquette
        status_cap = "ON " if self.show_cap else "OFF"
        color_cap = (100, 255, 100) if self.show_cap else (100, 100, 100)
        box_color_cap = (50, 150, 50) if self.show_cap else (50, 50, 50)
        
        cv2.rectangle(menu, (col2_x, y_offset - 20), (col2_x + 250, y_offset + 5), box_color_cap, -1)
        self.draw_rounded_rectangle(menu, (col2_x, y_offset - 20), (col2_x + 250, y_offset + 5), (80, 80, 80), 2, 8)
        cv2.putText(menu, "[H]", (col2_x + 10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 2)
        cv2.putText(menu, "Casquette", (col2_x + 60, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(menu, status_cap, (col2_x + 200, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_cap, 2)
        
        y_offset += 35
        
        # Lunettes
        status_glasses = "ON " if self.show_glasses else "OFF"
        color_glasses = (100, 255, 100) if self.show_glasses else (100, 100, 100)
        box_color_glasses = (50, 150, 50) if self.show_glasses else (50, 50, 50)
        
        cv2.rectangle(menu, (col2_x, y_offset - 20), (col2_x + 250, y_offset + 5), box_color_glasses, -1)
        self.draw_rounded_rectangle(menu, (col2_x, y_offset - 20), (col2_x + 250, y_offset + 5), (80, 80, 80), 2, 8)
        cv2.putText(menu, "[G]", (col2_x + 10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 2)
        cv2.putText(menu, "Lunettes", (col2_x + 60, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(menu, status_glasses, (col2_x + 200, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_glasses, 2)
        
        y_offset += 35
        
        # Barbe
        status_beard = "ON " if self.show_beard else "OFF"
        color_beard = (100, 255, 100) if self.show_beard else (100, 100, 100)
        box_color_beard = (50, 150, 50) if self.show_beard else (50, 50, 50)
        
        cv2.rectangle(menu, (col2_x, y_offset - 20), (col2_x + 250, y_offset + 5), box_color_beard, -1)
        self.draw_rounded_rectangle(menu, (col2_x, y_offset - 20), (col2_x + 250, y_offset + 5), (80, 80, 80), 2, 8)
        cv2.putText(menu, "[B]", (col2_x + 10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 2)
        cv2.putText(menu, "Barbe Noel", (col2_x + 60, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(menu, status_beard, (col2_x + 200, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_beard, 2)
        
        # Ligne de séparation
        cv2.line(menu, (20, menu_height - 35), (menu_width - 20, menu_height - 35), (80, 80, 80), 2)
        
        # Bouton Quitter
        quit_y = menu_height - 20
        cv2.putText(menu, "[Q] QUITTER", (menu_width//2 - 80, quit_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
        
        return menu
    
    def run(self):
        """Boucle principale"""
        print("=" * 60)
        print("🎅 MASQUES WEBCAM - THEME NOEL 🎄")
        print("=" * 60)
        print("\n📸 Démarrage de la webcam...")
        print("\n⌨️  CONTROLES:")
        print("   [H] - Casquette Père Noël")
        print("   [G] - Lunettes")
        print("   [B] - Barbe de Père Noël")
        print("   [S] - Filtre Sépia")
        print("   [C] - Contraste amélioré")
        print("   [Q] - Quitter\n")
        print("=" * 60)
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Ajout d'objets tombants toutes les 30 frames
            if frame_count % 30 == 0:
                self.falling_objects.append(self.create_falling_object(frame.shape[1]))
            
            # Détection des visages
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Application des filtres globaux
            if self.filter_sepia:
                frame = self.apply_sepia(frame)
            
            if self.filter_contrast:
                frame = self.adjust_contrast(frame)
            
            # Traitement de chaque visage
            for (x, y, w, h) in faces:
                # ========== CASQUETTE (en haut du visage) ==========
                if self.show_cap and self.cap_img is not None:
                 
                    cap_width = int(w * 2.1)  # Plus large pour couvrir la tête
                    cap_y = y - int(h * 0.85)  # Au-dessus de la tête
                    cap_x = x - int((cap_width - w) / 3)  # Centré
                    frame = self.overlay_image(frame, self.cap_img, cap_x, cap_y, target_width=cap_width)
                
                # ========== LUNETTES (MODIFIÉ POUR REGLAGE Y) ==========
                if self.show_glasses and self.glasses_img is not None:
                    # --- PARAMETRES DE REGLAGE ---
                    # 1. Hauteur par défaut (si les yeux ne sont PAS trouvés)
                    # 0.25 = niveau yeux (plus petit = plus haut, plus grand = plus bas)
                    hauteur_standard = -0.23
                    
                    # 2. Ajustement fin (si les yeux SONT trouvés)
                    # 0.15 = on remonte un peu au dessus du centre des yeux
                    ajustement_yeux = 0.15
                    # ------------------------------
                    
                    roi_gray_eyes = gray[y:y+h, x:x+w]
                    eyes = self.eye_cascade.detectMultiScale(roi_gray_eyes, 1.1, 5)
                    
                    if len(eyes) >= 2:
                        # Positionner sur les yeux détectés
                        eyes_sorted = sorted(eyes, key=lambda e: e[0])
                        eye1, eye2 = eyes_sorted[0], eyes_sorted[1]
                        
                        # Calculer le centre vertical des yeux
                        eye_center_y = y + ((eye1[1] + eye1[3]//2) + (eye2[1] + eye2[3]//2)) // 2
                        
                        # Positionner les lunettes avec l'ajustement
                        glasses_y = eye_center_y - int(h * ajustement_yeux)
                    else:
                        # Fallback : position manuelle basée sur hauteur_standard
                        glasses_y = y + int(h * hauteur_standard)
                    
                    glasses_width = int(w * 1.35)
                    glasses_x = x - int((glasses_width - w) / 2)
                    frame = self.overlay_image(frame, self.glasses_img, glasses_x, glasses_y, target_width=glasses_width)
                
                if self.show_beard and self.beard_img is not None:
                    # --- REGLAGE HAUTEUR BARBE ---
                    # 0.55 = Niveau standard (bouche/menton)
                    # Pour MONTER la barbe : diminue ce chiffre (ex: 0.40, 0.30 ou même négatif si besoin)
                    # Pour DESCENDRE la barbe : augmente ce chiffre (ex: 0.70, 0.80)
                    hauteur_barbe = 0.24
                    # -----------------------------

                    beard_width = int(w * 1.3) 
                    
                    # Calcul de la position Y avec ton réglage
                    beard_y = y + int(h * hauteur_barbe) 
                    
                    beard_x = x - int((beard_width - w) / 2)
                    frame = self.overlay_image(frame, self.beard_img, beard_x, beard_y, target_width=beard_width)
                
                # ========== DÉTECTION DU SOURIRE ==========
                roi_gray = gray[y:y+h, x:x+w]
                smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
                
                current_time = time.time()
                if len(smiles) > 0 and current_time - self.last_smile_time > 0.5:
                    self.last_smile_time = current_time
                    # Créer des étoiles dorées qui apparaissent
                    for _ in range(5):
                        self.smile_objects.append({
                            'x': x + w//2 + np.random.randint(-80, 80),
                            'y': y + h//2 + np.random.randint(-40, 40),
                            'lifetime': 40,
                            'size': np.random.randint(12, 20),
                            'angle': np.random.randint(0, 360)
                        })
            
            # ========== ANIMATION DES FLOCONS DE NEIGE ==========
            objects_to_remove = []
            for i, obj in enumerate(self.falling_objects):
                obj['y'] += obj['speed']
                
                # Vérifier l'intersection avec les visages
                obj['intersecting'] = False
                for (fx, fy, fw, fh) in faces:
                    if self.check_intersection(obj['x'], obj['y'], obj['size'], fx, fy, fw, fh):
                        obj['intersecting'] = True
                        break
                
                # Changer la couleur si intersection (vert pour Noël)
                color = (0, 255, 0) if obj['intersecting'] else obj['color']
                
                # Dessiner le flocon
                self.draw_snowflake(frame, obj['x'], obj['y'], obj['size'], color)
                
                # Retirer si hors écran
                if obj['y'] > frame.shape[0]:
                    objects_to_remove.append(i)
            
            # Nettoyer les objets tombés
            for i in reversed(objects_to_remove):
                self.falling_objects.pop(i)
            
            # ========== ANIMATION DES ÉTOILES (sourire) ==========
            smile_objects_to_remove = []
            for i, obj in enumerate(self.smile_objects):
                obj['lifetime'] -= 1
                obj['angle'] += 5  # Rotation
                
                # Dessiner une étoile dorée rotative
                pts = []
                for j in range(5):
                    angle = j * 2 * np.pi / 5 - np.pi / 2 + np.radians(obj['angle'])
                    pts.append([
                        int(obj['x'] + obj['size'] * np.cos(angle)),
                        int(obj['y'] + obj['size'] * np.sin(angle))
                    ])
                pts = np.array(pts, np.int32)
                
                # Couleur dorée de Noël
                alpha_factor = obj['lifetime'] / 40.0
                color = (0, int(215 * alpha_factor), int(255 * alpha_factor))
                cv2.fillPoly(frame, [pts], color)
                
                if obj['lifetime'] <= 0:
                    smile_objects_to_remove.append(i)
            
            for i in reversed(smile_objects_to_remove):
                self.smile_objects.pop(i)
            
            # Dessiner le menu
            menu = self.draw_menu(frame)
            
            # Combiner frame et menu
            combined = np.vstack([frame, menu])
            
            cv2.imshow('Projet Traitement d\'Images - Theme Noel', combined)
            
            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Au revoir et Joyeux Noël! 🎄")
                break
            elif key == ord('s'):
                self.filter_sepia = not self.filter_sepia
                print(f"✨ Filtre Sépia: {'ON' if self.filter_sepia else 'OFF'}")
            elif key == ord('c'):
                self.filter_contrast = not self.filter_contrast
                print(f"✨ Contraste: {'ON' if self.filter_contrast else 'OFF'}")
            elif key == ord('h'):
                self.show_cap = not self.show_cap
                print(f"🎅 Casquette: {'ON' if self.show_cap else 'OFF'}")
            elif key == ord('g'):
                self.show_glasses = not self.show_glasses
                print(f"👓 Lunettes: {'ON' if self.show_glasses else 'OFF'}")
            elif key == ord('b'):
                self.show_beard = not self.show_beard
                print(f"🎅 Barbe: {'ON' if self.show_beard else 'OFF'}")
            
            frame_count += 1
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = WebcamMasks()
    app.run()