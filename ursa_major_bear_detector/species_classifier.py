import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
import cv2
import numpy as np
from pathlib import Path

class BearSpeciesClassifier:
    """Advanced bear species classification using deep learning"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = [
            'American Black Bear',
            'Brown/Grizzly Bear', 
            'Polar Bear',
            'Asiatic Black Bear',
            'Sloth Bear',
            'Sun Bear',
            'Not a Bear'
        ]
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.model = self._build_model()
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            print("Using pre-trained ResNet-50 features for classification")
    
    def _build_model(self):
        """Build the classification model"""
        model = resnet50(pretrained=True)
        
        # Modify final layer for bear species classification
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(self.classes))
        )
        
        model = model.to(self.device)
        return model
    
    def load_model(self, model_path):
        """Load trained model weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def classify(self, image_roi):
        """Classify bear species from image ROI"""
        try:
            if image_roi.size == 0:
                return "Unknown", 0.0
            
            # Preprocess image
            input_tensor = self.transform(image_roi).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
                
                predicted_class = self.classes[predicted_idx.item()]
                confidence_score = confidence.item()
                
                return predicted_class, confidence_score
                
        except Exception as e:
            print(f"Classification error: {e}")
            return "Unknown", 0.0
    
    def get_top_predictions(self, image_roi, top_k=3):
        """Get top K predictions with confidence scores"""
        try:
            if image_roi.size == 0:
                return []
            
            input_tensor = self.transform(image_roi).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                top_k_prob, top_k_idx = torch.topk(probabilities, top_k)
                
                predictions = []
                for i in range(top_k):
                    class_name = self.classes[top_k_idx[i].item()]
                    confidence = top_k_prob[i].item()
                    predictions.append((class_name, confidence))
                
                return predictions
                
        except Exception as e:
            print(f"Top-K classification error: {e}")
            return []

class AdvancedBearAnalyzer:
    """Advanced bear analysis with multiple detection methods"""
    
    def __init__(self):
        self.species_classifier = BearSpeciesClassifier()
        
    def analyze_bear_features(self, image_roi):
        """Analyze bear features for classification"""
        features = {}
        
        try:
            if image_roi.size == 0:
                return features
            
            # Color analysis
            hsv = cv2.cvtColor(image_roi, cv2.COLOR_BGR2HSV)
            mean_hue = np.mean(hsv[:, :, 0])
            mean_saturation = np.mean(hsv[:, :, 1])
            mean_value = np.mean(hsv[:, :, 2])
            
            features['color'] = {
                'hue': mean_hue,
                'saturation': mean_saturation,
                'brightness': mean_value
            }
            
            # Size analysis
            height, width = image_roi.shape[:2]
            aspect_ratio = width / height if height > 0 else 0
            
            features['size'] = {
                'width': width,
                'height': height,
                'aspect_ratio': aspect_ratio
            }
            
            # Texture analysis (simplified)
            gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
            # Calculate standard deviation as texture measure
            texture_measure = np.std(gray)
            
            features['texture'] = texture_measure
            
            return features
            
        except Exception as e:
            print(f"Feature analysis error: {e}")
            return features
    
    def comprehensive_classification(self, image_roi):
        """Perform comprehensive bear classification"""
        results = {
            'primary_classification': None,
            'confidence': 0.0,
            'alternative_predictions': [],
            'features': {},
            'analysis_notes': []
        }
        
        try:
            # Primary species classification
            species, confidence = self.species_classifier.classify(image_roi)
            results['primary_classification'] = species
            results['confidence'] = confidence
            
            # Get alternative predictions
            top_predictions = self.species_classifier.get_top_predictions(image_roi, top_k=3)
            results['alternative_predictions'] = top_predictions
            
            # Feature analysis
            features = self.analyze_bear_features(image_roi)
            results['features'] = features
            
            # Analysis notes based on features
            if 'color' in features:
                brightness = features['color']['brightness']
                if brightness > 180:
                    results['analysis_notes'].append("High brightness suggests possible polar bear")
                elif brightness < 60:
                    results['analysis_notes'].append("Low brightness suggests black bear coloration")
            
            if 'size' in features:
                aspect_ratio = features['size']['aspect_ratio']
                if aspect_ratio > 1.5:
                    results['analysis_notes'].append("Wide aspect ratio suggests side view")
                
        except Exception as e:
            results['analysis_notes'].append(f"Analysis error: {e}")
        
        return results